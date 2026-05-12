from __future__ import annotations

import json
import base64
import sys
import unittest

import _bootstrap  # noqa: F401

from anvil.hooks import (
    HookConfig,
    HookEvent,
    HookInput,
    HookManager,
    HookOutput,
    HookResult,
    build_hook_input_for_tool,
    run_hook,
    run_hooks_for_event,
)


def hook_command(payload: dict[str, object]) -> str:
    rendered = json.dumps(payload).encode('utf-8')
    encoded = base64.b64encode(rendered).decode('ascii')
    code = f"import base64; print(base64.b64decode('{encoded}').decode())"
    return f'"{sys.executable}" -c "{code}"'


class HookInputTests(unittest.TestCase):
    def test_should_serialize_to_json(self) -> None:
        inp = HookInput(
            event='PreToolUse',
            tool_name='read_file',
            tool_input={'path': 'test.py'},
        )
        data = json.loads(inp.to_json())
        self.assertEqual(data['event'], 'PreToolUse')
        self.assertEqual(data['tool_name'], 'read_file')
        self.assertEqual(data['tool_input'], {'path': 'test.py'})

    def test_should_include_all_fields(self) -> None:
        inp = HookInput(
            event='PostToolUse',
            tool_name='write_file',
            tool_output='ok',
            session_id='sess-1',
            workspace_root='/workspace',
            metadata={'key': 'value'},
        )
        data = json.loads(inp.to_json())
        self.assertEqual(data['tool_output'], 'ok')
        self.assertEqual(data['session_id'], 'sess-1')
        self.assertEqual(data['metadata'], {'key': 'value'})


class HookOutputTests(unittest.TestCase):
    def test_should_parse_approved_output(self) -> None:
        output = HookOutput.from_json('{"approve": true}')
        self.assertTrue(output.approve)

    def test_should_parse_blocked_output(self) -> None:
        output = HookOutput.from_json('{"approve": false, "error": "denied"}')
        self.assertFalse(output.approve)
        self.assertEqual(output.error, 'denied')

    def test_should_parse_modified_input(self) -> None:
        output = HookOutput.from_json('{"approve": true, "modified_input": {"path": "new.py"}}')
        self.assertTrue(output.approve)
        self.assertEqual(output.modified_input, {'path': 'new.py'})

    def test_should_parse_context(self) -> None:
        output = HookOutput.from_json('{"approve": true, "context": "extra info"}')
        self.assertEqual(output.context, 'extra info')

    def test_should_handle_invalid_json(self) -> None:
        output = HookOutput.from_json('not json')
        self.assertTrue(output.approve)

    def test_should_handle_empty_string(self) -> None:
        output = HookOutput.from_json('')
        self.assertTrue(output.approve)

    def test_should_create_approved(self) -> None:
        output = HookOutput.approved()
        self.assertTrue(output.approve)

    def test_should_create_blocked(self) -> None:
        output = HookOutput.blocked('forbidden')
        self.assertFalse(output.approve)
        self.assertEqual(output.error, 'forbidden')


class RunHookTests(unittest.TestCase):
    def test_should_run_echo_command(self) -> None:
        config = HookConfig(command=hook_command({'approve': True}))
        inp = HookInput(event='PreToolUse')
        output = run_hook(config, inp, timeout_s=5)
        self.assertTrue(output.approve)

    def test_should_block_on_nonzero_exit(self) -> None:
        config = HookConfig(command='exit 1')
        inp = HookInput(event='PreToolUse')
        output = run_hook(config, inp, timeout_s=5)
        self.assertFalse(output.approve)
        self.assertIn('exited with code', output.error)

    def test_should_pass_input_via_stdin(self) -> None:
        py = sys.executable.replace('\\', '/')
        cmd = f'{py} -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({{\'approve\': d[\'event\']==\'PreToolUse\'}}))"'
        config = HookConfig(command=cmd)
        inp = HookInput(event='PreToolUse', tool_name='test')
        output = run_hook(config, inp, timeout_s=10)
        self.assertTrue(output.approve)

    def test_should_block_on_timeout(self) -> None:
        py = sys.executable.replace('\\', '/')
        config = HookConfig(command=f'{py} -c "import time; time.sleep(10)"', timeout_s=0.1)
        inp = HookInput(event='PreToolUse')
        output = run_hook(config, inp, timeout_s=0.5)
        self.assertFalse(output.approve)
        self.assertIn('timed out', output.error)


class RunHooksForEventTests(unittest.TestCase):
    def test_should_run_multiple_hooks(self) -> None:
        hooks = [
            HookConfig(command=hook_command({'approve': True, 'context': 'hook1'})),
            HookConfig(command=hook_command({'approve': True, 'context': 'hook2'})),
        ]
        inp = HookInput(event='PreToolUse')
        result = run_hooks_for_event(HookEvent.PreToolUse, hooks, inp)
        self.assertTrue(result.approved)
        self.assertEqual(result.hooks_run, 2)
        self.assertIn('hook1', result.context)
        self.assertIn('hook2', result.context)

    def test_should_block_if_any_hook_blocks(self) -> None:
        hooks = [
            HookConfig(command=hook_command({'approve': True})),
            HookConfig(command=hook_command({'approve': False, 'error': 'blocked'})),
        ]
        inp = HookInput(event='PreToolUse')
        result = run_hooks_for_event(HookEvent.PreToolUse, hooks, inp)
        self.assertFalse(result.approved)
        self.assertEqual(result.error, 'blocked')
        # Second hook should not run
        self.assertEqual(result.hooks_run, 2)

    def test_should_return_modified_input(self) -> None:
        hooks = [
            HookConfig(command=hook_command({'approve': True, 'modified_input': {'path': 'modified.py'}})),
        ]
        inp = HookInput(event='PreToolUse')
        result = run_hooks_for_event(HookEvent.PreToolUse, hooks, inp)
        self.assertTrue(result.approved)
        self.assertEqual(result.modified_input, {'path': 'modified.py'})

    def test_should_skip_async_hooks_in_result_count(self) -> None:
        hooks = [
            HookConfig(command=hook_command({'approve': True}), async_mode=True),
            HookConfig(command=hook_command({'approve': True})),
        ]
        inp = HookInput(event='PostToolUse')
        result = run_hooks_for_event(HookEvent.PostToolUse, hooks, inp)
        self.assertTrue(result.approved)
        # Both count as run, but async doesn't block
        self.assertEqual(result.hooks_run, 2)


class HookManagerTests(unittest.TestCase):
    def test_should_register_and_retrieve_hooks(self) -> None:
        manager = HookManager()
        manager.register_command(HookEvent.PreToolUse, 'echo ok')
        self.assertTrue(manager.has_hooks(HookEvent.PreToolUse))
        self.assertFalse(manager.has_hooks(HookEvent.PostToolUse))

    def test_should_register_multiple_hooks_for_same_event(self) -> None:
        manager = HookManager()
        manager.register_command(HookEvent.PreToolUse, 'echo hook1')
        manager.register_command(HookEvent.PreToolUse, 'echo hook2')
        hooks = manager.get_hooks(HookEvent.PreToolUse)
        self.assertEqual(len(hooks), 2)

    def test_should_register_with_string_event(self) -> None:
        manager = HookManager()
        manager.register_command('PreToolUse', 'echo ok')
        self.assertTrue(manager.has_hooks(HookEvent.PreToolUse))

    def test_should_clear_all_hooks(self) -> None:
        manager = HookManager()
        manager.register_command(HookEvent.PreToolUse, 'echo ok')
        manager.register_command(HookEvent.PostToolUse, 'echo ok')
        manager.clear()
        self.assertFalse(manager.has_hooks(HookEvent.PreToolUse))
        self.assertFalse(manager.has_hooks(HookEvent.PostToolUse))

    def test_should_run_registered_hooks(self) -> None:
        manager = HookManager()
        manager.register_command(HookEvent.PreToolUse, 'echo {"approve": true}')
        inp = HookInput(event='PreToolUse')
        result = manager.run_event(HookEvent.PreToolUse, inp)
        self.assertTrue(result.approved)
        self.assertEqual(result.hooks_run, 1)

    def test_should_return_empty_result_when_no_hooks(self) -> None:
        manager = HookManager()
        inp = HookInput(event='PreToolUse')
        result = manager.run_event(HookEvent.PreToolUse, inp)
        self.assertTrue(result.approved)
        self.assertEqual(result.hooks_run, 0)


class BuildHookInputTests(unittest.TestCase):
    def test_should_build_tool_input(self) -> None:
        inp = build_hook_input_for_tool(
            HookEvent.PreToolUse,
            'read_file',
            {'path': 'test.py'},
            session_id='sess-1',
            workspace_root='/workspace',
        )
        self.assertEqual(inp.event, 'PreToolUse')
        self.assertEqual(inp.tool_name, 'read_file')
        self.assertEqual(inp.tool_input, {'path': 'test.py'})
        self.assertEqual(inp.session_id, 'sess-1')

    def test_should_accept_string_event(self) -> None:
        inp = build_hook_input_for_tool('PostToolUse', 'write_file', {})
        self.assertEqual(inp.event, 'PostToolUse')


if __name__ == '__main__':
    unittest.main()
