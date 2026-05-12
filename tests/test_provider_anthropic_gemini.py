from __future__ import annotations

import argparse
import os
import unittest

import _bootstrap  # noqa: F401

from anvil.llm.providers import (
    _extract_anthropic_text,
    _native_tool_prompt,
    _prompt_has_successful_tool_result,
    _prompt_requires_file_tool,
    _prompt_should_force_write_file,
    build_invoke_from_args,
)


class ProviderAnthropicTests(unittest.TestCase):
    def test_should_extract_anthropic_text_from_supported_response_shapes(self) -> None:
        self.assertEqual(
            _extract_anthropic_text({'content': [{'type': 'text', 'text': 'hello'}]}),
            'hello',
        )
        self.assertEqual(_extract_anthropic_text({'content': 'hello'}), 'hello')
        self.assertEqual(
            _extract_anthropic_text({'choices': [{'message': {'content': 'hello'}}]}),
            'hello',
        )
        self.assertEqual(
            _extract_anthropic_text({'choices': [{'message': {'content': [{'text': 'hello'}]}}]}),
            'hello',
        )
        self.assertEqual(
            _extract_anthropic_text({'output': [{'content': [{'type': 'output_text', 'text': 'hello'}]}]}),
            'hello',
        )
        self.assertEqual(_extract_anthropic_text({'delta': {'text': 'hello'}}), 'hello')
        self.assertEqual(_extract_anthropic_text({'content': [{'value': 'hello'}]}), 'hello')
        self.assertEqual(_extract_anthropic_text({'content': [{'answer': 'hello'}]}), 'hello')
        self.assertEqual(_extract_anthropic_text({'content': [{'thinking': 'hello'}]}), 'hello')
        self.assertEqual(_extract_anthropic_text({'content': [{'reasoning_content': 'hello'}]}), 'hello')

    def test_should_convert_anthropic_tool_use_blocks_to_agent_step_json(self) -> None:
        text = _extract_anthropic_text(
            {
                'content': [
                    {'type': 'text', 'text': 'create requested file'},
                    {
                        'type': 'tool_use',
                        'id': 'toolu_1',
                        'name': 'write_file',
                        'input': {'path': 'abc/current-path.json', 'content': 'D:\\workspace\\Anvil'},
                    },
                ]
            }
        )

        self.assertIn('"name": "write_file"', text)
        self.assertIn('"path": "abc/current-path.json"', text)
        self.assertIn('"final": null', text)

    def test_should_detect_file_goals_for_native_anthropic_tools(self) -> None:
        prompt = (
            'You are a coding agent.\n'
            'Goal:\n'
            '新增一个abc文件夹，并在abc新增一个空白的json文件，里面写当前路径\n'
            'History:\n'
            '[]'
        )

        self.assertTrue(_prompt_requires_file_tool(prompt))
        self.assertTrue(_prompt_should_force_write_file(prompt))

    def test_should_build_native_tool_prompt_without_json_protocol(self) -> None:
        prompt = (
            'You are a coding agent. Return strict JSON matching schema.\n'
            'Goal:\n'
            '新增一个abc文件夹，并在abc新增一个空白的json文件，里面写当前路径\n'
            'History:\n'
            '[]\n'
            'StateSummary:\n'
            '{"workspace":{"root":"D:\\\\workspace\\\\Anvil"}}\n'
            'LastSteps:\n'
            '[]\n'
            'ToolResults:\n'
            '[]\n'
            'Only output JSON.'
        )

        native_prompt = _native_tool_prompt(prompt)

        self.assertIn('Call the tool through the API', native_prompt)
        self.assertIn('新增一个abc文件夹', native_prompt)
        self.assertIn('D:\\\\workspace\\\\Anvil', native_prompt)
        self.assertNotIn('Return strict JSON', native_prompt)
        self.assertNotIn('Only output JSON', native_prompt)

    def test_should_detect_successful_tool_result_in_prompt(self) -> None:
        prompt = (
            'ToolResults:\n'
            "[{'id': 'call_1', 'ok': True, 'output': 'ok', 'error': None}]\n"
            'Only output JSON.'
        )

        self.assertTrue(_prompt_has_successful_tool_result(prompt))

    def test_should_raise_anthropic_error_message(self) -> None:
        with self.assertRaisesRegex(ValueError, 'bad request'):
            _extract_anthropic_text({'error': {'message': 'bad request'}})

    def test_should_build_anthropic_invoke(self) -> None:
        os.environ['ANTHROPIC_API_KEY'] = 'test-key'
        try:
            args = argparse.Namespace(
                provider='anthropic',
                model='claude-3-opus-20240229',
                api_key_env='ANTHROPIC_API_KEY',
                temperature=0.2,
                provider_timeout_s=30.0,
                max_retries=2,
                retry_backoff_s=1.0,
                retry_http_code=[],
            )
            invoke = build_invoke_from_args(args)
            # Just verify it creates successfully, actual API call would need key
            self.assertIsNotNone(invoke)
        finally:
            os.environ.pop('ANTHROPIC_API_KEY', None)

    def test_should_raise_when_anthropic_provider_without_key(self) -> None:
        backup = os.environ.pop('ANTHROPIC_API_KEY', None)
        try:
            args = argparse.Namespace(
                provider='anthropic',
                model='claude-3-opus-20240229',
                api_key_env='ANTHROPIC_API_KEY',
                temperature=0.2,
                provider_timeout_s=30.0,
                max_retries=2,
                retry_backoff_s=1.0,
                retry_http_code=[],
            )
            with self.assertRaises(ValueError):
                build_invoke_from_args(args)
        finally:
            if backup is not None:
                os.environ['ANTHROPIC_API_KEY'] = backup


class ProviderGeminiTests(unittest.TestCase):
    def test_should_build_gemini_invoke(self) -> None:
        os.environ['GEMINI_API_KEY'] = 'test-key'
        try:
            args = argparse.Namespace(
                provider='gemini',
                model='gemini-pro',
                api_key_env='GEMINI_API_KEY',
                temperature=0.2,
                provider_timeout_s=30.0,
                max_retries=2,
                retry_backoff_s=1.0,
                retry_http_code=[],
            )
            invoke = build_invoke_from_args(args)
            # Just verify it creates successfully, actual API call would need key
            self.assertIsNotNone(invoke)
        finally:
            os.environ.pop('GEMINI_API_KEY', None)

    def test_should_raise_when_gemini_provider_without_key(self) -> None:
        backup = os.environ.pop('GEMINI_API_KEY', None)
        try:
            args = argparse.Namespace(
                provider='gemini',
                model='gemini-pro',
                api_key_env='GEMINI_API_KEY',
                temperature=0.2,
                provider_timeout_s=30.0,
                max_retries=2,
                retry_backoff_s=1.0,
                retry_http_code=[],
            )
            with self.assertRaises(ValueError):
                build_invoke_from_args(args)
        finally:
            if backup is not None:
                os.environ['GEMINI_API_KEY'] = backup


if __name__ == '__main__':
    unittest.main()
