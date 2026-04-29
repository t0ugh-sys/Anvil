from __future__ import annotations

from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from anvil.api import AgentConfig, AnvilAPI


class ApiTests(unittest.TestCase):
    def test_should_validate_and_normalize_agent_config(self) -> None:
        config = AgentConfig(
            provider='mock',
            model='mock-model',
            workspace='.',
        )

        validated = config.validate()

        self.assertIs(validated, config)
        self.assertIsInstance(config.workspace, Path)
        self.assertFalse(hasattr(config, 'goal'))

    def test_should_build_invoke_from_configured_provider(self) -> None:
        config = AgentConfig(
            provider='openai_compatible',
            model='gpt-4o-mini',
            base_url='https://api.example.com/v1',
            api_key_env='OPENAI_API_KEY',
            history_window=2,
        )
        api = AnvilAPI(config)

        with patch('anvil.api.build_invoke_from_args', return_value=lambda prompt: '{"answer":"ok","done":true}') as build:
            result = api.run('say hi')

        self.assertTrue(result.success)
        self.assertEqual(build.call_count, 1)
        args = build.call_args.args[0]
        self.assertEqual(args.provider, 'openai_compatible')
        self.assertEqual(args.base_url, 'https://api.example.com/v1')
        self.assertEqual(build.call_args.kwargs['mode'], 'json')

    def test_should_build_coding_invoke_from_configured_provider(self) -> None:
        config = AgentConfig(
            provider='mock',
            model='mock-v3',
            workspace=Path('.'),
        )
        api = AnvilAPI(config)
        fake_invoke = lambda prompt: '{"thought":"done","plan":[],"tool_calls":[],"final":"done"}'
        fake_result = SimpleNamespace(
            done=True,
            final_output='done',
            steps=1,
            stop_reason=SimpleNamespace(value='done'),
            elapsed_s=0.1,
        )

        with patch('anvil.api.build_invoke_from_args', return_value=fake_invoke) as build, patch('anvil.coding_agent.build_coding_step') as build_coding_step, patch('anvil.coding_agent.run_coding_agent', return_value=fake_result) as run_coding_agent:
            result = api.run_coding('inspect repo')

        self.assertTrue(result.success)
        self.assertEqual(build.call_args.kwargs['mode'], 'coding')
        self.assertIs(build_coding_step.call_args.args[0], fake_invoke)
        self.assertIs(run_coding_agent.call_args.kwargs['decider'], fake_invoke)
