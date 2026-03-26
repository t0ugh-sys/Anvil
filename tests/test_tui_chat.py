from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from loop_agent.ui.tui_chat import (
    ChatConfig,
    _build_model_config,
    _build_provider_config,
    _cfg_banner,
    _parse_provider_choice,
    _provider_candidates,
)


class TuiChatTests(unittest.TestCase):
    def test_should_build_provider_config_with_defaults(self) -> None:
        current = ChatConfig(
            provider='openai_compatible',
            model='gpt-4o-mini',
            base_url='https://api.openai.com/v1',
            api_key_env='OPENAI_API_KEY',
            temperature=0.4,
            provider_timeout_s=30.0,
            history_limit=20,
        )

        updated = _build_provider_config(current, 'anthropic')

        self.assertEqual(updated.provider, 'anthropic')
        self.assertEqual(updated.model, 'claude-3-5-sonnet-latest')
        self.assertEqual(updated.base_url, '')
        self.assertEqual(updated.api_key_env, 'ANTHROPIC_API_KEY')
        self.assertEqual(updated.temperature, 0.4)
        self.assertEqual(updated.provider_timeout_s, 30.0)
        self.assertEqual(updated.history_limit, 20)

    def test_should_build_model_config_preserving_provider(self) -> None:
        current = ChatConfig(
            provider='gemini',
            model='gemini-1.5-flash',
            base_url='',
            api_key_env='GEMINI_API_KEY',
            temperature=0.2,
            provider_timeout_s=60.0,
            history_limit=30,
        )

        updated = _build_model_config(current, 'gemini-1.5-pro')

        self.assertEqual(updated.provider, 'gemini')
        self.assertEqual(updated.model, 'gemini-1.5-pro')
        self.assertEqual(updated.api_key_env, 'GEMINI_API_KEY')

    def test_should_parse_provider_choice_from_picker_label(self) -> None:
        choice = _provider_candidates()[0]
        parsed = _parse_provider_choice(choice)
        self.assertEqual(parsed, 'openai_compatible')

    def test_should_render_friendly_banner(self) -> None:
        cfg = ChatConfig(
            provider='anthropic',
            model='claude-3-5-sonnet-latest',
            base_url='',
            api_key_env='ANTHROPIC_API_KEY',
            temperature=0.2,
            provider_timeout_s=60.0,
            history_limit=30,
        )

        banner = _cfg_banner(cfg)

        self.assertIn('Anthropic Claude', banner)
        self.assertIn('claude-3-5-sonnet-latest', banner)
        self.assertIn('(n/a)', banner)


if __name__ == '__main__':
    unittest.main()
