from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


@dataclass(frozen=True)
class ChatConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float
    provider_timeout_s: float
    history_limit: int


PROVIDERS = ['openai_compatible', 'anthropic', 'gemini']

PROVIDER_LABELS: Dict[str, str] = {
    'openai_compatible': 'OpenAI Compatible',
    'anthropic': 'Anthropic Claude',
    'gemini': 'Google Gemini',
}

PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    'openai_compatible': {
        'model': 'gpt-4o-mini',
        'base_url': 'https://api.openai.com/v1',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'anthropic': {
        'model': 'claude-3-5-sonnet-latest',
        'base_url': '',
        'api_key_env': 'ANTHROPIC_API_KEY',
    },
    'gemini': {
        'model': 'gemini-1.5-flash',
        'base_url': '',
        'api_key_env': 'GEMINI_API_KEY',
    },
}


def _cfg_banner(cfg: ChatConfig) -> str:
    base = cfg.base_url or '(n/a)'
    label = PROVIDER_LABELS.get(cfg.provider, cfg.provider)
    return f'Provider: {label} | Model: {cfg.model} | Base URL: {base} | API key env: {cfg.api_key_env}'


def _help_text() -> str:
    return 'Commands: /status, /provider [name] (or Ctrl+P), /model [name] (or Ctrl+M), /reset, /exit'


def _welcome_text(chat_id: str, chat_dir: Path, cfg: ChatConfig, *, reset_note: str = '') -> str:
    lines = [
        f'Chat: {chat_id}',
        _cfg_banner(cfg),
        f'Logs: {chat_dir}',
    ]
    if reset_note:
        lines.append(reset_note)
    lines.append(_help_text())
    return '\n'.join(lines) + '\n'


def _provider_candidates() -> list[str]:
    return [f'{provider} - {PROVIDER_LABELS.get(provider, provider)}' for provider in PROVIDERS]


def _parse_provider_choice(choice: str) -> str:
    provider = choice.split(' - ', 1)[0].strip()
    if provider not in PROVIDERS:
        raise ValueError(f'unknown provider: {provider}')
    return provider


def _build_provider_config(current_cfg: ChatConfig, provider: str) -> ChatConfig:
    provider = provider.strip()
    if provider not in PROVIDERS:
        raise ValueError(f'unknown provider: {provider}')

    defaults = PROVIDER_DEFAULTS[provider]
    return ChatConfig(
        provider=provider,
        model=defaults['model'],
        base_url=defaults['base_url'],
        api_key_env=defaults['api_key_env'],
        temperature=current_cfg.temperature,
        provider_timeout_s=current_cfg.provider_timeout_s,
        history_limit=current_cfg.history_limit,
    )


def _build_model_config(current_cfg: ChatConfig, model: str) -> ChatConfig:
    cleaned = model.strip()
    if not cleaned:
        raise ValueError('model must not be empty')
    return ChatConfig(
        provider=current_cfg.provider,
        model=cleaned,
        base_url=current_cfg.base_url,
        api_key_env=current_cfg.api_key_env,
        temperature=current_cfg.temperature,
        provider_timeout_s=current_cfg.provider_timeout_s,
        history_limit=current_cfg.history_limit,
    )


def _apply_model_change(current_cfg: ChatConfig, model: str) -> Tuple[ChatConfig, str]:
    updated = _build_model_config(current_cfg, model)
    return updated, _cfg_banner(updated)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='anvil-chat')
    p.add_argument('--provider', choices=PROVIDERS, default='openai_compatible')
    p.add_argument('--model', default='')
    p.add_argument('--base-url', default='')
    p.add_argument('--api-key-env', default='')
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--provider-timeout-s', type=float, default=60.0)
    p.add_argument('--history-limit', type=int, default=30, help='max messages to send as context')
    p.add_argument('--chat-id', default='')
    p.add_argument('--chat-dir', default='.anvil/chats', help='chat logs root dir')
    return p


def _require_textual():
    try:
        from textual.app import App  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            'TUI dependencies are not installed. Install with: pip install -e .[tui]\n'
            f'Import error: {e}'
        )


def _build_chat_invoke(cfg: ChatConfig):
    if cfg.provider == 'openai_compatible':
        from ..llm.providers import openai_compatible_chat_invoke_factory

        api_key = os.getenv(cfg.api_key_env, '').strip()
        if not api_key:
            raise SystemExit(f'missing api key env: {cfg.api_key_env}')

        return openai_compatible_chat_invoke_factory(
            base_url=cfg.base_url,
            api_key=api_key,
            model=cfg.model,
            fallback_models=[],
            temperature=cfg.temperature,
            timeout_s=cfg.provider_timeout_s,
            debug=False,
            extra_headers={},
            max_retries=2,
            retry_backoff_s=1.0,
            retry_http_codes={502, 503, 504, 524},
        )

    if cfg.provider == 'anthropic':
        from ..llm.providers import anthropic_invoke_factory

        api_key = os.getenv(cfg.api_key_env, '').strip()
        if not api_key:
            raise SystemExit(f'missing api key env: {cfg.api_key_env}')

        return anthropic_invoke_factory(
            api_key=api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            timeout_s=cfg.provider_timeout_s,
            debug=False,
        )

    if cfg.provider == 'gemini':
        from ..llm.providers import gemini_invoke_factory

        api_key = os.getenv(cfg.api_key_env, '').strip()
        if not api_key:
            raise SystemExit(f'missing api key env: {cfg.api_key_env}')

        return gemini_invoke_factory(
            api_key=api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            timeout_s=cfg.provider_timeout_s,
            debug=False,
        )

    raise SystemExit(f'unknown provider: {cfg.provider}')


def _chat_dir(root: Path, chat_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    d = root / chat_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write('\n')


def run(argv: Optional[list[str]] = None) -> int:
    _require_textual()

    from textual.app import App, ComposeResult
    from textual.containers import Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Footer, Header, Input, Static, OptionList

    args = build_parser().parse_args(argv)

    defaults = PROVIDER_DEFAULTS.get(args.provider, PROVIDER_DEFAULTS['openai_compatible'])

    chat_id = args.chat_id.strip() or _utc_run_id()
    chat_root = Path(args.chat_dir)
    chat_dir = _chat_dir(chat_root, chat_id)

    cfg = ChatConfig(
        provider=args.provider,
        model=str(args.model or defaults['model']),
        base_url=str(args.base_url or defaults['base_url']),
        api_key_env=str(args.api_key_env or defaults['api_key_env']),
        temperature=float(args.temperature),
        provider_timeout_s=float(args.provider_timeout_s),
        history_limit=int(args.history_limit),
    )

    invoke = _build_chat_invoke(cfg)

    current_cfg = cfg
    current_invoke = invoke

    def _apply_provider_change(provider: str) -> Tuple[ChatConfig, Any, str]:
        new_cfg = _build_provider_config(current_cfg, provider)
        new_invoke = _build_chat_invoke(new_cfg)
        return new_cfg, new_invoke, _cfg_banner(new_cfg)

    messages_path = chat_dir / 'messages.jsonl'

    def load_messages(limit: int) -> list[dict[str, str]]:
        if not messages_path.exists():
            return []
        out: list[dict[str, str]] = []
        for line in messages_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            role = row.get('role')
            text = row.get('text')
            if role in {'user', 'assistant'} and isinstance(text, str):
                out.append({'role': role, 'content': text})
        return out[-limit:] if limit > 0 else out

    def _model_candidates(provider: str) -> list[str]:
        if provider == 'openai_compatible':
            return [
                'gpt-4o-mini',
                'gpt-4o',
                'gpt-4.1-mini',
                'gpt-4.1',
                'o3-mini',
            ]
        if provider == 'anthropic':
            return [
                'claude-3-5-sonnet-latest',
                'claude-3-5-haiku-latest',
            ]
        if provider == 'gemini':
            return [
                'gemini-1.5-flash',
                'gemini-1.5-pro',
            ]
        return []

    class ModelPickerScreen(ModalScreen[Optional[str]]):
        def compose(self) -> ComposeResult:
            with Vertical(id='panel'):
                yield Static(f'Pick a model ({current_cfg.provider})', id='title')
                yield OptionList(*_model_candidates(current_cfg.provider), id='options')
                yield Static('Enter: select | Esc: cancel', id='hint')

        def on_mount(self) -> None:
            self.query_one('#options', OptionList).focus()

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            self.dismiss(str(event.option.prompt))

        def key_escape(self) -> None:
            self.dismiss(None)

    class ProviderPickerScreen(ModalScreen[Optional[str]]):
        def compose(self) -> ComposeResult:
            with Vertical(id='panel'):
                yield Static('Pick a provider', id='title')
                yield OptionList(*_provider_candidates(), id='options')
                yield Static('Enter: select | Esc: cancel', id='hint')

        def on_mount(self) -> None:
            self.query_one('#options', OptionList).focus()

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            self.dismiss(_parse_provider_choice(str(event.option.prompt)))

        def key_escape(self) -> None:
            self.dismiss(None)

    class ChatApp(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #log {
            height: 1fr;
            overflow-y: auto;
            border: round $surface;
            padding: 1;
        }
        #input {
            border: round $surface;
        }

        ModelPickerScreen {
            align: center middle;
        }
        ModelPickerScreen > #panel {
            width: 80%;
            max-width: 100;
            height: auto;
            max-height: 80%;
            border: round $surface;
            background: $panel;
            padding: 1;
        }
        ModelPickerScreen #title {
            padding: 0 0 1 0;
        }
        ModelPickerScreen #options {
            height: auto;
            max-height: 20;
            border: round $surface;
        }
        """

        BINDINGS = [
            ('ctrl+c', 'quit', 'Quit'),
            ('ctrl+p', 'pick_provider', 'Provider'),
            ('ctrl+m', 'pick_model', 'Model'),
        ]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Vertical():
                yield Static('', id='log')
                yield Input(placeholder='Say something... (/exit to quit)', id='input')
            yield Footer()

        def action_pick_provider(self) -> None:
            self.push_screen(ProviderPickerScreen(), self._on_provider_picked)

        def action_pick_model(self) -> None:
            self._open_model_picker()

        def _open_model_picker(self) -> None:
            self.push_screen(ModelPickerScreen(), self._on_model_picked)

        def _on_provider_picked(self, value: Optional[str]) -> None:
            if not value:
                return

            nonlocal current_cfg, current_invoke

            try:
                new_cfg, new_invoke, banner = _apply_provider_change(value)
            except Exception as e:
                log = self.query_one('#log', Static)
                existing = str(log.renderable)
                log.update(existing + f'ERROR: {e}\n')
                return

            current_cfg = new_cfg
            current_invoke = new_invoke

            log = self.query_one('#log', Static)
            existing = str(log.renderable)
            log.update(existing + f'\n[{banner}]\n')

        def _on_model_picked(self, value: Optional[str]) -> None:
            if not value:
                return

            nonlocal current_cfg, current_invoke

            try:
                new_cfg, banner = _apply_model_change(current_cfg, value)
                new_invoke = _build_chat_invoke(new_cfg)
            except Exception as e:
                log = self.query_one('#log', Static)
                existing = str(log.renderable)
                log.update(existing + f'ERROR: {e}\n')
                return

            current_cfg = new_cfg
            current_invoke = new_invoke

            log = self.query_one('#log', Static)
            existing = str(log.renderable)
            log.update(existing + f'\n[{banner}]\n')

        def on_ready(self) -> None:
            self.query_one('#input', Input).focus()

        def on_mount(self) -> None:
            log = self.query_one('#log', Static)
            log.update(_welcome_text(chat_id, chat_dir, current_cfg))

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            nonlocal current_cfg, current_invoke

            text = event.value.strip()
            event.input.value = ''
            if not text:
                return
            if text in ('/exit', '/quit'):
                self.exit()
                return

            if text == '/reset':
                if messages_path.exists():
                    backup = messages_path.with_suffix('.bak')
                    backup.write_text(messages_path.read_text(encoding='utf-8'), encoding='utf-8')
                    messages_path.unlink()
                log = self.query_one('#log', Static)
                log.update(
                    _welcome_text(
                        chat_id,
                        chat_dir,
                        current_cfg,
                        reset_note='(reset: messages.jsonl cleared; backup: messages.bak)',
                    )
                )
                return

            if text == '/status':
                log = self.query_one('#log', Static)
                existing = str(log.renderable)
                log.update(existing + f'\n[{_cfg_banner(current_cfg)}]\n')
                return

            if text == '/model':
                self._open_model_picker()
                return

            if text.startswith('/model '):
                model_name = text.split(maxsplit=1)[1].strip()
                try:
                    new_cfg, banner = _apply_model_change(current_cfg, model_name)
                    new_invoke = _build_chat_invoke(new_cfg)
                except Exception as e:
                    log = self.query_one('#log', Static)
                    existing = str(log.renderable)
                    log.update(existing + f'ERROR: {e}\n')
                    return

                current_cfg = new_cfg
                current_invoke = new_invoke

                log = self.query_one('#log', Static)
                existing = str(log.renderable)
                log.update(existing + f'\n[{banner}]\n')
                return

            if text.startswith('/provider'):
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    self.action_pick_provider()
                    return

                try:
                    new_cfg, new_invoke, banner = _apply_provider_change(parts[1])
                except Exception as e:
                    log = self.query_one('#log', Static)
                    existing = str(log.renderable)
                    log.update(existing + f'ERROR: {e}\n')
                    return

                current_cfg = new_cfg
                current_invoke = new_invoke

                log = self.query_one('#log', Static)
                existing = str(log.renderable)
                log.update(existing + f'\n[{banner}]\n')
                return

            log = self.query_one('#log', Static)
            existing = str(log.renderable)
            log.update(existing + f'\n> {text}\n')

            _append_jsonl(messages_path, {'role': 'user', 'text': text, 'ts': datetime.now(timezone.utc).isoformat()})

            try:
                messages = load_messages(current_cfg.history_limit)
                reply = current_invoke(messages)
            except Exception as e:
                reply = f'ERROR: {e}'

            _append_jsonl(
                messages_path,
                {'role': 'assistant', 'text': reply, 'ts': datetime.now(timezone.utc).isoformat()},
            )
            log = self.query_one('#log', Static)
            existing = str(log.renderable)
            log.update(existing + reply + '\n')

    ChatApp().run()
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == '__main__':
    main()
