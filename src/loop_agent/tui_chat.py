from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='loopagent-chat')
    p.add_argument('--provider', choices=['openai_compatible'], default='openai_compatible')
    p.add_argument('--model', default='gpt-4o-mini')
    p.add_argument('--base-url', default='https://api.openai.com/v1')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--provider-timeout-s', type=float, default=60.0)
    p.add_argument('--history-limit', type=int, default=30, help='max messages to send as context')
    p.add_argument('--chat-id', default='')
    p.add_argument('--chat-dir', default='.loopagent/chats', help='chat logs root dir')
    return p


def _require_textual():
    try:
        from textual.app import App  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            'TUI dependencies are not installed. Install with: pip install -e .[tui]\n'
            f'Import error: {e}'
        )


def _build_openai_compatible_chat_invoke(cfg: ChatConfig):
    from .llm.providers import openai_compatible_chat_invoke_factory

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
    from textual.widgets import Footer, Header, Input, Static

    args = build_parser().parse_args(argv)

    chat_id = args.chat_id.strip() or _utc_run_id()
    chat_root = Path(args.chat_dir)
    chat_dir = _chat_dir(chat_root, chat_id)

    cfg = ChatConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=float(args.temperature),
        provider_timeout_s=float(args.provider_timeout_s),
        history_limit=int(args.history_limit),
    )

    invoke = _build_openai_compatible_chat_invoke(cfg)

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
        """

        BINDINGS = [
            ('ctrl+c', 'quit', 'Quit'),
        ]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Vertical():
                yield Static('', id='log')
                yield Input(placeholder='Say something... (/exit to quit)', id='input')
            yield Footer()

        def on_mount(self) -> None:
            log = self.query_one('#log', Static)
            log.update(
                f'Chat: {chat_id}\nProvider: {cfg.provider}\nModel: {cfg.model}\nBase URL: {cfg.base_url}\n'
                f'Logs: {chat_dir}\n'
            )

        async def on_input_submitted(self, event: Input.Submitted) -> None:
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
                    f'Chat: {chat_id}\nProvider: {cfg.provider}\nModel: {cfg.model}\nBase URL: {cfg.base_url}\n'
                    f'Logs: {chat_dir}\n(reset: messages.jsonl cleared; backup: messages.bak)\n'
                )
                return

            log = self.query_one('#log', Static)
            existing = str(log.renderable)
            log.update(existing + f'\n> {text}\n')

            _append_jsonl(messages_path, {'role': 'user', 'text': text, 'ts': datetime.now(timezone.utc).isoformat()})

            try:
                messages = load_messages(cfg.history_limit)
                reply = invoke(messages)
            except Exception as e:
                reply = f'ERROR: {e}'

            _append_jsonl(
                messages_path,
                {'role': 'assistant', 'text': reply, 'ts': datetime.now(timezone.utc).isoformat()},
            )
            log = self.query_one('#log', Static)
            existing = str(log.renderable)
            log.update(existing + reply + '\n')

    ChatApp(title='LoopAgent Chat').run()
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == '__main__':
    main()
