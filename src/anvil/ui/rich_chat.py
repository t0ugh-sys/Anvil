"""
Rich REPL chat interface for Anvil.

Line-based terminal UI inspired by Claude Code:
- Simple prompt with model info
- Rich formatted output (colors, panels)
- Slash commands for configuration
- JSONL message persistence
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from rich.theme import Theme
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


# ============== Config ==============

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

ANVIL_THEME = Theme({
    'anvil.prompt': 'bold cyan',
    'anvil.model': 'bold green',
    'anvil.error': 'bold red',
    'anvil.info': 'dim',
    'anvil.user': 'bold white',
    'anvil.assistant': 'white',
    'anvil.command': 'bold yellow',
})


# ============== Helpers ==============

def _cfg_status(cfg: ChatConfig) -> str:
    return f'{cfg.model}'


def _help_text() -> str:
    return (
        '[bold]Commands:[/bold]\n'
        '  [cyan]/status[/cyan]        Show current config\n'
        '  [cyan]/model [name][/cyan]   Switch model\n'
        '  [cyan]/provider [name][/cyan] Switch provider\n'
        '  [cyan]/reset[/cyan]          Clear message history\n'
        '  [cyan]/history[/cyan]        Show recent messages\n'
        '  [cyan]/help[/cyan]           Show this help\n'
        '  [cyan]/exit[/cyan]           Exit'
    )


def _model_candidates(provider: str) -> list[str]:
    if provider == 'openai_compatible':
        return ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1', 'o3-mini']
    if provider == 'anthropic':
        return [
            'claude-sonnet-4-6', 'claude-haiku-4-5-20251001',
            'claude-3-5-sonnet-latest', 'claude-3-5-haiku-latest',
            'mimo-v2.5-pro', 'mimo-v2-pro',
        ]
    if provider == 'gemini':
        return ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash']
    return []


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


# ============== Chat Invoke ==============

def _build_chat_invoke(cfg: ChatConfig):
    if cfg.provider == 'openai_compatible':
        from ..llm.providers import openai_compatible_chat_invoke_factory
        api_key = os.getenv(cfg.api_key_env, '').strip()
        if not api_key:
            raise SystemExit(f'missing api key env: {cfg.api_key_env}')
        return openai_compatible_chat_invoke_factory(
            base_url=cfg.base_url, api_key=api_key, model=cfg.model,
            fallback_models=[], temperature=cfg.temperature,
            timeout_s=cfg.provider_timeout_s, debug=False, extra_headers={},
            max_retries=2, retry_backoff_s=1.0, retry_http_codes={502, 503, 504, 524},
        )

    if cfg.provider == 'anthropic':
        from ..llm.providers import anthropic_invoke_factory
        api_key = os.getenv(cfg.api_key_env, '').strip()
        if not api_key:
            raise SystemExit(f'missing api key env: {cfg.api_key_env}')
        return anthropic_invoke_factory(
            api_key=api_key, model=cfg.model,
            temperature=cfg.temperature, timeout_s=cfg.provider_timeout_s, debug=False,
        )

    if cfg.provider == 'gemini':
        from ..llm.providers import gemini_invoke_factory
        api_key = os.getenv(cfg.api_key_env, '').strip()
        if not api_key:
            raise SystemExit(f'missing api key env: {cfg.api_key_env}')
        return gemini_invoke_factory(
            api_key=api_key, model=cfg.model,
            temperature=cfg.temperature, timeout_s=cfg.provider_timeout_s, debug=False,
        )

    raise SystemExit(f'unknown provider: {cfg.provider}')


# ============== Message Store ==============

def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write('\n')


def _load_messages(path: Path, limit: int) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    out: List[Dict[str, str]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
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


# ============== REPL ==============

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='anvil-chat')
    p.add_argument('--provider', choices=PROVIDERS, default='openai_compatible')
    p.add_argument('--model', default='')
    p.add_argument('--base-url', default='')
    p.add_argument('--api-key-env', default='')
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--provider-timeout-s', type=float, default=60.0)
    p.add_argument('--history-limit', type=int, default=30)
    p.add_argument('--chat-id', default='')
    p.add_argument('--chat-dir', default='.anvil/chats')
    return p


def run(argv: Optional[list[str]] = None) -> int:
    if not HAS_RICH:
        print('Rich is required. Install with: pip install rich', file=sys.stderr)
        return 1

    args = build_parser().parse_args(argv)
    defaults = PROVIDER_DEFAULTS.get(args.provider, PROVIDER_DEFAULTS['openai_compatible'])

    chat_id = args.chat_id.strip() or _utc_run_id()
    chat_root = Path(args.chat_dir)
    chat_dir = chat_root / chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)
    messages_path = chat_dir / 'messages.jsonl'

    cfg = ChatConfig(
        provider=args.provider,
        model=str(args.model or defaults['model']),
        base_url=str(args.base_url or defaults['base_url']),
        api_key_env=str(args.api_key_env or defaults['api_key_env']),
        temperature=float(args.temperature),
        provider_timeout_s=float(args.provider_timeout_s),
        history_limit=int(args.history_limit),
    )

    console = Console(theme=ANVIL_THEME)
    invoke = _build_chat_invoke(cfg)

    # Welcome
    console.print()
    console.print(Panel(
        f'[bold]Anvil[/bold]  [anvil.model]{_cfg_status(cfg)}[/anvil.model]\n'
        f'[anvil.info]Chat: {chat_id} | Logs: {chat_dir}[/anvil.info]',
        border_style='cyan',
    ))
    console.print(_help_text())
    console.print()

    prompt_text = Text()
    prompt_text.append('anvil', style='anvil.prompt')
    prompt_text.append(' > ', style='bold')

    while True:
        try:
            line = console.input(prompt_text)
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0

        text = line.strip()
        if not text:
            continue

        # ============== Slash Commands ==============
        if text.startswith('/'):
            cmd_parts = text.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ''

            if cmd in ('/exit', '/quit'):
                return 0

            if cmd == '/help':
                console.print(_help_text())
                continue

            if cmd == '/status':
                console.print(Panel(
                    f'[bold]Provider:[/bold] {PROVIDER_LABELS.get(cfg.provider, cfg.provider)}\n'
                    f'[bold]Model:[/bold] {cfg.model}\n'
                    f'[bold]Base URL:[/bold] {cfg.base_url or "(n/a)"}\n'
                    f'[bold]API key env:[/bold] {cfg.api_key_env}\n'
                    f'[bold]Temperature:[/bold] {cfg.temperature}\n'
                    f'[bold]History limit:[/bold] {cfg.history_limit}',
                    title='Status',
                    border_style='cyan',
                ))
                continue

            if cmd == '/model':
                if not arg:
                    candidates = _model_candidates(cfg.provider)
                    current = cfg.model
                    console.print('[bold]Available models:[/bold]')
                    for m in candidates:
                        marker = ' [green]*[/green]' if m == current else ''
                        console.print(f'  [cyan]{m}[/cyan]{marker}')
                    console.print(f'\n[anvil.info]Usage: /model <name>[/anvil.info]')
                    continue
                try:
                    new_cfg = _build_model_config(cfg, arg)
                    invoke = _build_chat_invoke(new_cfg)
                    cfg = new_cfg
                    console.print(f'[anvil.model]Model: {cfg.model}[/anvil.model]')
                except Exception as e:
                    console.print(f'[anvil.error]Error: {e}[/anvil.error]')
                continue

            if cmd == '/provider':
                if not arg:
                    console.print('[bold]Available providers:[/bold]')
                    for p in PROVIDERS:
                        label = PROVIDER_LABELS.get(p, p)
                        marker = ' [green]*[/green]' if p == cfg.provider else ''
                        console.print(f'  [cyan]{p}[/cyan] - {label}{marker}')
                    console.print(f'\n[anvil.info]Usage: /provider <name>[/anvil.info]')
                    continue
                try:
                    new_cfg = _build_provider_config(cfg, arg)
                    invoke = _build_chat_invoke(new_cfg)
                    cfg = new_cfg
                    console.print(
                        f'[anvil.model]{PROVIDER_LABELS.get(cfg.provider, cfg.provider)} | {cfg.model}[/anvil.model]'
                    )
                except Exception as e:
                    console.print(f'[anvil.error]Error: {e}[/anvil.error]')
                continue

            if cmd == '/reset':
                if messages_path.exists():
                    backup = messages_path.with_suffix('.bak')
                    backup.write_text(messages_path.read_text(encoding='utf-8'), encoding='utf-8')
                    messages_path.unlink()
                console.print('[anvil.info]History cleared (backup: messages.bak)[/anvil.info]')
                continue

            if cmd == '/history':
                msgs = _load_messages(messages_path, 10)
                if not msgs:
                    console.print('[anvil.info]No messages yet.[/anvil.info]')
                    continue
                for msg in msgs:
                    role = msg['role']
                    content = msg['content']
                    if role == 'user':
                        console.print(f'[anvil.user]> {content}[/anvil.user]')
                    else:
                        console.print(f'[anvil.assistant]{content}[/anvil.assistant]')
                continue

            console.print(f'[anvil.error]Unknown command: {cmd}[/anvil.error]')
            continue

        # ============== Chat Message ==============
        _append_jsonl(messages_path, {
            'role': 'user', 'text': text,
            'ts': datetime.now(timezone.utc).isoformat(),
        })

        console.print(f'[anvil.user]> {text}[/anvil.user]')
        console.print('[dim]Thinking...[/dim]', end='\r')

        try:
            messages = _load_messages(messages_path, cfg.history_limit)
            reply = invoke(messages)
        except Exception as e:
            reply = f'ERROR: {e}'

        # Clear "Thinking..." line
        console.file.write('\033[K')
        console.file.flush()

        _append_jsonl(messages_path, {
            'role': 'assistant', 'text': reply,
            'ts': datetime.now(timezone.utc).isoformat(),
        })

        # Try rendering as markdown, fall back to plain text
        try:
            console.print(Markdown(reply))
        except Exception:
            console.print(f'[anvil.assistant]{reply}[/anvil.assistant]')

        console.print()


def main() -> None:
    raise SystemExit(run())


if __name__ == '__main__':
    main()
