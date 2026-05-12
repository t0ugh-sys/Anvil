"""
Rich REPL chat interface for Anvil.

Claude Code-style terminal UI:
- Minimal prompt marker
- Model info in footer line
- Rich markdown output
- Slash commands
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chrome import (
    DOT_SEPARATOR,
    PROMPT_MARKER,
    RESPONSE_MARKER,
    WORKING_MARKER,
    bounded_width,
    box_lines,
    response_lines,
    truncate,
)

try:
    from rich.console import Console
    from rich.markdown import Markdown
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
    'anthropic': 'Anthropic',
    'gemini': 'Gemini',
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

THEME = Theme({
    'anvil.prompt': 'bold #7ee787',
    'anvil.working': 'bold #7dd3fc',
    'anvil.response': 'bold #8ab4f8',
    'anvil.model': '#9aa4b2',
    'anvil.error': 'bold red',
    'anvil.dim': '#8b949e',
    'anvil.user': 'bold',
    'anvil.border': '#4f8f8f',
    'anvil.output': '#d7dde8',
})


# ============== Helpers ==============

def _model_candidates(provider: str) -> list[str]:
    if provider == 'openai_compatible':
        return [
            'gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1', 'o3-mini',
            'mimo-v2.5-pro', 'mimo-v2-pro',
        ]
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
        provider=provider, model=defaults['model'],
        base_url=defaults['base_url'], api_key_env=defaults['api_key_env'],
        temperature=current_cfg.temperature,
        provider_timeout_s=current_cfg.provider_timeout_s,
        history_limit=current_cfg.history_limit,
    )


def _build_model_config(current_cfg: ChatConfig, model: str) -> ChatConfig:
    cleaned = model.strip()
    if not cleaned:
        raise ValueError('model must not be empty')
    return ChatConfig(
        provider=current_cfg.provider, model=cleaned,
        base_url=current_cfg.base_url, api_key_env=current_cfg.api_key_env,
        temperature=current_cfg.temperature,
        provider_timeout_s=current_cfg.provider_timeout_s,
        history_limit=current_cfg.history_limit,
    )


def _detect_provider_from_url(url: str, default: str) -> str:
    lower = url.lower()
    if '/anthropic/' in lower or lower.endswith('/anthropic'):
        return 'anthropic'
    if '/gemini/' in lower:
        return 'gemini'
    return default


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
            base_url=cfg.base_url,
            temperature=cfg.temperature, timeout_s=cfg.provider_timeout_s, debug=False,
        )

    if cfg.provider == 'gemini':
        from ..llm.providers import gemini_invoke_factory
        api_key = os.getenv(cfg.api_key_env, '').strip()
        if not api_key:
            raise SystemExit(f'missing api key env: {cfg.api_key_env}')
        return gemini_invoke_factory(
            api_key=api_key, model=cfg.model,
            base_url=cfg.base_url,
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


# ============== Footer ==============

def _print_footer(console: Console, cfg: ChatConfig) -> None:
    t = Text()
    t.append('  ', style='dim')
    t.append(cfg.model, style='anvil.model')
    console.print(t)


def _ui_width(console: Console) -> int:
    return bounded_width(console.width)


def _status_line(cfg: ChatConfig, *, width: int) -> str:
    provider = PROVIDER_LABELS.get(cfg.provider, cfg.provider)
    return truncate(f'  {cfg.model} {DOT_SEPARATOR} {provider} {DOT_SEPARATOR} {Path.cwd()}', width)


def _print_welcome(console: Console, cfg: ChatConfig, *, chat_dir: Path) -> None:
    width = _ui_width(console)
    lines = [
        f'{WORKING_MARKER} Welcome to Anvil',
        '',
        f'cwd: {Path.cwd()}',
        f'chat: {chat_dir.name}',
        f'logs: {chat_dir}',
        f'model: {cfg.model}',
        f'provider: {PROVIDER_LABELS.get(cfg.provider, cfg.provider)}',
    ]
    welcome_lines = box_lines(lines, width=width, title='Anvil')
    for index, line in enumerate(welcome_lines):
        style = 'anvil.border' if index in {0, len(welcome_lines) - 1} else 'anvil.dim'
        console.print(line, style=style, markup=False)
    hint = f'  ? for shortcuts {DOT_SEPARATOR} /help {DOT_SEPARATOR} /status {DOT_SEPARATOR} /model {DOT_SEPARATOR} /exit'
    console.print(truncate(hint, width), style='anvil.dim', markup=False)
    console.print(_status_line(cfg, width=width), style='anvil.model', markup=False)
    console.print()


def _print_prompt(console: Console) -> None:
    console.print(PROMPT_MARKER, style='anvil.prompt', end=' ')


def _print_help(console: Console, cfg: ChatConfig) -> None:
    console.print('Commands:', style='anvil.dim')
    console.print('  ?                  Show this help', style='anvil.output')
    console.print('  /model [name]      Switch or list models', style='anvil.output')
    console.print('  /provider [name]   Switch or list providers', style='anvil.output')
    console.print('  /status            Show config', style='anvil.output')
    console.print('  /history           Recent messages', style='anvil.output')
    console.print('  /reset             Clear history', style='anvil.output')
    console.print('  /exit              Quit', style='anvil.output')
    console.print()
    _print_footer(console, cfg)


def _print_response(console: Console, text: str, cfg: ChatConfig) -> None:
    width = _ui_width(console)
    for line in response_lines(text, width=width):
        if line.lstrip().startswith(RESPONSE_MARKER):
            prefix, _, rest = line.partition(RESPONSE_MARKER)
            rendered = Text(prefix)
            rendered.append(RESPONSE_MARKER, style='anvil.response')
            rendered.append(rest, style='anvil.output')
            console.print(rendered)
        else:
            console.print(line, style='anvil.output', markup=False)
    console.print(_status_line(cfg, width=width), style='anvil.model', markup=False)
    console.print()


# ============== Safe Print ==============

def _safe_print(console: Console, text: str) -> None:
    """Print text handling Unicode encoding issues on Windows."""
    try:
        console.print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        safe = text.encode(encoding, errors='replace').decode(encoding)
        console.print(safe)


def _safe_print_markdown(console: Console, text: str) -> None:
    """Print markdown handling Unicode encoding issues on Windows."""
    try:
        console.print(Markdown(text))
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        safe = text.encode(encoding, errors='replace').decode(encoding)
        try:
            console.print(Markdown(safe))
        except Exception:
            console.print(safe)
    except Exception:
        _safe_print(console, text)


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

    base_url = str(args.base_url or defaults['base_url'])
    provider = _detect_provider_from_url(base_url, args.provider)

    chat_dir = Path(args.chat_dir) / (args.chat_id.strip() or _utc_run_id())
    chat_dir.mkdir(parents=True, exist_ok=True)
    messages_path = chat_dir / 'messages.jsonl'

    cfg = ChatConfig(
        provider=provider,
        model=str(args.model or defaults['model']),
        base_url=base_url,
        api_key_env=str(args.api_key_env or defaults['api_key_env']),
        temperature=float(args.temperature),
        provider_timeout_s=float(args.provider_timeout_s),
        history_limit=int(args.history_limit),
    )

    console = Console(theme=THEME)
    try:
        invoke = _build_chat_invoke(cfg)
    except SystemExit as e:
        console.print(f'[anvil.error]{e}[/anvil.error]')
        return 1

    _print_welcome(console, cfg, chat_dir=chat_dir)

    while True:
        try:
            _print_prompt(console)
            line = sys.stdin.readline()
            if not line:
                console.print()
                return 0
        except KeyboardInterrupt:
            console.print()
            return 0

        text = line.strip()
        if not text:
            sys.stdout.write('\x1b[1A\x1b[2K')
            sys.stdout.flush()
            continue

        # ============== Slash Commands ==============
        if text.startswith('/'):
            cmd_parts = text.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ''

            if cmd in ('/exit', '/quit'):
                return 0

            if cmd == '/help':
                _print_help(console, cfg)
                continue

            if cmd == '/status':
                console.print(f'  provider: {PROVIDER_LABELS.get(cfg.provider, cfg.provider)}', style='anvil.output')
                console.print(f'  model:    {cfg.model}', style='anvil.output')
                console.print(f'  base_url: {cfg.base_url or "(n/a)"}', style='anvil.output')
                console.print(f'  api_key:  {cfg.api_key_env}', style='anvil.output')
                console.print(f'  temp:     {cfg.temperature}', style='anvil.output')
                console.print()
                _print_footer(console, cfg)
                continue

            if cmd == '/model':
                if not arg:
                    candidates = _model_candidates(cfg.provider)
                    for m in candidates:
                        marker = ' *' if m == cfg.model else ''
                        console.print(f'  {m}{marker}', style='anvil.output')
                    console.print()
                    _print_footer(console, cfg)
                    continue
                try:
                    new_cfg = _build_model_config(cfg, arg)
                    invoke = _build_chat_invoke(new_cfg)
                    cfg = new_cfg
                except Exception as e:
                    console.print(f'[anvil.error]{e}[/anvil.error]')
                console.print()
                _print_footer(console, cfg)
                continue

            if cmd == '/provider':
                if not arg:
                    for p in PROVIDERS:
                        label = PROVIDER_LABELS.get(p, p)
                        marker = ' *' if p == cfg.provider else ''
                        console.print(f'  {p} ({label}){marker}', style='anvil.output')
                    console.print()
                    _print_footer(console, cfg)
                    continue
                try:
                    new_cfg = _build_provider_config(cfg, arg)
                    invoke = _build_chat_invoke(new_cfg)
                    cfg = new_cfg
                except Exception as e:
                    console.print(f'[anvil.error]{e}[/anvil.error]')
                console.print()
                _print_footer(console, cfg)
                continue

            if cmd == '/reset':
                if messages_path.exists():
                    messages_path.with_suffix('.bak').write_text(
                        messages_path.read_text(encoding='utf-8'), encoding='utf-8'
                    )
                    messages_path.unlink()
                console.print('History cleared', style='anvil.dim')
                console.print()
                continue

            if cmd == '/history':
                msgs = _load_messages(messages_path, 10)
                if not msgs:
                    console.print('No messages yet', style='anvil.dim')
                    console.print()
                    continue
                for msg in msgs:
                    if msg['role'] == 'user':
                        console.print(f'> {msg["content"]}', style='anvil.prompt')
                    else:
                        _safe_print(console, msg['content'])
                console.print()
                continue

            console.print(f'[anvil.error]Unknown: {cmd}[/anvil.error]')
            console.print()
            continue

        if text == '?':
            _print_help(console, cfg)
            continue

        # ============== Chat ==============
        _append_jsonl(messages_path, {
            'role': 'user', 'text': text,
            'ts': datetime.now(timezone.utc).isoformat(),
        })
        console.print(f'  {WORKING_MARKER} Working...', style='anvil.working', markup=False)

        try:
            messages = _load_messages(messages_path, cfg.history_limit)
            llm_prompt = '\n'.join(f'{m["role"]}: {m["content"]}' for m in messages)
            reply = invoke(llm_prompt)
        except Exception as e:
            reply = f'ERROR: {e}'

        _append_jsonl(messages_path, {
            'role': 'assistant', 'text': reply,
            'ts': datetime.now(timezone.utc).isoformat(),
        })

        _print_response(console, reply, cfg)


def main() -> None:
    raise SystemExit(run())


if __name__ == '__main__':
    main()
