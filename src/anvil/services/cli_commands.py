from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..ops.doctor import format_doctor_report, run_provider_doctor
from ..skills import get_skill, list_skills
from ..tools import build_default_tools, builtin_tool_specs
from ..tool_views import render_tool_overview


def run_tools_command(args: argparse.Namespace) -> int:
    specs = sorted(builtin_tool_specs(), key=lambda item: item.name)
    if getattr(args, 'verbose', False):
        print(render_tool_overview(specs))
        return 0
    names = sorted(build_default_tools().keys())
    print('\n'.join(names))
    return 0


def run_skills_command(_: argparse.Namespace) -> int:
    skills = list_skills()
    print('Available skills:')
    for name in skills:
        skill = get_skill(name)
        if skill:
            print(f'  - {name}: {skill.description}')
    print('\nUse --skill <name> to load specific skills')
    print('Use --skill all to load all skills')
    return 0


def _resolve_replay_events_file(args: argparse.Namespace) -> Path:
    events_file_value = getattr(args, 'events_file', '')
    if getattr(args, 'session_id', ''):
        return Path(args.sessions_dir) / args.session_id / 'events.jsonl'
    return Path(events_file_value)


def _load_event_rows(events_file: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in events_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _filter_replay_rows(rows: Iterable[Dict[str, Any]], filter_name: str) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        event = str(row.get('event', ''))
        payload = row.get('payload', {})
        payload_dict = payload if isinstance(payload, dict) else {}
        permission_decision = str(row.get('permission_decision', '') or '')
        tool_name = str(row.get('tool_name', '') or '')

        if filter_name == 'all':
            selected.append(row)
            continue
        if filter_name == 'tool-only':
            if tool_name or event == 'step_succeeded':
                selected.append(row)
            continue
        if filter_name == 'permissions-only':
            if permission_decision:
                selected.append(row)
            continue
        if filter_name == 'errors-only':
            error_text = str(payload_dict.get('error', '') or '')
            metadata = payload_dict.get('metadata', {})
            metadata_dict = metadata if isinstance(metadata, dict) else {}
            tool_results = metadata_dict.get('tool_results', [])
            has_tool_error = any(
                isinstance(item, dict) and (item.get('error') or item.get('permission_decision') in {'ask', 'deny'})
                for item in tool_results
            )
            stop_reason = str(payload_dict.get('stop_reason', '') or '')
            if error_text or has_tool_error or stop_reason in {'step_error'}:
                selected.append(row)
    return selected


def _trim_text(value: object, *, limit: int = 120) -> str:
    if not isinstance(value, str):
        return ''
    collapsed = ' '.join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + '...'


def _render_pretty_replay(rows: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for row in rows:
        event = str(row.get('event', 'unknown'))
        step = row.get('step')
        ts = str(row.get('ts', ''))
        session_id = str(row.get('session_id', '') or '')
        payload = row.get('payload', {})
        payload_dict = payload if isinstance(payload, dict) else {}
        prefix = f'[{ts}]'
        if step is not None:
            prefix += f' step={step}'
        prefix += f' {event}'
        if session_id:
            prefix += f' session={session_id}'

        detail = ''
        if event == 'run_started':
            detail = _trim_text(payload_dict.get('goal'))
        elif event in {'chat_user', 'chat_assistant'}:
            role = str(payload_dict.get('role', '') or event.replace('chat_', ''))
            detail = f'{role}: {_trim_text(payload_dict.get("content"))}'
        elif event == 'step_succeeded':
            tool_name = str(row.get('tool_name', '') or '')
            permission_decision = str(row.get('permission_decision', '') or '')
            output = _trim_text(payload_dict.get('output'))
            parts = [part for part in [tool_name, permission_decision, output] if part]
            detail = ' | '.join(parts)
        elif event == 'run_finished':
            done = payload_dict.get('done')
            stop_reason = str(payload_dict.get('stop_reason', '') or '')
            steps = payload_dict.get('steps')
            parts = [f'done={done}', f'stop_reason={stop_reason}']
            if steps is not None:
                parts.append(f'steps={steps}')
            detail = ' '.join(parts)
        else:
            detail = _trim_text(json.dumps(payload_dict, ensure_ascii=False))

        lines.append(prefix if not detail else f'{prefix} {detail}')
    return '\n'.join(lines)


def run_replay_command(args: argparse.Namespace) -> int:
    events_file = _resolve_replay_events_file(args)
    if not events_file.exists():
        print(f'events file not found: {events_file}')
        return 1
    if not getattr(args, 'pretty', False):
        print(events_file.read_text(encoding='utf-8'))
        return 0
    rows = _load_event_rows(events_file)
    rows = _filter_replay_rows(rows, str(getattr(args, 'filter', 'all') or 'all'))
    limit = getattr(args, 'limit', None)
    if isinstance(limit, int) and limit > 0:
        rows = rows[-limit:]
    print(_render_pretty_replay(rows))
    return 0


def run_doctor_command(args: argparse.Namespace) -> int:
    api_key = os.getenv(args.api_key_env, '').strip()
    payload = run_provider_doctor(
        base_url=args.base_url,
        model=args.model,
        wire_api=args.wire_api,
        timeout_s=args.provider_timeout_s,
        api_key_present=bool(api_key),
        extra_headers=args.provider_header,
    )
    if args.output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(format_doctor_report(payload))
    return 0
