from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .entrypoints.parser_builders import (
    register_doctor_parser,
    register_replay_parser,
    register_skills_parser,
    register_tools_parser,
)
from .llm.providers import build_invoke_from_args
from .services.catalog_service import render_skills, render_tools
from .services import coding_runtime as _coding_runtime
from .services.replay_service import render_replay, resolve_events_file
from .services.session_runtime import should_launch_interactive as _should_launch_interactive
from .services.team_service import (
    run_team_add_task_command as _team_service_run_team_add_task_command,
    run_team_broadcast_command as _team_service_run_team_broadcast_command,
    run_team_run_command as _team_service_run_team_run_command,
    run_team_send_command as _team_service_run_team_send_command,
    run_team_serve_command as _team_service_run_team_serve_command,
    run_team_shutdown_command as _team_service_run_team_shutdown_command,
)
from .ops.doctor import format_doctor_report, run_provider_doctor
from .tools import builtin_tool_specs
from .utils import default_run_id


def _build_coding_prompt(
    *,
    goal: str,
    history: Tuple[str, ...],
    tool_results: Tuple[Any, ...],
    state_summary: Dict[str, object],
    last_steps: Tuple[str, ...],
    history_window: int,
    skills=None,
) -> str:
    return _coding_runtime.build_coding_prompt(
        goal=goal,
        history=history,
        tool_results=tool_results,
        state_summary=state_summary,
        last_steps=last_steps,
        history_window=history_window,
        skills=skills,
    )


def _build_coding_decider(args: argparse.Namespace, skills=None):
    original = _coding_runtime.build_invoke_from_args
    _coding_runtime.build_invoke_from_args = build_invoke_from_args
    try:
        return _coding_runtime.build_coding_decider(args, skills)
    finally:
        _coding_runtime.build_invoke_from_args = original


def _build_coding_summarizer(args: argparse.Namespace):
    original = _coding_runtime.build_invoke_from_args
    _coding_runtime.build_invoke_from_args = build_invoke_from_args
    try:
        return _coding_runtime.build_coding_summarizer(args)
    finally:
        _coding_runtime.build_invoke_from_args = original


def _load_skills_from_args(args: argparse.Namespace):
    return _coding_runtime.load_skills_from_args(args)


def _run_code_command(args: argparse.Namespace) -> int:
    return _coding_runtime.run_code_command(args)


def _run_team_run_command(args: argparse.Namespace) -> int:
    return _team_service_run_team_run_command(
        args,
        decider_builder=_build_coding_decider,
        skills_loader=_load_skills_from_args,
    )


def _run_team_serve_command(args: argparse.Namespace) -> int:
    return _team_service_run_team_serve_command(
        args,
        decider_builder=_build_coding_decider,
        skills_loader=_load_skills_from_args,
    )


def _run_team_add_task_command(args: argparse.Namespace) -> int:
    return _team_service_run_team_add_task_command(args)


def _run_team_send_command(args: argparse.Namespace) -> int:
    return _team_service_run_team_send_command(args)


def _run_team_broadcast_command(args: argparse.Namespace) -> int:
    return _team_service_run_team_broadcast_command(args)


def _run_team_shutdown_command(args: argparse.Namespace) -> int:
    return _team_service_run_team_shutdown_command(args)


def _run_tools_command(args: argparse.Namespace) -> int:
    print(render_tools(verbose=getattr(args, 'verbose', False)))
    return 0


def _run_skills_command(_: argparse.Namespace) -> int:
    print(render_skills())
    return 0


def _run_replay_command(args: argparse.Namespace) -> int:
    events_file = resolve_events_file(
        events_file=getattr(args, 'events_file', ''),
        session_id=getattr(args, 'session_id', ''),
        sessions_dir=str(args.sessions_dir),
    )
    if not events_file.exists():
        print(f'events file not found: {events_file}')
        return 1
    print(render_replay(events_file=events_file, pretty=getattr(args, 'pretty', False), limit=getattr(args, 'limit', None)))
    return 0


def _run_doctor_command(args: argparse.Namespace) -> int:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='anvil',
        description='Run Anvil as a tool-use feedback loop: model decides, tools execute, results feed back.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    code = subparsers.add_parser(
        'code',
        help='run the coding tool-use loop',
        description='Run the coding runtime as a tool-use feedback loop over a workspace.',
        epilog=(
            'Examples:\n'
            '  anvil code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3\n'
            '  anvil code --goal-file goal.txt --workspace . --provider openai_compatible --model gpt-5.3-codex \\\n'
            '    --base-url https://codex-api.packycode.com/v1 --wire-api responses --output json\n'
            '  anvil code --goal "search docs then summarize" --workspace . --skill web_search --skill memory\n'
            '  anvil code --goal "fix tests" --workspace . --observer-file events.jsonl --memory-dir .anvil/runs\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    goal_group = code.add_mutually_exclusive_group(required=False)
    goal_group.add_argument('--goal', help='Direct goal text for the tool-use loop')
    goal_group.add_argument('--goal-file', help='UTF-8 goal file for the tool-use loop')

    execution_group = code.add_argument_group('execution')
    execution_group.add_argument('--workspace', default='.', help='Workspace root available to tools')
    execution_group.add_argument('--max-steps', type=int, default=12, help='Maximum tool-use rounds before stopping')
    execution_group.add_argument('--timeout-s', type=float, default=120.0, help='Maximum elapsed seconds for the run')
    execution_group.add_argument('--output', choices=['text', 'json'], default='text')
    execution_group.add_argument('--include-history', action='store_true')
    execution_group.add_argument('--session-id', default='', help='Resume or append to an existing session id')
    execution_group.add_argument('--sessions-dir', default='.anvil/sessions', help='Root directory for persisted sessions')
    execution_group.add_argument('--permission-mode', choices=['strict', 'balanced', 'unsafe'], default='balanced')

    provider_group = code.add_argument_group('provider')
    provider_group.add_argument(
        '--provider',
        choices=['mock', 'openai_compatible', 'anthropic', 'gemini'],
        default='mock',
    )
    provider_group.add_argument('--model', default='mock-model')
    provider_group.add_argument('--base-url', default='')
    provider_group.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='chat_completions')
    provider_group.add_argument('--api-key-env', default='OPENAI_API_KEY')
    provider_group.add_argument('--temperature', type=float, default=0.2)
    provider_group.add_argument('--provider-timeout-s', type=float, default=60.0)
    provider_group.add_argument('--provider-debug', action='store_true')
    provider_group.add_argument('--fallback-model', action='append', default=[])
    provider_group.add_argument('--max-retries', type=int, default=2)
    provider_group.add_argument('--retry-backoff-s', type=float, default=1.0)
    provider_group.add_argument('--retry-http-code', action='append', type=int, default=[])
    provider_group.add_argument(
        '--provider-header',
        action='append',
        default=[],
        help='provider extra header Key:Value, repeatable',
    )

    memory_group = code.add_argument_group('memory and artifacts')
    memory_group.add_argument('--history-window', type=int, default=8, help='How many recent loop outputs to feed back')
    memory_group.add_argument('--observer-file', help='Write observer events as JSONL')
    memory_group.add_argument('--memory-dir', default='.anvil/runs', help='Persistent memory root for run state')
    memory_group.add_argument('--run-id', help='Optional run id for memory and artifacts')
    memory_group.add_argument('--summarize-every', type=int, default=5, help='Refresh memory summary every N rounds')
    memory_group.add_argument('--record-run', action='store_true', default=True)
    memory_group.add_argument('--no-record-run', action='store_false', dest='record_run')
    memory_group.add_argument('--runs-dir', default='.anvil/runs', help='Directory for structured run artifacts')
    memory_group.add_argument('--tasks-dir', default='.tasks', help='Workspace-relative task graph directory injected into context')
    memory_group.add_argument('--transcripts-dir', default='.transcripts', help='Workspace-relative archive directory for compacted transcripts')
    memory_group.add_argument('--max-context-tokens', type=int, default=50000, help='Auto-compact when estimated context exceeds this budget')
    memory_group.add_argument('--micro-compact-keep', type=int, default=3, help='Keep full content for the most recent N tool results')
    memory_group.add_argument('--recent-transcript-entries', type=int, default=8, help='Expose the most recent compacted transcript entries in state summary')

    tool_group = code.add_argument_group('tool dispatch')
    tool_group.add_argument(
        '--skill',
        action='append',
        default=[],
        dest='skills',
        help='Skills to load into the tool dispatch (web_search, memory, files, commands, browser, or "all")',
    )

    register_tools_parser(subparsers, handler=_run_tools_command)
    register_skills_parser(subparsers, handler=_run_skills_command)
    register_replay_parser(subparsers, handler=_run_replay_command)

    team = subparsers.add_parser(
        'team',
        help='run or control persistent teammates',
        description='Manage persistent teammate loops backed by .team/config.json and .team/inbox/*.jsonl',
    )
    team_subparsers = team.add_subparsers(dest='team_command', required=True)

    team_run = team_subparsers.add_parser(
        'run',
        help='start teammate loops in the current process',
        description='Spawn persistent teammates, deliver messages, and keep threads alive until idle or timeout.',
    )
    team_run.add_argument('--workspace', default='.', help='Workspace root for teammate execution')
    team_run.add_argument('--team-dir', default='.team', help='Workspace-relative team runtime directory')
    team_run.add_argument('--teammate', action='append', default=[], required=True, help='Teammate in NAME:ROLE format')
    team_run.add_argument('--message', action='append', default=[], help='Send startup message TARGET=BODY')
    team_run.add_argument('--broadcast', action='append', default=[], help='Broadcast startup message to all teammates')
    team_run.add_argument('--task', action='append', default=[], help='Create and dispatch one task goal per entry')
    team_run.add_argument('--sender', default='lead')
    team_run.add_argument('--service-timeout-s', type=float, default=5.0)
    team_run.add_argument('--poll-interval-s', type=float, default=0.05)
    team_run.add_argument('--output', choices=['text', 'json'], default='text')
    team_run.add_argument('--max-steps', type=int, default=12)
    team_run.add_argument('--timeout-s', type=float, default=120.0)
    team_run.add_argument(
        '--provider',
        choices=['mock', 'openai_compatible', 'anthropic', 'gemini'],
        default='mock',
    )
    team_run.add_argument('--model', default='mock-model')
    team_run.add_argument('--base-url', default='')
    team_run.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='chat_completions')
    team_run.add_argument('--api-key-env', default='OPENAI_API_KEY')
    team_run.add_argument('--temperature', type=float, default=0.2)
    team_run.add_argument('--provider-timeout-s', type=float, default=60.0)
    team_run.add_argument('--provider-debug', action='store_true')
    team_run.add_argument('--fallback-model', action='append', default=[])
    team_run.add_argument('--max-retries', type=int, default=2)
    team_run.add_argument('--retry-backoff-s', type=float, default=1.0)
    team_run.add_argument('--retry-http-code', action='append', type=int, default=[])
    team_run.add_argument('--provider-header', action='append', default=[])
    team_run.add_argument('--history-window', type=int, default=8)
    team_run.add_argument('--skill', action='append', default=[], dest='skills')
    team_run.set_defaults(handler=_run_team_run_command)

    team_serve = team_subparsers.add_parser(
        'serve',
        help='start a long-lived team service',
        description='Spawn persistent teammates and keep the process alive to serve inbox traffic and task dispatch.',
    )
    team_serve.add_argument('--workspace', default='.', help='Workspace root for teammate execution')
    team_serve.add_argument('--team-dir', default='.team', help='Workspace-relative team runtime directory')
    team_serve.add_argument('--teammate', action='append', default=[], required=True, help='Teammate in NAME:ROLE format')
    team_serve.add_argument('--message', action='append', default=[], help='Send startup message TARGET=BODY')
    team_serve.add_argument('--broadcast', action='append', default=[], help='Broadcast startup message to all teammates')
    team_serve.add_argument('--task', action='append', default=[], help='Create startup task goals before entering service loop')
    team_serve.add_argument('--sender', default='lead')
    team_serve.add_argument('--service-timeout-s', type=float, default=300.0, help='Max wall time for the service; <=0 means no deadline')
    team_serve.add_argument('--idle-exit-s', type=float, default=0.0, help='Optional idle timeout before the service exits')
    team_serve.add_argument('--poll-interval-s', type=float, default=0.05)
    team_serve.add_argument('--output', choices=['text', 'json'], default='text')
    team_serve.add_argument('--max-steps', type=int, default=12)
    team_serve.add_argument('--timeout-s', type=float, default=120.0)
    team_serve.add_argument(
        '--provider',
        choices=['mock', 'openai_compatible', 'anthropic', 'gemini'],
        default='mock',
    )
    team_serve.add_argument('--model', default='mock-model')
    team_serve.add_argument('--base-url', default='')
    team_serve.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='chat_completions')
    team_serve.add_argument('--api-key-env', default='OPENAI_API_KEY')
    team_serve.add_argument('--temperature', type=float, default=0.2)
    team_serve.add_argument('--provider-timeout-s', type=float, default=60.0)
    team_serve.add_argument('--provider-debug', action='store_true')
    team_serve.add_argument('--fallback-model', action='append', default=[])
    team_serve.add_argument('--max-retries', type=int, default=2)
    team_serve.add_argument('--retry-backoff-s', type=float, default=1.0)
    team_serve.add_argument('--retry-http-code', action='append', type=int, default=[])
    team_serve.add_argument('--provider-header', action='append', default=[])
    team_serve.add_argument('--history-window', type=int, default=8)
    team_serve.add_argument('--skill', action='append', default=[], dest='skills')
    team_serve.set_defaults(handler=_run_team_serve_command)

    team_send = team_subparsers.add_parser('send', help='append one message to a teammate inbox')
    team_send.add_argument('--workspace', default='.')
    team_send.add_argument('--team-dir', default='.team')
    team_send.add_argument('--to', required=True)
    team_send.add_argument('--message', required=True)
    team_send.add_argument('--sender', default='lead')
    team_send.set_defaults(handler=_run_team_send_command)

    team_broadcast = team_subparsers.add_parser('broadcast', help='append one broadcast message to all teammate inboxes')
    team_broadcast.add_argument('--workspace', default='.')
    team_broadcast.add_argument('--team-dir', default='.team')
    team_broadcast.add_argument('--message', required=True)
    team_broadcast.add_argument('--sender', default='lead')
    team_broadcast.set_defaults(handler=_run_team_broadcast_command)

    team_shutdown = team_subparsers.add_parser('shutdown', help='request teammate shutdown')
    team_shutdown.add_argument('--workspace', default='.')
    team_shutdown.add_argument('--team-dir', default='.team')
    team_shutdown.add_argument('--sender', default='lead')
    team_shutdown.add_argument('--timeout-s', type=float, default=5.0)
    target_group = team_shutdown.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--to')
    target_group.add_argument('--all', action='store_true')
    team_shutdown.set_defaults(handler=_run_team_shutdown_command)

    team_add_task = team_subparsers.add_parser('add-task', help='append one task into the team task graph')
    team_add_task.add_argument('--workspace', default='.')
    team_add_task.add_argument('--team-dir', default='.team')
    team_add_task.add_argument('--goal', required=True)
    team_add_task.add_argument('--title')
    team_add_task.add_argument('--task-id')
    team_add_task.add_argument('--depends-on', action='append', default=[])
    team_add_task.add_argument('--assignee')
    team_add_task.add_argument('--role')
    team_add_task.add_argument('--sender', default='lead')
    team_add_task.add_argument('--output', choices=['text', 'json'], default='text')
    team_add_task.set_defaults(handler=_run_team_add_task_command)

    register_doctor_parser(subparsers, handler=_run_doctor_command)

    code.set_defaults(handler=_run_code_command)
    return parser


def main() -> None:
    from .entrypoints.agent import main as entrypoint_main

    entrypoint_main()


if __name__ == '__main__':
    main()
