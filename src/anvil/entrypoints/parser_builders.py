from __future__ import annotations

import argparse


def _add_common_provider_args(parser: argparse.ArgumentParser, *, default_provider: str = 'mock') -> None:
    parser.add_argument(
        '--provider',
        choices=['mock', 'openai_compatible', 'anthropic', 'gemini'],
        default=default_provider,
    )
    parser.add_argument('--model', default='mock-model' if default_provider == 'mock' else 'gpt-5.3-codex')
    parser.add_argument('--base-url', default='' if default_provider == 'mock' else None, required=default_provider != 'mock')
    parser.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='chat_completions' if default_provider == 'mock' else 'responses')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--provider-timeout-s', type=float, default=60.0 if default_provider == 'mock' else 20.0)
    parser.add_argument('--provider-debug', action='store_true')
    parser.add_argument('--fallback-model', action='append', default=[])
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--retry-backoff-s', type=float, default=1.0)
    parser.add_argument('--retry-http-code', action='append', type=int, default=[])
    parser.add_argument('--provider-header', action='append', default=[])


def register_code_parser(subparsers, *, handler) -> argparse.ArgumentParser:
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
    _add_common_provider_args(provider_group)

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
    code.set_defaults(handler=handler)
    return code


def register_team_parser(subparsers, *, run_handler, serve_handler, send_handler, broadcast_handler, shutdown_handler, add_task_handler) -> argparse.ArgumentParser:
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
    _add_common_provider_args(team_run)
    team_run.add_argument('--history-window', type=int, default=8)
    team_run.add_argument('--skill', action='append', default=[], dest='skills')
    team_run.set_defaults(handler=run_handler)

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
    _add_common_provider_args(team_serve)
    team_serve.add_argument('--history-window', type=int, default=8)
    team_serve.add_argument('--skill', action='append', default=[], dest='skills')
    team_serve.set_defaults(handler=serve_handler)

    team_send = team_subparsers.add_parser('send', help='append one message to a teammate inbox')
    team_send.add_argument('--workspace', default='.')
    team_send.add_argument('--team-dir', default='.team')
    team_send.add_argument('--to', required=True)
    team_send.add_argument('--message', required=True)
    team_send.add_argument('--sender', default='lead')
    team_send.set_defaults(handler=send_handler)

    team_broadcast = team_subparsers.add_parser('broadcast', help='append one broadcast message to all teammate inboxes')
    team_broadcast.add_argument('--workspace', default='.')
    team_broadcast.add_argument('--team-dir', default='.team')
    team_broadcast.add_argument('--message', required=True)
    team_broadcast.add_argument('--sender', default='lead')
    team_broadcast.set_defaults(handler=broadcast_handler)

    team_shutdown = team_subparsers.add_parser('shutdown', help='request teammate shutdown')
    team_shutdown.add_argument('--workspace', default='.')
    team_shutdown.add_argument('--team-dir', default='.team')
    team_shutdown.add_argument('--sender', default='lead')
    team_shutdown.add_argument('--timeout-s', type=float, default=5.0)
    target_group = team_shutdown.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--to')
    target_group.add_argument('--all', action='store_true')
    team_shutdown.set_defaults(handler=shutdown_handler)

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
    team_add_task.set_defaults(handler=add_task_handler)
    return team


def register_tools_parser(subparsers, *, handler) -> argparse.ArgumentParser:
    tools = subparsers.add_parser(
        'tools',
        help='list tools exposed to the loop',
        description='List tool handlers available to the coding tool-use loop.',
    )
    tools.add_argument('--verbose', action='store_true', help='Show tool descriptions and capability metadata')
    tools.set_defaults(handler=handler)
    return tools


def register_skills_parser(subparsers, *, handler) -> argparse.ArgumentParser:
    skills_parser = subparsers.add_parser(
        'skills',
        help='list available skills',
        description='List optional skills that can extend the loop tool dispatch.',
    )
    skills_parser.set_defaults(handler=handler)
    return skills_parser


def register_replay_parser(subparsers, *, handler) -> argparse.ArgumentParser:
    replay = subparsers.add_parser(
        'replay',
        help='print recorded loop events',
        description='Print a recorded JSONL event stream for a previous tool-use run.',
    )
    replay.add_argument('--events-file', default='')
    replay.add_argument('--session-id', default='')
    replay.add_argument('--sessions-dir', default='.anvil/sessions')
    replay.add_argument('--pretty', action='store_true', help='Render a human-readable event stream')
    replay.add_argument('--limit', type=int, help='Limit pretty replay to the most recent N events')
    replay.add_argument(
        '--filter',
        choices=['all', 'errors-only', 'tool-only', 'permissions-only'],
        default='all',
        help='Filter pretty replay output for debugging specific event classes',
    )
    replay.set_defaults(handler=handler)
    return replay


def register_doctor_parser(subparsers, *, handler) -> argparse.ArgumentParser:
    doctor = subparsers.add_parser(
        'doctor',
        help='diagnose provider connectivity',
        description='Diagnose provider connectivity before running the tool-use loop.',
    )
    doctor.add_argument('--provider', choices=['openai_compatible'], default='openai_compatible')
    doctor.add_argument('--model', default='gpt-5.3-codex')
    doctor.add_argument('--base-url', required=True)
    doctor.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='responses')
    doctor.add_argument('--api-key-env', default='OPENAI_API_KEY')
    doctor.add_argument('--provider-timeout-s', type=float, default=20.0)
    doctor.add_argument('--provider-header', action='append', default=[])
    doctor.add_argument('--output', choices=['text', 'json'], default='text')
    doctor.set_defaults(handler=handler)
    return doctor
