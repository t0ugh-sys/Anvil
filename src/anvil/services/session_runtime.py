from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List

from ..coding_agent import run_coding_agent
from ..core.types import StopConfig
from ..runtime import CodeRuntime
from ..session import SessionStore
from ..tools import builtin_tool_specs
from .chat_runtime import InteractiveRuntime, TurnExecution
from .coding_runtime import build_coding_decider, build_coding_summarizer, load_skills_from_args
from .runtime_config import RuntimeConfigManager


def build_interactive_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='anvil',
        description='Run Anvil as an interactive terminal coding agent runtime.',
    )
    parser.add_argument('--workspace', default='.', help='Workspace root available to tools')
    parser.add_argument('--session-id', default='', help='Resume an existing interactive session id')
    parser.add_argument('--sessions-dir', default='.anvil/sessions', help='Root directory for persisted sessions')
    parser.add_argument('--permission-mode', choices=['strict', 'balanced', 'unsafe'], default='balanced')
    parser.add_argument(
        '--provider',
        choices=['mock', 'openai_compatible', 'anthropic', 'gemini'],
        default='mock',
    )
    parser.add_argument('--model', default='mock-model')
    parser.add_argument('--base-url', default='')
    parser.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='chat_completions')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--provider-timeout-s', type=float, default=60.0)
    parser.add_argument('--provider-debug', action='store_true')
    parser.add_argument('--fallback-model', action='append', default=[])
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--retry-backoff-s', type=float, default=1.0)
    parser.add_argument('--retry-http-code', action='append', type=int, default=[])
    parser.add_argument('--provider-header', action='append', default=[])
    parser.add_argument('--max-steps', type=int, default=12)
    parser.add_argument('--timeout-s', type=float, default=120.0)
    parser.add_argument('--history-window', type=int, default=8)
    parser.add_argument('--memory-dir', default='.anvil/runs')
    parser.add_argument('--run-id')
    parser.add_argument('--summarize-every', type=int, default=5)
    parser.add_argument('--record-run', action='store_true', default=True)
    parser.add_argument('--no-record-run', action='store_false', dest='record_run')
    parser.add_argument('--runs-dir', default='.anvil/runs')
    parser.add_argument('--observer-file')
    parser.add_argument('--include-history', action='store_true')
    parser.add_argument('--tasks-dir', default='.tasks')
    parser.add_argument('--transcripts-dir', default='.transcripts')
    parser.add_argument('--max-context-tokens', type=int, default=50000)
    parser.add_argument('--micro-compact-keep', type=int, default=3)
    parser.add_argument('--recent-transcript-entries', type=int, default=8)
    parser.add_argument('--output', choices=['text', 'json'], default='text')
    parser.add_argument(
        '--skill',
        action='append',
        default=[],
        dest='skills',
        help='Skills to load into the tool dispatch',
    )
    return parser


def should_launch_interactive(argv: List[str]) -> bool:
    if not argv:
        return True
    first = argv[0]
    if first in {'-h', '--help'}:
        return False
    return first not in {'code', 'tools', 'skills', 'replay', 'team', 'doctor'}


def build_interactive_turn_runner(base_args: argparse.Namespace, *, session_id: str, runtime_config_manager: RuntimeConfigManager):
    def run_turn(user_text: str) -> TurnExecution:
        turn_args = copy.deepcopy(base_args)
        runtime_config_manager.apply_to_args(turn_args)
        turn_args.session_id = session_id
        turn_args.goal = user_text
        turn_args.goal_file = ''
        runtime = CodeRuntime(turn_args, goal=user_text)
        permission_before = dict(runtime.session_store.state.permission_stats)
        skills = load_skills_from_args(turn_args)
        decider = build_coding_decider(turn_args, skills)
        summarizer = build_coding_summarizer(turn_args)
        if runtime.observer is not None:
            runtime.observer('run_started', {'goal': runtime.goal, 'strategy': 'coding', 'facts': []})
        result = run_coding_agent(
            goal=runtime.goal,
            decider=decider,
            workspace_root=runtime.workspace_root,
            stop=StopConfig(max_steps=turn_args.max_steps, max_elapsed_s=turn_args.timeout_s),
            observer=runtime.observer,
            context_provider=runtime.build_context_provider(),
            skills=skills,
            policy=runtime.build_policy(),
            task_store=runtime.task_store,
            compression_config=runtime.compression_config,
            transcripts_dir=runtime.transcripts_dir,
            summarizer=summarizer,
        )
        payload = runtime.finalize(result)
        permission_after = payload.get('permission_stats', {})
        permission_delta = {}
        if isinstance(permission_after, dict):
            for key in ('allow', 'deny', 'ask'):
                after_value = int(permission_after.get(key, 0) or 0)
                before_value = int(permission_before.get(key, 0) or 0)
                permission_delta[key] = max(0, after_value - before_value)
        return TurnExecution(
            output=str(payload.get('final_output', '') or ''),
            stop_reason=str(payload.get('stop_reason', '') or ''),
            steps=int(payload.get('steps', 0) or 0),
            permission_delta=permission_delta,
            blocked_tool_name=runtime.session_store.state.last_blocked_tool_name,
            blocked_tool_reason=runtime.session_store.state.last_blocked_tool_reason,
        )

    return run_turn


def run_interactive_command(args: argparse.Namespace, *, default_run_id: str) -> int:
    workspace_root = Path(args.workspace).resolve()
    sessions_root = Path(args.sessions_dir)
    if not sessions_root.is_absolute():
        if str(args.sessions_dir) == '.anvil/sessions':
            sessions_root = (workspace_root / sessions_root).resolve()
        else:
            sessions_root = sessions_root.resolve()
    if args.session_id:
        session_store = SessionStore.load(root_dir=sessions_root, session_id=args.session_id)
    else:
        session_store = SessionStore.create(
            root_dir=sessions_root,
            workspace_root=workspace_root,
            goal='',
            memory_run_dir=Path(args.memory_dir) / (args.run_id or default_run_id),
        )
    runtime_config_manager = RuntimeConfigManager.from_args(session_store=session_store, args=args)
    runtime = InteractiveRuntime(
        session_store=session_store,
        tool_specs=builtin_tool_specs(),
        run_turn=build_interactive_turn_runner(
            args,
            session_id=session_store.state.session_id,
            runtime_config_manager=runtime_config_manager,
        ),
        runtime_config_manager=runtime_config_manager,
        stdin=sys.stdin,
        stdout=sys.stdout,
    )
    return runtime.run()
