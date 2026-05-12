from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List

from ..coding_agent import run_coding_agent
from ..core.types import StopConfig
from ..llm.providers import build_invoke_from_args
from ..runtime import CodeRuntime
from ..session import SessionStore
from ..tools import builtin_tool_specs
from .chat_runtime import InteractiveRuntime
from .coding_runtime import build_coding_decider, build_coding_summarizer, load_skills_from_args


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


def _extract_interactive_output(payload: dict) -> str:
    final_output = str(payload.get('final_output') or '').strip()
    if final_output:
        return final_output

    error = str(payload.get('error') or '').strip()
    if error:
        return f'Run failed: {error}'

    history = payload.get('history')
    if isinstance(history, list):
        for item in reversed(history):
            text = str(item or '').strip()
            if text:
                return text

    last_steps = payload.get('memory_last_steps')
    if isinstance(last_steps, list):
        for item in reversed(last_steps):
            text = str(item or '').strip()
            if text:
                return text

    stop_reason = str(payload.get('stop_reason') or 'unknown').strip()
    return f'Stopped without final output (reason: {stop_reason}).'


def _format_chat_history(history_tail: list[str], *, current_user_text: str, limit: int = 12) -> str:
    filtered = list(history_tail)
    current_entry = f'user: {current_user_text}'
    if filtered and filtered[-1] == current_entry:
        filtered = filtered[:-1]
    return '\n'.join(filtered[-limit:])


def _run_plain_chat_fallback(
    base_args: argparse.Namespace,
    user_text: str,
    *,
    history_tail: list[str] | None = None,
) -> str:
    if str(getattr(base_args, 'provider', 'mock')) == 'mock':
        return ''
    invoke = build_invoke_from_args(base_args, mode='chat')
    history = _format_chat_history(history_tail or [], current_user_text=user_text)
    prompt = (
        'You are Anvil, an interactive terminal coding assistant. '
        'Answer the user directly and concisely. Use the conversation history when it is relevant.\n\n'
        + (f'Conversation history:\n{history}\n\n' if history else '')
        + 'User:\n'
        + user_text
    )
    return invoke(prompt).strip()


def _should_use_plain_chat_fallback(output: str) -> bool:
    text = output.strip()
    if text.startswith('Stopped without final output'):
        return True
    provider_format_errors = (
        'invalid agent step json',
        'invalid Anthropic response format',
        'invalid openai-compatible response',
        'invalid Gemini response format',
    )
    return any(item in text for item in provider_format_errors)


def _looks_like_action_request(user_text: str) -> bool:
    normalized = user_text.lower()
    action_tokens = (
        '\u65b0\u589e',
        '\u521b\u5efa',
        '\u65b0\u5efa',
        '\u5199\u5165',
        '\u5199\u5230',
        '\u4fee\u6539',
        '\u5220\u9664',
        'create',
        'write',
        'edit',
        'delete',
    )
    target_tokens = (
        '\u6587\u4ef6',
        '\u6587\u4ef6\u5939',
        '\u76ee\u5f55',
        '.txt',
        '.md',
        '.json',
        'file',
        'folder',
        'directory',
    )
    return any(token in normalized for token in action_tokens) and any(
        token in normalized for token in target_tokens
    )


def build_interactive_turn_runner(base_args: argparse.Namespace, *, session_id: str):
    def run_turn(user_text: str) -> str:
        turn_args = copy.deepcopy(base_args)
        turn_args.interactive_trusted_workspace = True
        turn_args.session_id = session_id
        turn_args.goal = user_text
        turn_args.goal_file = ''
        runtime = CodeRuntime(turn_args, goal=user_text)
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
        output = _extract_interactive_output(payload)
        if _should_use_plain_chat_fallback(output):
            if _looks_like_action_request(user_text):
                return output
            fallback = _run_plain_chat_fallback(
                base_args,
                user_text,
                history_tail=list(runtime.session_store.state.history_tail),
            )
            if fallback:
                return fallback
        return output

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
    runtime = InteractiveRuntime(
        session_store=session_store,
        tool_specs=builtin_tool_specs(),
        run_turn=build_interactive_turn_runner(args, session_id=session_store.state.session_id),
        stdin=sys.stdin,
        stdout=sys.stdout,
        model=str(args.model),
        permission_mode=str(args.permission_mode),
    )
    return runtime.run()
