from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent_protocol import render_agent_step_schema
from .coding_agent import run_coding_agent
from .core.serialization import run_result_to_dict
from .core.types import ContextSnapshot, ObserverFn, StopConfig
from .doctor import format_doctor_report, run_provider_doctor
from .llm.providers import build_invoke_from_args
from .memory.jsonl_store import JsonlMemoryStore
from .run_recorder import RunRecorder
from .tools import build_default_tools


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _resolve_goal(args: argparse.Namespace) -> str:
    if args.goal_file:
        return Path(args.goal_file).read_text(encoding='utf-8-sig').strip()
    return args.goal.strip()


def _build_coding_decider(args: argparse.Namespace):
    invoke = build_invoke_from_args(args, mode='coding')

    def decider(
        goal: str,
        history: Tuple[str, ...],
        tool_results,
        state_summary: Dict[str, object],
        last_steps: Tuple[str, ...],
    ) -> str:
        history_window = max(1, args.history_window)
        prompt = (
            'You are a coding agent. Return strict JSON matching schema:\n'
            + render_agent_step_schema()
            + '\nGoal:\n'
            + goal
            + '\nHistory:\n'
            + str(list(history[-history_window:]))
            + '\nStateSummary:\n'
            + json.dumps(state_summary, ensure_ascii=False)
            + '\nLastSteps:\n'
            + str(list(last_steps))
            + '\nToolResults:\n'
            + str([{'id': r.id, 'ok': r.ok, 'output': r.output[:500], 'error': r.error} for r in tool_results])
            + '\nOnly output JSON.'
        )
        return invoke(prompt)

    return decider


def _build_jsonl_observer(path: str) -> ObserverFn:
    def observer(event: str, payload: Dict[str, Any]) -> None:
        record = {'event': event, 'payload': payload}
        with open(path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write('\n')

    return observer


def _merge_observers(observers: List[ObserverFn]) -> Optional[ObserverFn ]:
    active = [item for item in observers if item is not None]
    if not active:
        return None

    def merged(event: str, payload: Dict[str, Any]) -> None:
        for observer in active:
            observer(event, payload)

    return merged


def _run_code_command(args: argparse.Namespace) -> int:
    goal = _resolve_goal(args)
    workspace_root = Path(args.workspace).resolve()
    run_id = args.run_id or _default_run_id()
    memory_run_dir = Path(args.memory_dir) / run_id
    memory_store = JsonlMemoryStore(memory_dir=memory_run_dir, summarize_every=args.summarize_every)
    memory_store.on_event('run_started', {'goal': goal, 'strategy': 'coding', 'facts': []})

    recorder: Optional[RunRecorder ] = None
    observers: List[ObserverFn] = []
    if args.observer_file:
        observers.append(_build_jsonl_observer(args.observer_file))
    if args.record_run:
        recorder = RunRecorder.create(base_dir=Path(args.runs_dir))
        observers.append(recorder.write_event)
    observers.append(memory_store.on_event)
    observer = _merge_observers(observers)

    def context_provider() -> ContextSnapshot:
        context = memory_store.load_context(goal=goal, last_k_steps=args.history_window)
        return ContextSnapshot(state_summary=context.state_summary, last_steps=context.last_steps)

    decider = _build_coding_decider(args)
    result = run_coding_agent(
        goal=goal,
        decider=decider,
        workspace_root=workspace_root,
        stop=StopConfig(max_steps=args.max_steps, max_elapsed_s=args.timeout_s),
        observer=observer,
        context_provider=context_provider,
    )
    memory_store.on_event(
        'run_finished',
        {'done': result.done, 'stop_reason': result.stop_reason.value, 'steps': result.steps},
    )
    if recorder is not None:
        recorder.write_summary(run_result_to_dict(result, include_history=True))

    memory_context = memory_store.load_context(goal=goal, last_k_steps=args.history_window)

    payload = run_result_to_dict(result, include_history=args.include_history)
    payload['workspace'] = str(workspace_root)
    payload['provider'] = args.provider
    payload['model'] = args.model
    payload['memory_state'] = memory_context.state_summary
    payload['memory_last_steps'] = list(memory_context.last_steps)
    payload['memory_run_dir'] = str(memory_run_dir)
    if recorder is not None:
        payload['run_dir'] = str(recorder.run_dir)
    if args.output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"done: {result.done}")
        print(f"stop_reason: {result.stop_reason.value}")
        print(f"steps: {result.steps}")
        print(f"final_output: {result.final_output}")
        print(f"memory_run_dir: {memory_run_dir}")
        if recorder is not None:
            print(f"run_dir: {recorder.run_dir}")
    return 0 if result.done else 1


def _run_tools_command(_: argparse.Namespace) -> int:
    names = sorted(build_default_tools().keys())
    print('\n'.join(names))
    return 0


def _run_replay_command(args: argparse.Namespace) -> int:
    events_file = Path(args.events_file)
    if not events_file.exists():
        print(f'events file not found: {events_file}')
        return 1
    print(events_file.read_text(encoding='utf-8'))
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
    parser = argparse.ArgumentParser(prog='agent_cli')
    subparsers = parser.add_subparsers(dest='command', required=True)

    code = subparsers.add_parser('code', help='run coding agent loop')
    goal_group = code.add_mutually_exclusive_group(required=True)
    goal_group.add_argument('--goal')
    goal_group.add_argument('--goal-file')
    code.add_argument('--workspace', default='.')
    code.add_argument('--provider', choices=['mock', 'openai_compatible', 'anthropic', 'gemini'], default='mock')
    code.add_argument('--model', default='mock-model')
    code.add_argument('--base-url', default='')
    code.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='chat_completions')
    code.add_argument('--api-key-env', default='OPENAI_API_KEY')
    code.add_argument('--temperature', type=float, default=0.2)
    code.add_argument('--provider-timeout-s', type=float, default=60.0)
    code.add_argument('--provider-debug', action='store_true')
    code.add_argument('--fallback-model', action='append', default=[])
    code.add_argument('--max-retries', type=int, default=2)
    code.add_argument('--retry-backoff-s', type=float, default=1.0)
    code.add_argument('--retry-http-code', action='append', type=int, default=[])
    code.add_argument(
        '--provider-header',
        action='append',
        default=[],
        help='provider extra header Key:Value, repeatable',
    )
    code.add_argument('--max-steps', type=int, default=12)
    code.add_argument('--timeout-s', type=float, default=120.0)
    code.add_argument('--history-window', type=int, default=8)
    code.add_argument('--observer-file')
    code.add_argument('--memory-dir', default='.loopagent/runs')
    code.add_argument('--run-id')
    code.add_argument('--summarize-every', type=int, default=5)
    code.add_argument('--record-run', action='store_true', default=True)
    code.add_argument('--no-record-run', action='store_false', dest='record_run')
    code.add_argument('--runs-dir', default='runs')
    code.add_argument('--include-history', action='store_true')
    code.add_argument('--output', choices=['text', 'json'], default='text')

    tools = subparsers.add_parser('tools', help='list available tools')
    tools.set_defaults(handler=_run_tools_command)

    replay = subparsers.add_parser('replay', help='print events jsonl')
    replay.add_argument('--events-file', required=True)
    replay.set_defaults(handler=_run_replay_command)

    doctor = subparsers.add_parser('doctor', help='diagnose provider connectivity and gateway status')
    doctor.add_argument('--provider', choices=['openai_compatible'], default='openai_compatible')
    doctor.add_argument('--model', default='gpt-5.3-codex')
    doctor.add_argument('--base-url', required=True)
    doctor.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='responses')
    doctor.add_argument('--api-key-env', default='OPENAI_API_KEY')
    doctor.add_argument('--provider-timeout-s', type=float, default=20.0)
    doctor.add_argument('--provider-header', action='append', default=[])
    doctor.add_argument('--output', choices=['text', 'json'], default='text')
    doctor.set_defaults(handler=_run_doctor_command)

    code.set_defaults(handler=_run_code_command)
    return parser


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    parser = build_parser()
    args = parser.parse_args()
    code = args.handler(args)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
