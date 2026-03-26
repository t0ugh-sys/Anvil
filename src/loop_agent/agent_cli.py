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
from .llm.providers import build_invoke_from_args
from .memory.jsonl_store import JsonlMemoryStore
from .ops.doctor import format_doctor_report, run_provider_doctor
from .run_recorder import RunRecorder
from .skills import SkillLoader, list_skills, get_skill
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
            'You are a coding agent. Return strict JSON matching schema.\n'
            'Use tools when needed. Keep a visible todo list updated via the todo_write tool when progress changes.\n'
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


def _load_skills_from_args(args: argparse.Namespace) -> SkillLoader | None:
    """Load skills from command-line arguments."""
    skills_arg = getattr(args, 'skills', None)
    if not skills_arg:
        return None
    
    loader = SkillLoader()
    for skill_name in skills_arg:
        if skill_name == 'all':
            # Load all built-in skills
            for name in list_skills():
                loader.load(name)
        else:
            if not loader.load(skill_name):
                print(f"Warning: Unknown skill '{skill_name}' - skipping")
    return loader


def _build_jsonl_observer(path: str) -> ObserverFn:
    def observer(event: str, payload: Dict[str, Any]) -> None:
        record = {'event': event, 'payload': payload}
        with open(path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write('\n')

    return observer


def _merge_observers(observers: List[ObserverFn]) -> Optional[ObserverFn]:
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

    recorder: Optional[RunRecorder] = None
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
    skills = _load_skills_from_args(args)
    result = run_coding_agent(
        goal=goal,
        decider=decider,
        workspace_root=workspace_root,
        stop=StopConfig(max_steps=args.max_steps, max_elapsed_s=args.timeout_s),
        observer=observer,
        context_provider=context_provider,
        skills=skills,
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


def _run_skills_command(_: argparse.Namespace) -> int:
    """List all available skills."""
    skills = list_skills()
    print("Available skills:")
    for name in skills:
        skill = get_skill(name)
        if skill:
            print(f"  - {name}: {skill.description}")
    print("\nUse --skill <name> to load specific skills")
    print("Use --skill all to load all skills")
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
    parser = argparse.ArgumentParser(
        prog='agent_cli',
        description='Run LoopAgent as a tool-use feedback loop: model decides, tools execute, results feed back.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    code = subparsers.add_parser(
        'code',
        help='run the coding tool-use loop',
        description='Run the coding runtime as a tool-use feedback loop over a workspace.',
        epilog=(
            'Examples:\n'
            '  agent_cli code --goal "inspect README then finish" --workspace . --provider mock --model mock-v3\n'
            '  agent_cli code --goal-file goal.txt --workspace . --provider openai_compatible --model gpt-5.3-codex \\\n'
            '    --base-url https://codex-api.packycode.com/v1 --wire-api responses --output json\n'
            '  agent_cli code --goal "search docs then summarize" --workspace . --skill web_search --skill memory\n'
            '  agent_cli code --goal "fix tests" --workspace . --observer-file events.jsonl --memory-dir .loopagent/runs\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    goal_group = code.add_mutually_exclusive_group(required=True)
    goal_group.add_argument('--goal', help='Direct goal text for the tool-use loop')
    goal_group.add_argument('--goal-file', help='UTF-8 goal file for the tool-use loop')

    execution_group = code.add_argument_group('execution')
    execution_group.add_argument('--workspace', default='.', help='Workspace root available to tools')
    execution_group.add_argument('--max-steps', type=int, default=12, help='Maximum tool-use rounds before stopping')
    execution_group.add_argument('--timeout-s', type=float, default=120.0, help='Maximum elapsed seconds for the run')
    execution_group.add_argument('--output', choices=['text', 'json'], default='text')
    execution_group.add_argument('--include-history', action='store_true')

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
    memory_group.add_argument('--memory-dir', default='.loopagent/runs', help='Persistent memory root for run state')
    memory_group.add_argument('--run-id', help='Optional run id for memory and artifacts')
    memory_group.add_argument('--summarize-every', type=int, default=5, help='Refresh memory summary every N rounds')
    memory_group.add_argument('--record-run', action='store_true', default=True)
    memory_group.add_argument('--no-record-run', action='store_false', dest='record_run')
    memory_group.add_argument('--runs-dir', default='.loopagent/runs', help='Directory for structured run artifacts')

    tool_group = code.add_argument_group('tool dispatch')
    tool_group.add_argument(
        '--skill',
        action='append',
        default=[],
        dest='skills',
        help='Skills to load into the tool dispatch (web_search, memory, files, commands, browser, or "all")',
    )

    tools = subparsers.add_parser(
        'tools',
        help='list tools exposed to the loop',
        description='List tool handlers available to the coding tool-use loop.',
    )
    tools.set_defaults(handler=_run_tools_command)

    skills_parser = subparsers.add_parser(
        'skills',
        help='list available skills',
        description='List optional skills that can extend the loop tool dispatch.',
    )
    skills_parser.set_defaults(handler=_run_skills_command)

    replay = subparsers.add_parser(
        'replay',
        help='print recorded loop events',
        description='Print a recorded JSONL event stream for a previous tool-use run.',
    )
    replay.add_argument('--events-file', required=True)
    replay.set_defaults(handler=_run_replay_command)

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
