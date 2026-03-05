from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent_protocol import render_agent_step_schema
from .coding_agent import run_coding_agent
from .core.serialization import run_result_to_dict
from .core.types import StopConfig
from .llm.providers import build_invoke_from_args
from .tools import build_default_tools


def _resolve_goal(args: argparse.Namespace) -> str:
    if args.goal_file:
        return Path(args.goal_file).read_text(encoding='utf-8-sig').strip()
    return args.goal.strip()


def _build_coding_decider(args: argparse.Namespace):
    invoke = build_invoke_from_args(args, mode='coding')

    def decider(goal: str, history: tuple[str, ...], tool_results) -> str:
        prompt = (
            'You are a coding agent. Return strict JSON matching schema:\n'
            + render_agent_step_schema()
            + '\nGoal:\n'
            + goal
            + '\nHistory:\n'
            + str(list(history[-8:]))
            + '\nToolResults:\n'
            + str([{'id': r.id, 'ok': r.ok, 'output': r.output[:500], 'error': r.error} for r in tool_results])
            + '\nOnly output JSON.'
        )
        return invoke(prompt)

    return decider


def _run_code_command(args: argparse.Namespace) -> int:
    goal = _resolve_goal(args)
    workspace_root = Path(args.workspace).resolve()
    decider = _build_coding_decider(args)
    result = run_coding_agent(
        goal=goal,
        decider=decider,
        workspace_root=workspace_root,
        stop=StopConfig(max_steps=args.max_steps, max_elapsed_s=args.timeout_s),
    )

    payload = run_result_to_dict(result, include_history=True)
    payload['workspace'] = str(workspace_root)
    payload['provider'] = args.provider
    payload['model'] = args.model
    if args.output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"done: {result.done}")
        print(f"stop_reason: {result.stop_reason.value}")
        print(f"steps: {result.steps}")
        print(f"final_output: {result.final_output}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='openclaw_cli')
    subparsers = parser.add_subparsers(dest='command', required=True)

    code = subparsers.add_parser('code', help='run coding agent loop')
    goal_group = code.add_mutually_exclusive_group(required=True)
    goal_group.add_argument('--goal')
    goal_group.add_argument('--goal-file')
    code.add_argument('--workspace', default='.')
    code.add_argument('--provider', choices=['mock', 'openai_compatible'], default='mock')
    code.add_argument('--model', default='mock-model')
    code.add_argument('--base-url', default='')
    code.add_argument('--api-key-env', default='OPENAI_API_KEY')
    code.add_argument('--temperature', type=float, default=0.2)
    code.add_argument('--provider-timeout-s', type=float, default=60.0)
    code.add_argument('--max-steps', type=int, default=12)
    code.add_argument('--timeout-s', type=float, default=120.0)
    code.add_argument('--output', choices=['text', 'json'], default='text')

    tools = subparsers.add_parser('tools', help='list available tools')
    tools.set_defaults(handler=_run_tools_command)

    replay = subparsers.add_parser('replay', help='print events jsonl')
    replay.add_argument('--events-file', required=True)
    replay.set_defaults(handler=_run_replay_command)

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

