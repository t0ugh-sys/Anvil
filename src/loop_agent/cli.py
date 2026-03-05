from __future__ import annotations

import argparse
import sys

from .core.agent import LoopAgent
from .core.types import StopConfig
from .steps.registry import StepRegistry, build_default_registry


def build_parser(registry: StepRegistry) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='LoopAgent')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--goal', help='用户目标/需求描述（短文本，尽量避免复杂编码问题）')
    group.add_argument('--goal-file', help='从 UTF-8 文件读取目标（推荐用于中文/长文本）')
    parser.add_argument('--strategy', choices=registry.names(), default='demo')
    parser.add_argument('--history-window', type=int, default=3, help='JSON 策略下带入历史输出条数')
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--timeout-s', type=float, default=60.0)
    return parser


def resolve_goal(args: argparse.Namespace) -> str:
    if args.goal_file:
        with open(args.goal_file, 'r', encoding='utf-8-sig') as file:
            goal = file.read().strip()
    else:
        goal = args.goal
    if not goal.strip():
        raise ValueError('goal must not be empty')
    return goal


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]

    registry = build_default_registry()
    args = build_parser(registry).parse_args()
    goal = resolve_goal(args)
    step, initial_state = registry.create(args.strategy, args)

    agent = LoopAgent(step=step, stop=StopConfig(max_steps=args.max_steps, max_elapsed_s=args.timeout_s))
    result = agent.run(goal=goal, initial_state=initial_state)

    print('done:', result.done)
    print('stop_reason:', result.stop_reason.value)
    print('steps:', result.steps)
    print('final_output:', result.final_output)


if __name__ == '__main__':
    main()
