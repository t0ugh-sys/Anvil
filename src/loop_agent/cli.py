from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core.agent import LoopAgent
from .core.serialization import run_result_to_dict, run_result_to_json
from .core.types import ObserverFn, RunResult, StopConfig, StopReason
from .run_recorder import RunRecorder
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
    parser.add_argument('--output', choices=['text', 'json'], default='text')
    parser.add_argument('--include-history', action='store_true', help='JSON 输出时是否包含 history')
    parser.add_argument('--observer-file', help='将事件回调按 JSONL 写入指定文件')
    parser.add_argument('--exit-on-failure', action='store_true', help='当未完成时返回非零退出码')
    parser.add_argument('--record-run', action='store_true', default=True, help='记录本次运行到 runs 目录（默认开启）')
    parser.add_argument('--no-record-run', action='store_false', dest='record_run', help='关闭本次运行记录')
    parser.add_argument('--runs-dir', default='runs', help='运行记录根目录')
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


def should_exit_failure(result: RunResult[Any]) -> bool:
    failure_reasons = {StopReason.timeout, StopReason.max_steps, StopReason.step_error}
    return (not result.done) and (result.stop_reason in failure_reasons)


def build_jsonl_observer(path: str) -> ObserverFn:
    def observer(event: str, payload: dict[str, Any]) -> None:
        record = {'event': event, 'payload': payload}
        with open(path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write('\n')

    return observer


def merge_observers(observers: list[ObserverFn]) -> ObserverFn | None:
    active = [item for item in observers if item is not None]
    if not active:
        return None

    def merged(event: str, payload: dict[str, Any]) -> None:
        for observer in active:
            observer(event, payload)

    return merged


def execute(args: argparse.Namespace, registry: StepRegistry) -> tuple[str, int]:
    goal = resolve_goal(args)
    step, initial_state = registry.create(args.strategy, args)
    recorder: RunRecorder | None = None
    observers: list[ObserverFn] = []
    if args.observer_file:
        observers.append(build_jsonl_observer(args.observer_file))
    if args.record_run:
        recorder = RunRecorder.create(base_dir=Path(args.runs_dir))
        observers.append(recorder.write_event)
    observer = merge_observers(observers)

    agent = LoopAgent(step=step, stop=StopConfig(max_steps=args.max_steps, max_elapsed_s=args.timeout_s))
    result = agent.run(goal=goal, initial_state=initial_state, observer=observer)
    if recorder is not None:
        recorder.write_summary(run_result_to_dict(result, include_history=True))

    if args.output == 'json':
        if recorder is None:
            rendered = run_result_to_json(result, include_history=args.include_history)
        else:
            payload = run_result_to_dict(result, include_history=args.include_history)
            payload['run_dir'] = str(recorder.run_dir)
            rendered = json.dumps(payload, ensure_ascii=False)
    else:
        lines = [
            f'done: {result.done}',
            f'stop_reason: {result.stop_reason.value}',
            f'steps: {result.steps}',
            f'final_output: {result.final_output}',
        ]
        if recorder is not None:
            lines.append(f'run_dir: {recorder.run_dir}')
        rendered = '\n'.join(
            [
                *lines,
            ]
        )

    exit_code = 1 if (args.exit_on_failure and should_exit_failure(result)) else 0
    return rendered, exit_code


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]

    registry = build_default_registry()
    args = build_parser(registry).parse_args()
    rendered, exit_code = execute(args, registry)
    print(rendered)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
