from __future__ import annotations

from loop_agent.api import run_goal


def main() -> None:
    result = run_goal(
        'Answer in the built-in JSON loop protocol and finish when ready.',
        provider='mock',
        model='mock-v3',
        max_steps=4,
    )
    print(f'success={result.success}')
    print(f'steps={result.steps}')
    print(f'stop_reason={result.stop_reason}')
    print(f'output={result.output}')


if __name__ == '__main__':
    main()
