from __future__ import annotations

import json
from pathlib import Path

from loop_agent.coding_agent import run_coding_agent
from loop_agent.core.serialization import run_result_to_dict
from loop_agent.core.types import StopConfig
from loop_agent.llm.providers import _mock_invoke_factory


def build_mock_decider():
    invoke = _mock_invoke_factory('mock-v3', mode='coding')

    def decider(goal, history, tool_results, state_summary, last_steps):
        prompt = {
            'goal': goal,
            'history': list(history),
            'tool_results': [result.id for result in tool_results],
            'state_summary': state_summary,
            'last_steps': list(last_steps),
        }
        return invoke(json.dumps(prompt, ensure_ascii=False))

    return decider


def main() -> None:
    result = run_coding_agent(
        goal='inspect README and finish',
        decider=build_mock_decider(),
        workspace_root=Path.cwd(),
        stop=StopConfig(max_steps=4, max_elapsed_s=10.0),
    )
    payload = run_result_to_dict(result, include_history=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
