from __future__ import annotations

from pathlib import Path

from loop_agent.api import AgentConfig, LoopAgentAPI
from loop_agent.llm.providers import _mock_invoke_factory


def main() -> None:
    api = LoopAgentAPI(
        AgentConfig(
            provider='mock',
            model='mock-v3',
            max_steps=4,
            timeout_s=10.0,
            history_window=4,
            workspace=Path.cwd(),
        )
    )
    api.set_provider(_mock_invoke_factory('mock-v3', mode='json'))
    result = api.run('Summarize what this harness does and stop when complete.')
    print(result.to_dict())


if __name__ == '__main__':
    main()
