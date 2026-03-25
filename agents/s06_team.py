from __future__ import annotations

import json
import shutil
from pathlib import Path

from loop_agent.llm.providers import _mock_invoke_factory
from loop_agent.mailbox import JsonlMailbox
from loop_agent.subagents import SubAgentRuntime, SubAgentSpec
from loop_agent.task_graph import Task, TaskGraph


def build_mock_decider():
    invoke = _mock_invoke_factory('mock-v3', mode='coding')

    def decider(goal, history, tool_results, state_summary, last_steps):
        payload = {
            'goal': goal,
            'history': list(history),
            'last_steps': list(last_steps),
            'state_summary': state_summary,
        }
        return invoke(json.dumps(payload, ensure_ascii=False))

    return decider


def main() -> None:
    workspace = Path.cwd()
    mailbox_dir = Path('.loopagent/demo-team')
    shutil.rmtree(mailbox_dir, ignore_errors=True)

    graph = TaskGraph(
        [
            Task(id='task_readme', title='Inspect README', goal='inspect README and finish'),
            Task(
                id='task_summary',
                title='Produce summary',
                goal='summarize what was inspected',
                dependencies=('task_readme',),
            ),
        ]
    )
    mailbox = JsonlMailbox(mailbox_dir)
    runtime = SubAgentRuntime(mailbox=mailbox, task_graph=graph)
    worker = SubAgentSpec(agent_id='worker-1', role='reader', workspace_root=workspace, skills=('files',))

    first = runtime.dispatch_ready_tasks(specs=(worker,), decider=build_mock_decider())
    second = runtime.dispatch_ready_tasks(specs=(worker,), decider=build_mock_decider())

    print(f'first_batch={len(first)} second_batch={len(second)}')
    print(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))
    print(json.dumps(mailbox.summary_for('coordinator'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
