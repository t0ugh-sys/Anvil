from __future__ import annotations

import json
import shutil
from pathlib import Path

from loop_agent.task_graph import Task, TaskGraph
from loop_agent.task_store import TaskStore


def main() -> None:
    root = Path('.loopagent/demo-tasks')
    shutil.rmtree(root, ignore_errors=True)
    store = TaskStore(root / '.tasks')

    graph = TaskGraph(
        [
            Task(id='1', title='Inspect repo', goal='inspect repository layout'),
            Task(id='2', title='Edit docs', goal='update docs', dependencies=('1',)),
            Task(id='3', title='Run tests', goal='run tests after docs', dependencies=('2',)),
        ]
    )
    store.save_graph(graph)
    print('initial:')
    for path in store.list_task_files():
        print(path.name)
        print(path.read_text(encoding='utf-8'))

    graph.mark_completed('1')
    store.save_graph(graph)
    print('\nafter completing task 1:')
    print(json.dumps(store.load_graph().to_store_dict(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
