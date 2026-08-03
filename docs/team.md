# Multi-Agent Team Runtime

`PersistentTeamRuntime` in `anvil/runtime/team.py` lets you run several coding
agents in parallel, coordinated through persistent message inboxes and a
dependency-aware task graph.

## Architecture

```
PersistentTeamRuntime
  ├── TeamConfigStore          (config.json — member registry)
  ├── JsonlTeamInboxStore      (inbox/<name>.jsonl — per-agent mailboxes)
  └── TaskStore                (tasks/ — dependency graph)
```

Each teammate runs in its own daemon thread, polling its inbox every 50 ms.

## Quick Start

```python
from pathlib import Path
from anvil.runtime.team import PersistentTeamRuntime, PersistentTeammateSpec
from anvil.core.types import StopConfig
from anvil.services.coding_runtime import build_coding_decider

runtime = PersistentTeamRuntime(root_dir=Path('.anvil/team'))

spec = PersistentTeammateSpec(
    name='worker-1',
    role='coder',
    workspace_root=Path('.'),
    decider=build_coding_decider(args),
    stop=StopConfig(max_steps=10, max_elapsed_s=120.0),
)
runtime.spawn_teammate(spec)

runtime.send_message('worker-1', 'Refactor anvil/utils.py to add type hints')
```

## Message Types

`TeamMessageType` (string enum) controls how messages are routed:

| Type | Description |
|---|---|
| `message` | Direct message to one teammate |
| `broadcast` | Sent to all members except the sender |
| `shutdown_request` | Ask a teammate to stop its loop |
| `shutdown_response` | Acknowledgement from the shutting-down teammate |
| `plan_approval_request` | Ask a teammate to review a plan |
| `plan_approval_response` | Approval (`approved: true`) or rejection with feedback |

## Sending Messages

```python
# Unicast
runtime.send_message('worker-1', 'task description', sender='lead')

# Broadcast to all members
runtime.broadcast('sprint started — pick up tasks', sender='lead')

# Plan review workflow
runtime.request_plan_approval('reviewer', plan_text, sender='worker-1')
runtime.approve_plan('worker-1', request_id, sender='reviewer')
runtime.reject_plan('worker-1', request_id, sender='reviewer', feedback='needs tests')
```

## Task Graph

Assign structured tasks with explicit dependencies:

```python
from anvil.runtime.task_graph import Task

runtime.replace_task_graph([
    Task(id='t1', title='Write tests', goal='Write unit tests for anvil/utils.py'),
    Task(id='t2', title='Fix bug', goal='Fix the off-by-one in token_estimation.py',
         depends_on=['t1']),
])

# Dispatch all tasks whose dependencies are satisfied
dispatched = runtime.dispatch_ready_tasks(sender='scheduler')
```

`dispatch_ready_tasks` picks idle teammates, assigns tasks round-robin (respecting
`assignee` and `metadata.role` hints), and sends the task goal as a message.

### Task Statuses

`pending` → `ready` → `running` → `completed` / `failed`

## Lifecycle

```python
# Check progress
print(runtime.teammate_status('worker-1'))    # idle | working | shutdown
print(runtime.has_active_tasks())
print(runtime.all_teammates_shutdown())

# Graceful shutdown
runtime.shutdown_teammate('worker-1')
runtime.shutdown_all(timeout_s=10.0)
```

## Safety: Ping-Pong Protection

Each teammate tracks consecutive messages from the same sender.  If a sender
exceeds `PersistentTeammateSpec.max_consecutive_same_sender` (default 5), the
teammate replies with a PING-PONG LIMIT notice and resets the counter, preventing
infinite agent-to-agent loops.

## Data Structures

### `TeamMessage`
Immutable `dataclass(frozen=True)`.  Fields: `id`, `sender`, `recipient`,
`message_type`, `body`, `created_at`, `metadata`.  Serialized to JSONL.

### `TeamMember`
Immutable registry entry.  Fields: `name`, `role`, `status`, `metadata`.
Persisted in `config.json` under the team root directory.

### `PersistentTeammateSpec`
Configuration for one teammate thread.  Not persisted — passed at `spawn_teammate`
time.

| Field | Default | Description |
|---|---|---|
| `name` | — | Unique identifier |
| `role` | — | Free-form role label, used for task routing |
| `workspace_root` | — | `Path` — all file tools are scoped here |
| `decider` | — | `DeciderFn` — LLM call factory |
| `stop` | `max_steps=6, max_elapsed_s=60` | Per-task stopping criteria |
| `policy` | `allow_all` | Tool permission policy |
| `skills` | `()` | Skill names to load for this teammate |
| `max_consecutive_same_sender` | `5` | Ping-pong guard threshold |
