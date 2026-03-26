from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .compression import CompressionConfig
from .core.agent import LoopAgent
from .core.types import ContextProviderFn, ObserverFn, RunResult, StopConfig
from .policies import ToolPolicy
from .task_store import TaskStore
from .tool_use_loop import DeciderFn, SummarizerFn, ToolUseState, make_tool_use_step

try:
    from .skills import SkillLoader
except ImportError:  # pragma: no cover
    SkillLoader = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CodingAgentState(ToolUseState):
    pass


def build_coding_step(
    decider: DeciderFn,
    workspace_root: Path,
    skills: Optional[SkillLoader] = None,
    policy: ToolPolicy = ToolPolicy.allow_all(),
    task_store: TaskStore | None = None,
    compression_config: CompressionConfig | None = None,
    transcripts_dir: Path | None = None,
    summarizer: SummarizerFn | None = None,
):
    return make_tool_use_step(
        decider=decider,
        workspace_root=workspace_root,
        skills=skills,
        policy=policy,
        task_store=task_store,
        compression_config=compression_config,
        transcripts_dir=transcripts_dir,
        summarizer=summarizer,
    )


def run_coding_agent(
    *,
    goal: str,
    decider: DeciderFn,
    workspace_root: Path,
    stop: Optional[StopConfig] = None,
    observer: Optional[ObserverFn] = None,
    context_provider: Optional[ContextProviderFn] = None,
    skills: Optional[SkillLoader] = None,
    policy: ToolPolicy = ToolPolicy.allow_all(),
    task_store: TaskStore | None = None,
    compression_config: CompressionConfig | None = None,
    transcripts_dir: Path | None = None,
    summarizer: SummarizerFn | None = None,
) -> RunResult[CodingAgentState]:
    step = build_coding_step(
        decider,
        workspace_root=workspace_root,
        skills=skills,
        policy=policy,
        task_store=task_store,
        compression_config=compression_config,
        transcripts_dir=transcripts_dir,
        summarizer=summarizer,
    )
    agent = LoopAgent(step=step, stop=stop or StopConfig(max_steps=20, max_elapsed_s=60.0))
    return agent.run(
        goal=goal,
        initial_state=CodingAgentState(),
        observer=observer,
        context_provider=context_provider,
    )
