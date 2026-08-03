from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from .compression import CompactConfig
from .core.agent import AnvilAgent
from .core.types import ContextProviderFn, ObserverFn, RunResult, StopConfig
from .infra.policies import ToolPolicy
from .runtime.task_store import TaskStore
from .agent.loop import DeciderFn, SummarizerFn, ToolUseState, make_tool_use_step, make_tool_use_step_async

try:
    from .infra.skills import SkillLoader
except ImportError:  # pragma: no cover
    SkillLoader = None  # type: ignore[assignment]

__all__ = [
    'run_coding_agent',
    'run_coding_agent_async',
    'build_coding_step',
    'CodingAgentState',
    'DeciderFn',
    'SummarizerFn',
    'ToolUseState',
]

# Backward-compatible alias
CodingAgentState = ToolUseState


def build_coding_step(
    decider: DeciderFn,
    workspace_root: Path,
    skills: Optional[SkillLoader] = None,
    policy: ToolPolicy = ToolPolicy.allow_all(),
    task_store: TaskStore | None = None,
    compression_config: CompactConfig | None = None,
    transcripts_dir: Path | None = None,
    summarizer: SummarizerFn | None = None,
    on_tool_result=None,
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
        on_tool_result=on_tool_result,
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
    compression_config: CompactConfig | None = None,
    transcripts_dir: Path | None = None,
    summarizer: SummarizerFn | None = None,
    on_tool_result=None,
) -> RunResult[ToolUseState]:
    step = build_coding_step(
        decider,
        workspace_root=workspace_root,
        skills=skills,
        policy=policy,
        task_store=task_store,
        compression_config=compression_config,
        transcripts_dir=transcripts_dir,
        summarizer=summarizer,
        on_tool_result=on_tool_result,
    )
    agent = AnvilAgent(step=step, stop=stop or StopConfig(max_steps=20, max_elapsed_s=60.0))
    return agent.run(
        goal=goal,
        initial_state=ToolUseState(),
        observer=observer,
        context_provider=context_provider,
    )


async def run_coding_agent_async(
    *,
    goal: str,
    decider: DeciderFn,
    workspace_root: Path,
    stop: Optional[StopConfig] = None,
    observer: Optional[ObserverFn] = None,
    context_provider: Optional[ContextProviderFn] = None,
    skills=None,
    policy: ToolPolicy = ToolPolicy.allow_all(),
    task_store: TaskStore | None = None,
    compression_config: CompactConfig | None = None,
    transcripts_dir: Path | None = None,
    summarizer: SummarizerFn | None = None,
    on_tool_result=None,
) -> RunResult[ToolUseState]:
    step = make_tool_use_step_async(
        decider=decider,
        workspace_root=workspace_root,
        skills=skills,
        policy=policy,
        task_store=task_store,
        compression_config=compression_config,
        transcripts_dir=transcripts_dir,
        summarizer=summarizer,
        on_tool_result=on_tool_result,
    )
    agent = AnvilAgent(step=step, stop=stop or StopConfig(max_steps=20, max_elapsed_s=60.0))
    return await agent.run_async(
        goal=goal,
        initial_state=ToolUseState(),
        observer=observer,
        context_provider=context_provider,
    )
