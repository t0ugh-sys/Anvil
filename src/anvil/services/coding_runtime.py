from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..agent_protocol import render_agent_step_schema
from ..coding_agent import run_coding_agent
from ..compression import summarize_entries_deterministically
from ..core.types import StopConfig
from ..llm.providers import build_invoke_from_args
from ..runtime import CodeRuntime
from ..skills import SkillLoader, list_skills


def resolve_goal(args: argparse.Namespace) -> str:
    if args.goal_file:
        return Path(args.goal_file).read_text(encoding='utf-8-sig').strip()
    return str(getattr(args, 'goal', '') or '').strip()


def build_coding_prompt(
    *,
    goal: str,
    history: Tuple[str, ...],
    tool_results: Tuple[Any, ...],
    state_summary: Dict[str, object],
    last_steps: Tuple[str, ...],
    history_window: int,
    skills: SkillLoader | None = None,
) -> str:
    skill_lines: list[str] = []
    if skills is not None:
        for item in skills.metadata():
            skill_lines.append(f'- {item["name"]}: {item["description"]}')
    return (
        'You are a coding agent. Return strict JSON matching schema.\n'
        'Use tools when needed. Keep a visible todo list updated via the todo_write tool when progress changes.\n'
        + ('Available skills:\n' + '\n'.join(skill_lines) + '\n' if skill_lines else '')
        + 'Do not inline full skill instructions in the prompt. Load them on demand with load_skill.\n'
        + render_agent_step_schema()
        + '\nGoal:\n'
        + goal
        + '\nHistory:\n'
        + str(list(history[-history_window:]))
        + '\nStateSummary:\n'
        + json.dumps(state_summary, ensure_ascii=False)
        + '\nLastSteps:\n'
        + str(list(last_steps))
        + '\nToolResults:\n'
        + str(
            [
                {
                    'id': r.id,
                    'ok': r.ok,
                    'output': r.output[:500],
                    'error': r.error,
                    'permission_decision': getattr(r, 'metadata', {}).get('permission_decision'),
                }
                for r in tool_results
            ]
        )
        + '\nOnly output JSON.'
    )


def build_coding_decider(args: argparse.Namespace, skills: SkillLoader | None = None):
    invoke = build_invoke_from_args(args, mode='coding')

    def decider(
        goal: str,
        history: Tuple[str, ...],
        tool_results: Tuple[Any, ...],
        state_summary: Dict[str, object],
        last_steps: Tuple[str, ...],
    ) -> str:
        history_window = max(1, args.history_window)
        prompt = build_coding_prompt(
            goal=goal,
            history=history,
            tool_results=tool_results,
            state_summary=state_summary,
            last_steps=last_steps,
            history_window=history_window,
            skills=skills,
        )
        return invoke(prompt)

    return decider


def build_coding_summarizer(args: argparse.Namespace) -> Optional[Any]:
    from ..compression import TranscriptEntry

    if str(args.provider) == 'mock':
        return None

    invoke = build_invoke_from_args(args, mode='coding')

    def summarizer(goal: str, previous_summary: str, transcript: Tuple[TranscriptEntry, ...]) -> str:
        transcript_lines = [entry.render_line()[:400] for entry in transcript[-16:]]
        prompt = (
            'Summarize the coding-agent conversation for long-running context compression.\n'
            'Return plain text only.\n'
            'Keep: user goal, constraints, files changed, tool outcomes, unfinished work.\n'
            f'Goal:\n{goal}\n'
            f'Previous summary:\n{previous_summary or "none"}\n'
            'Recent transcript:\n'
            + '\n'.join(transcript_lines)
        )
        response = invoke(prompt).strip()
        if response:
            return response
        return summarize_entries_deterministically(goal=goal, previous_summary=previous_summary, entries=transcript)

    return summarizer


def load_skills_from_args(args: argparse.Namespace) -> SkillLoader | None:
    skills_arg = getattr(args, 'skills', None)
    if not skills_arg:
        return None

    loader = SkillLoader()
    for skill_name in skills_arg:
        if skill_name == 'all':
            for name in list_skills():
                loader.load(name)
        else:
            if not loader.load(skill_name):
                print(f"Warning: Unknown skill '{skill_name}' - skipping")
    return loader


def run_code_command(args: argparse.Namespace) -> int:
    goal = resolve_goal(args)
    runtime = CodeRuntime(args, goal=goal)
    if not runtime.goal.strip():
        raise ValueError('goal is required unless resuming from a session with a stored goal')
    skills = load_skills_from_args(args)
    decider = build_coding_decider(args, skills)
    summarizer = build_coding_summarizer(args)
    if runtime.observer is not None:
        runtime.observer('run_started', {'goal': runtime.goal, 'strategy': 'coding', 'facts': []})
    result = run_coding_agent(
        goal=runtime.goal,
        decider=decider,
        workspace_root=runtime.workspace_root,
        stop=StopConfig(max_steps=args.max_steps, max_elapsed_s=args.timeout_s),
        observer=runtime.observer,
        context_provider=runtime.build_context_provider(),
        skills=skills,
        policy=runtime.build_policy(),
        task_store=runtime.task_store,
        compression_config=runtime.compression_config,
        transcripts_dir=runtime.transcripts_dir,
        summarizer=summarizer,
    )
    payload = runtime.finalize(result)
    if args.output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"done: {result.done}")
        print(f"stop_reason: {result.stop_reason.value}")
        print(f"steps: {result.steps}")
        print(f"final_output: {result.final_output}")
        print(f"session_id: {payload['session_id']}")
        print(f"memory_run_dir: {payload['memory_run_dir']}")
        if 'run_dir' in payload:
            print(f"run_dir: {payload['run_dir']}")
    return 0 if result.done else 1
