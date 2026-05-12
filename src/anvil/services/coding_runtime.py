from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Tuple

from ..agent_protocol import render_agent_step_schema
from ..coding_agent import run_coding_agent
from ..compression import summarize_entries_deterministically
from ..core.types import StopConfig
from ..llm.providers import build_invoke_from_args
from ..runtime import CodeRuntime
from ..skills import SkillLoader, list_skills
from ..utils import resolve_goal


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
    recent_history = list(history[-history_window:])
    format_repair = ''
    if any('invalid agent step json' in item for item in recent_history):
        format_repair = (
            'Your previous response did not match the required agent-step JSON schema. '
            'Retry the same user goal now. Return exactly one JSON object, with no markdown, '
            'no prose, and no text before or after the JSON. Use write_file for empty files; '
            'write_file creates parent directories when needed.\n'
        )
    tool_repair = ''
    if any('tool action required' in item for item in last_steps):
        tool_repair = (
            'Your previous response was rejected because tool_calls was empty for a file operation. '
            'Retry the same user goal now with at least one tool call that performs the requested work. '
            'For a directory plus file request, call write_file on the target file path; it creates parent directories. '
            'Do not set final until a tool result confirms the work succeeded.\n'
        )
    completion_repair = ''
    if any(getattr(item, 'ok', False) for item in tool_results):
        completion_repair = (
            'At least one previous tool call succeeded. If the user goal is now satisfied, do not call more tools; '
            'return final with a concise completion message.\n'
        )
    tool_guide = (
        'Available tool call names and common arguments:\n'
        '- read_file: {"path":"relative/path"}\n'
        '- write_file: {"path":"relative/path","content":"text"}; creates parent directories inside the workspace.\n'
        '- apply_patch: {"patch":"*** Begin Patch\\n...\\n*** End Patch"}\n'
        '- search: {"pattern":"literal text"}\n'
        '- run_command: {"cmd":["program","arg"]}\n'
        'For a requested empty JSON file, call write_file with a .json path and content "".\n'
        'Use paths relative to the workspace unless the user supplies an absolute path inside the workspace.\n'
        'If the user asks for the current path, use StateSummary.workspace.root.\n'
        'The schema describes field types, not values to copy. '
        'Never invent tool names. Use only the listed tool names, and choose tools that directly satisfy the user goal.\n'
    )
    return (
        'You are a coding agent. Return strict JSON matching schema.\n'
        + format_repair
        + tool_repair
        + completion_repair
        + 'Use tools when needed. Keep a visible todo list updated via the todo_write tool when progress changes.\n'
        + 'When the user asks you to create, edit, write, delete, move, or inspect files, do the work with tools. '
        + 'Do not answer with shell commands for the user to copy, and do not ask whether you should execute a clearly requested file operation. '
        + 'For creating an empty file, use write_file with an empty content string. '
        + 'For creating a directory and file together, write the target file path directly; write_file creates parent directories.\n'
        + tool_guide
        + ('Available skills:\n' + '\n'.join(skill_lines) + '\n' if skill_lines else '')
        + 'Do not inline full skill instructions in the prompt. Load them on demand with load_skill.\n'
        + render_agent_step_schema()
        + '\nGoal:\n'
        + goal
        + '\nHistory:\n'
        + str(recent_history)
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
    goal = resolve_goal(getattr(args, 'goal', None), getattr(args, 'goal_file', None))
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
