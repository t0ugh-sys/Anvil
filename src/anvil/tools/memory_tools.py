"""Memory and analysis tools: analyze_memory, compact, todo_write, load_skill."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ..agent_protocol import ToolResult
from ..policies import Capability
from ..tool_spec import ToolRisk, ToolSpec
from ..todo import render_todo_lines
from .base import ToolContext

__all__ = ['analyze_memory_tool', 'todo_write_tool', 'load_skill_tool', 'compact_tool', 'memory_tool_specs']


def analyze_memory_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Analyze past runs from memory store to learn patterns and insights.
    
    Args:
        memory_dir: Path to the memory store directory (default: .anvil/runs)
        goal_filter: Optional goal to filter runs by
        limit: Number of recent runs to analyze (default: 5)
    """
    call_id = str(args.get('id', 'analyze_memory'))
    memory_dir = str(args.get('memory_dir', '.anvil/runs'))
    goal_filter = str(args.get('goal_filter', '')).strip()
    limit = int(str(args.get('limit', '5')))
    
    try:
        memory_path = Path(memory_dir)
        if not memory_path.exists():
            return ToolResult(id=call_id, ok=False, output='', error=f'memory directory not found: {memory_dir}')
        
        # Find all run directories
        run_dirs = sorted([d for d in memory_path.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
        run_dirs = run_dirs[:limit]
        
        if not run_dirs:
            return ToolResult(id=call_id, ok=True, output='No past runs found in memory', error=None)
        
        analysis: List[str] = []
        total_runs = 0
        completed_runs = 0
        failed_runs = 0
        
        for run_dir in run_dirs:
            summary_file = run_dir / 'summary.json'
            if not summary_file.exists():
                continue
            
            try:
                with summary_file.open(encoding='utf-8') as f:
                    summary = json.load(f)
                
                goal = summary.get('goal', '')
                if goal_filter and goal_filter.lower() not in goal.lower():
                    continue
                
                total_runs += 1
                done = summary.get('done', False)
                stop_reason = summary.get('stop_reason', 'unknown')
                steps = summary.get('steps', 0)
                
                if done:
                    completed_runs += 1
                else:
                    failed_runs += 1
                
                analysis.append(f'Run: {run_dir.name}')
                analysis.append(f'  Goal: {goal[:80]}...' if len(goal) > 80 else f'  Goal: {goal}')
                analysis.append(f'  Result: {"✓ Completed" if done else "✗ Failed"} (stop: {stop_reason})')
                analysis.append(f'  Steps: {steps}')
                analysis.append('')
            except Exception:
                continue
        
        # Build summary
        summary_text = [
            f'=== Memory Analysis (Last {limit} runs) ===',
            f'Total runs analyzed: {total_runs}',
            f'Completed: {completed_runs}',
            f'Failed: {failed_runs}',
            f'Success rate: {completed_runs/total_runs*100:.1f}%' if total_runs > 0 else 'N/A',
            '',
            '--- Recent Runs ---',
            ''
        ]
        summary_text.extend(analysis)
        
        return ToolResult(id=call_id, ok=True, output='\n'.join(summary_text), error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def todo_write_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Update the visible todo list stored in runtime state."""
    call_id = str(args.get('id', 'todo_write'))
    manager = context.todo_manager
    if manager is None:
        return ToolResult(id=call_id, ok=False, output='', error='todo manager is not configured')

    items = args.get('items')
    if not isinstance(items, list):
        return ToolResult(id=call_id, ok=False, output='', error='items list is required')

    try:
        updated_items = manager.write(items)
        lines = [
            'todo updated',
            *[f'- {line}' for line in render_todo_lines(updated_items)],
        ]
        return ToolResult(id=call_id, ok=True, output='\n'.join(lines), error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def load_skill_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Load full skill instructions into the conversation on demand."""
    call_id = str(args.get('id', 'load_skill'))
    loader = context.skill_loader
    if loader is None:
        return ToolResult(id=call_id, ok=False, output='', error='skill loader is not configured')

    name = str(args.get('name', '')).strip()
    if not name:
        return ToolResult(id=call_id, ok=False, output='', error='skill name is required')

    body = loader.load_body(name)
    if body is None:
        return ToolResult(id=call_id, ok=False, output='', error=f'skill not loaded: {name}')
    return ToolResult(id=call_id, ok=True, output=f'<skill name="{name}">\n{body}\n</skill>')


def compact_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Request transcript compaction for long-running sessions."""
    call_id = str(args.get('id', 'compact'))
    manager = context.compact_manager
    if manager is None:
        return ToolResult(id=call_id, ok=False, output='', error='compact manager is not configured')

    reason = str(args.get('reason', '')).strip()
    manager.request(reason)
    message = 'compaction requested'
    if reason:
        message += f': {reason}'
    return ToolResult(id=call_id, ok=True, output=message, error=None)


def memory_tool_specs() -> List[ToolSpec]:
    """Return specs for memory/analysis tools."""
    return [
        ToolSpec(
            name='analyze_memory',
            description='Analyze prior run summaries under the memory directory.',
            capabilities=(Capability.memory,),
            risk_level=ToolRisk.medium,
            requires_workspace=True,
            input_notes='',
        ),
        ToolSpec(
            name='todo_write',
            description='Update the visible todo list stored in runtime state.',
            capabilities=(Capability.memory,),
            risk_level=ToolRisk.medium,
            requires_workspace=True,
            input_notes='Provide items as a list of todo objects.',
        ),
        ToolSpec(
            name='load_skill',
            description='Load full skill instructions into the conversation on demand.',
            capabilities=(Capability.read,),
            risk_level=ToolRisk.low,
            requires_workspace=True,
            input_notes='',
        ),
        ToolSpec(
            name='compact',
            description='Request transcript compaction for long-running sessions.',
            capabilities=(Capability.memory,),
            risk_level=ToolRisk.low,
            requires_workspace=True,
            input_notes='',
        ),
    ]
