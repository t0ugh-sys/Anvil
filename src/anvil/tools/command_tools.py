"""Command execution tools: run_command, run_command_async."""
from __future__ import annotations

import subprocess
from typing import Dict, List

from ..agent_protocol import ToolResult
from ..policies import Capability
from ..tool_spec import ToolRisk, ToolSpec
from .base import ToolContext


def run_command_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Run a command in the workspace using shell=False mode for security.

    Args:
        cmd: List of command arguments (e.g., ['ls', '-la'])
        id: Optional tool call ID

    Note: Shell features like pipes and wildcards are not supported.
          Use 'bash -c "cmd1 | cmd2"' if shell features are needed.
    """
    cmd_list = args.get('cmd')
    call_id = str(args.get('id', 'run_command'))

    if not isinstance(cmd_list, list):
        return ToolResult(id=call_id, ok=False, output='', error='cmd list is required')

    try:
        proc = subprocess.run(
            [str(item) for item in cmd_list],
            cwd=str(context.workspace_root),
            shell=False,
            check=False,
            text=True,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
        )
        merged = (proc.stdout or '') + (proc.stderr or '')
        ok = proc.returncode == 0
        return ToolResult(id=call_id, ok=ok, output=merged.strip(), error=None if ok else f'exit={proc.returncode}')
    except FileNotFoundError as exc:
        return ToolResult(id=call_id, ok=False, output='', error=f'command not found: {exc.filename}')
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def run_command_async_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Start one background command inside the workspace."""
    call_id = str(args.get('id', 'run_command_async'))
    runner = context.background_runner
    if runner is None:
        return ToolResult(id=call_id, ok=False, output='', error='background runner is not configured')

    cmd_list = args.get('cmd')
    if not isinstance(cmd_list, list):
        return ToolResult(id=call_id, ok=False, output='', error='cmd list is required')
    return runner.spawn(command=[str(item) for item in cmd_list], call_id=call_id)


def command_tool_specs() -> List[ToolSpec]:
    """Return specs for command tools."""
    return [
        ToolSpec(
            name='run_command',
            description='Run one command with shell disabled inside the workspace.',
            capabilities=(Capability.execute,),
            risk_level=ToolRisk.high,
            requires_workspace=True,
            input_notes='Use cmd as a list of arguments; shell syntax is not supported.',
        ),
        ToolSpec(
            name='run_command_async',
            description='Start one background command inside the workspace.',
            capabilities=(Capability.execute,),
            risk_level=ToolRisk.high,
            requires_workspace=True,
            input_notes='Use cmd as a list of arguments; command runs in background.',
        ),
    ]
