"""File operation tools: read, write, patch."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..agent_protocol import ToolResult
from ..policies import Capability
from ..tool_spec import ToolDef, ToolRisk, ToolSpec, ValidationResult
from .base import ToolContext, ToolFn, require_params, resolve_inside_workspace


def read_file_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Read one file from the workspace."""
    path = str(args.get('path', ''))
    call_id = str(args.get('id', 'read_file'))
    try:
        target = resolve_inside_workspace(context.workspace_root, path)
        content = target.read_text(encoding='utf-8')
        return ToolResult(id=call_id, ok=True, output=content)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def write_file_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Write one UTF-8 file inside the workspace."""
    path = str(args.get('path', ''))
    content = str(args.get('content', ''))
    call_id = str(args.get('id', 'write_file'))
    try:
        target = resolve_inside_workspace(context.workspace_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return ToolResult(id=call_id, ok=True, output='ok')
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _split_patch_sections(patch_text: str) -> List[List[str]]:
    lines = patch_text.replace('\r\n', '\n').split('\n')
    sections: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.startswith('*** ') and current:
            sections.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        sections.append(current)
    return sections


def _resolve_patch_target(context: ToolContext, header: str) -> Path:
    raw = header.split(':', 1)[1].strip()
    if not raw:
        raise ValueError('patch target path is empty')
    return resolve_inside_workspace(context.workspace_root, raw)


def _apply_update_hunks(origin: str, body_lines: List[str]) -> str:
    source_lines = origin.split('\n')
    cursor = 0
    index = 0

    while index < len(body_lines):
        line = body_lines[index]
        if not line.startswith('@@'):
            index += 1
            continue
        index += 1
        old_chunk: List[str] = []
        new_chunk: List[str] = []
        while index < len(body_lines):
            current = body_lines[index]
            if current.startswith('@@'):
                break
            if not current:
                old_chunk.append('')
                new_chunk.append('')
                index += 1
                continue
            marker = current[:1]
            value = current[1:]
            if marker == ' ':
                old_chunk.append(value)
                new_chunk.append(value)
            elif marker == '-':
                old_chunk.append(value)
            elif marker == '+':
                new_chunk.append(value)
            else:
                raise ValueError(f'unsupported patch marker: {marker}')
            index += 1

        if old_chunk:
            found = -1
            max_start = len(source_lines) - len(old_chunk)
            for start in range(cursor, max_start + 1):
                if source_lines[start : start + len(old_chunk)] == old_chunk:
                    found = start
                    break
            if found < 0:
                raise ValueError('patch hunk does not match file content')
            source_lines = source_lines[:found] + new_chunk + source_lines[found + len(old_chunk) :]
            cursor = found + len(new_chunk)
        else:
            source_lines = source_lines[:cursor] + new_chunk + source_lines[cursor:]
            cursor = cursor + len(new_chunk)

    return '\n'.join(source_lines)


def apply_patch_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Apply a structured patch to workspace files."""
    patch_text = str(args.get('patch', ''))
    call_id = str(args.get('id', 'apply_patch'))
    if not patch_text.strip():
        return ToolResult(id=call_id, ok=False, output='', error='patch is required')

    try:
        root = context.workspace_root.resolve()
        normalized = patch_text.replace('\r\n', '\n').strip('\n')
        if not normalized.startswith('*** Begin Patch') or not normalized.endswith('*** End Patch'):
            raise ValueError('patch must start with "*** Begin Patch" and end with "*** End Patch"')
        content = normalized[len('*** Begin Patch') : -len('*** End Patch')].strip('\n')
        if not content:
            raise ValueError('patch body is empty')

        sections = _split_patch_sections(content)
        changed: List[str] = []
        for section in sections:
            header = section[0]
            body = section[1:]
            if header.startswith('*** Add File:'):
                target = _resolve_patch_target(context, header)
                if target.exists():
                    raise ValueError(f'file already exists: {target}')
                add_lines = [line[1:] for line in body if line.startswith('+')]
                if len(add_lines) != len(body):
                    raise ValueError('add file section only supports + lines')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('\n'.join(add_lines), encoding='utf-8')
                changed.append(target.relative_to(root).as_posix())
                continue

            if header.startswith('*** Delete File:'):
                target = _resolve_patch_target(context, header)
                if target.exists():
                    target.unlink()
                changed.append(target.relative_to(root).as_posix())
                continue

            if header.startswith('*** Update File:'):
                target = _resolve_patch_target(context, header)
                if not target.exists():
                    raise ValueError(f'file not found: {target}')
                original = target.read_text(encoding='utf-8')
                updated = _apply_update_hunks(original, body)
                target.write_text(updated, encoding='utf-8')
                changed.append(target.relative_to(root).as_posix())
                continue

            raise ValueError(f'unsupported patch header: {header}')

        return ToolResult(id=call_id, ok=True, output='\n'.join(changed))
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def file_tool_specs() -> List[ToolSpec]:
    """Return specs for file operation tools."""
    return [
        ToolSpec(
            name='read_file',
            description='Read one file from the workspace.',
            capabilities=(Capability.read,),
            risk_level=ToolRisk.low,
            requires_workspace=True,
            input_notes='',
        ),
        ToolSpec(
            name='write_file',
            description='Write one UTF-8 file inside the workspace.',
            capabilities=(Capability.write,),
            risk_level=ToolRisk.high,
            requires_workspace=True,
            input_notes='',
        ),
        ToolSpec(
            name='apply_patch',
            description='Apply a structured patch to workspace files.',
            capabilities=(Capability.write,),
            risk_level=ToolRisk.high,
            requires_workspace=True,
            input_notes='Provide a full structured patch with Begin/End Patch markers.',
        ),
    ]
