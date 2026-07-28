from __future__ import annotations

import json
from typing import List


def _extract_text_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ''.join(_extract_text_value(item) for item in value).strip()
    if not isinstance(value, dict):
        return ''

    for key in (
        'text',
        'output_text',
        'value',
        'answer',
        'final',
        'response',
        'output',
        'thinking',
        'reasoning',
        'reasoning_content',
        'message_content',
    ):
        text = value.get(key)
        if isinstance(text, str):
            return text
        nested_text = _extract_text_value(text)
        if nested_text:
            return nested_text

    for key in ('content', 'message', 'delta'):
        text = _extract_text_value(value.get(key))
        if text:
            return text

    choices = value.get('choices')
    if isinstance(choices, list):
        for choice in choices:
            text = _extract_text_value(choice)
            if text:
                return text

    return ''


def _extract_anthropic_tool_use_json(response: dict) -> str:
    content = response.get('content')
    if not isinstance(content, list):
        return ''

    tool_calls: list[dict[str, object]] = []
    thoughts: list[str] = []
    thinking_content: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get('type')
        if block_type == 'thinking':
            # Extended thinking block — capture reasoning
            thinking_text = block.get('thinking', '')
            if isinstance(thinking_text, str) and thinking_text.strip():
                thinking_content.append(thinking_text.strip())
            continue
        if block_type == 'text':
            text = block.get('text')
            if isinstance(text, str) and text.strip():
                thoughts.append(text.strip())
            continue
        if block_type != 'tool_use':
            continue
        name = block.get('name')
        call_id = block.get('id')
        arguments = block.get('input')
        if not isinstance(name, str) or not isinstance(call_id, str):
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append({'id': call_id, 'name': name, 'arguments': arguments})

    if not tool_calls:
        return ''
    # Combine thinking and regular thoughts
    all_thoughts = thinking_content + thoughts
    return json.dumps(
        {
            'thought': '\n'.join(all_thoughts),
            'plan': [],
            'tool_calls': tool_calls,
            'final': None,
        },
        ensure_ascii=False,
    )


def _extract_anthropic_text(response: dict) -> str:
    error = response.get('error')
    if isinstance(error, dict):
        message = error.get('message')
        if isinstance(message, str) and message.strip():
            raise ValueError(message)

    native_tool_json = _extract_anthropic_tool_use_json(response)
    if native_tool_json:
        return native_tool_json

    text = _extract_text_value(response).strip()
    if text:
        return text
    keys = ','.join(sorted(str(key) for key in response.keys()))
    raise ValueError(f'no text content; keys={keys or "(none)"}')


def _anthropic_file_tools() -> list[dict[str, object]]:
    """Anthropic native tool definitions with prompt caching support.

    Each tool definition includes cache_control for prompt caching.
    Tool definitions are static across requests, so they are ideal
    cache targets. First request creates the cache (10x cost),
    subsequent requests read from cache (0.1x cost) for 5 minutes.
    """
    tools = [
        {
            'name': 'read_file',
            'description': 'Read one UTF-8 file inside the workspace.',
            'input_schema': {
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
        },
        {
            'name': 'write_file',
            'description': 'Write one UTF-8 file inside the workspace, creating parent directories as needed.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'content': {'type': 'string'},
                },
                'required': ['path', 'content'],
            },
        },
        {
            'name': 'apply_patch',
            'description': 'Apply a unified patch to files inside the workspace.',
            'input_schema': {
                'type': 'object',
                'properties': {'patch': {'type': 'string'}},
                'required': ['patch'],
            },
        },
        {
            'name': 'search',
            'description': 'Search text in files inside the workspace.',
            'input_schema': {
                'type': 'object',
                'properties': {'pattern': {'type': 'string'}},
                'required': ['pattern'],
            },
        },
        {
            'name': 'run_command',
            'description': 'Run a command in the workspace.',
            'input_schema': {
                'type': 'object',
                'properties': {'cmd': {'type': 'array', 'items': {'type': 'string'}}},
                'required': ['cmd'],
            },
        },
    ]
    # Add cache_control to the last tool definition.
    # Claude caches everything up to and including the marked block.
    # Since tool definitions are static across requests, this is an ideal
    # cache target: first call creates cache (1.25x cost), subsequent
    # calls read from cache (0.1x cost) for 5 minutes.
    if tools:
        tools[-1]['cache_control'] = {'type': 'ephemeral'}
    return tools


def _prompt_goal(prompt: str) -> str:
    marker = '\nGoal:\n'
    if marker not in prompt:
        return prompt
    after_goal = prompt.split(marker, 1)[1]
    return after_goal.split('\nHistory:\n', 1)[0]


def _prompt_section(prompt: str, start: str, end: str) -> str:
    if start in prompt:
        value = prompt.split(start, 1)[1]
    elif start.startswith('\n') and prompt.startswith(start[1:]):
        value = prompt[len(start) - 1 :]
    else:
        return ''
    if end in value:
        value = value.split(end, 1)[0]
    return value.strip()


def _native_tool_prompt(prompt: str) -> str:
    goal = _prompt_goal(prompt).strip()
    state_summary = _prompt_section(prompt, '\nStateSummary:\n', '\nLastSteps:\n')
    last_steps = _prompt_section(prompt, '\nLastSteps:\n', '\nToolResults:\n')
    parts = [
        'You are a coding agent. Use the provided tools to perform the user request.',
        'Do not answer with JSON tool_calls text. Call the tool through the API.',
        'For directory plus file creation, call write_file on the target file path; it creates parent directories.',
        'If the user asks for the current path, use StateSummary.workspace.root.',
        '',
        'Goal:',
        goal,
    ]
    if state_summary:
        parts.extend(['', 'StateSummary:', state_summary])
    if last_steps and last_steps != '[]':
        parts.extend(['', 'LastSteps:', last_steps])
    return '\n'.join(parts)


def _prompt_has_successful_tool_result(prompt: str) -> bool:
    tool_results = _prompt_section(prompt, '\nToolResults:\n', '\nOnly output JSON.')
    return (
        "'ok': True" in tool_results
        or '"ok": true' in tool_results
        or '"ok": True' in tool_results
    )


def _prompt_requires_file_tool(prompt: str) -> bool:
    goal = _prompt_goal(prompt).lower()
    action_tokens = (
        '新增',
        '创建',
        '新建',
        '写入',
        '写到',
        '修改',
        '删除',
        '查看',
        '检查',
        'create',
        'write',
        'edit',
        'delete',
        'inspect',
        'read',
    )
    target_tokens = (
        '文件',
        '文件夹',
        '目录',
        '.txt',
        '.md',
        '.json',
        'file',
        'folder',
        'directory',
    )
    return any(token in goal for token in action_tokens) and any(token in goal for token in target_tokens)


def _prompt_should_force_write_file(prompt: str) -> bool:
    goal = _prompt_goal(prompt).lower()
    write_tokens = (
        '新增',
        '创建',
        '新建',
        '写入',
        '写到',
        'create',
        'write',
    )
    file_tokens = ('文件', '.txt', '.md', '.json', 'file')
    return any(token in goal for token in write_tokens) and any(token in goal for token in file_tokens)


def _split_system_user(prompt: str) -> tuple[str, str]:
    """Split prompt into (system_prompt, user_prompt) for caching.

    When prompt caching is enabled, the system prompt is sent separately
    with cache_control so it's cached across requests (90% cost reduction
    on subsequent calls).

    Strategy:
    1. Look for section markers that separate static instructions from
       dynamic content (Goal, StateSummary, History, etc.)
    2. Everything before the first dynamic section = system prompt
    3. The system prompt must be substantial (≥200 chars) to be worth caching
    """
    # Dynamic section markers — everything after these is per-request
    dynamic_markers = (
        '\nGoal:\n', '\nGoal: ',
        '\nUser request:\n', '\nUser:\n',
        '\nStateSummary:\n',
        '\nTask:\n', '\nTask: ',
        '\nCurrent task:\n',
        '\nInstruction:\n',
    )

    # Find the earliest dynamic section
    earliest_idx = len(prompt)
    for marker in dynamic_markers:
        idx = prompt.find(marker)
        if 0 < idx < earliest_idx:
            earliest_idx = idx

    if earliest_idx < len(prompt):
        system_part = prompt[:earliest_idx].strip()
        user_part = prompt[earliest_idx:].strip()
        # Only split if system part is substantial enough for caching
        # (Claude API requires ≥1024 tokens to activate caching)
        if len(system_part) > 200:
            return system_part, user_part

    return '', prompt
