"""
CompactManager - Multi-layer Context Compression

参考 Claude Code services/compact/ 的设计，实现多层压缩策略。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# ============== Compression Types ==============


class CompactStrategy(Enum):
    """压缩策略"""
    NONE = auto()           # 无压缩
    MICRO = auto()          # 微压缩 - 工具结果截断
    PARTIAL = auto()        # 部分压缩 - 按轮次分组
    FULL = auto()          # 完全压缩 - LLM 摘要


class CompactReason(Enum):
    """压缩原因"""
    MANUAL = auto()         # 手动触发
    TOKEN_LIMIT = auto()    # 达到 token 上限
    ROUND_LIMIT = auto()    # 达到轮次上限
    PROMPT_TOO_LONG = auto()  # API 返回 prompt too long


@dataclass
class CompactConfig:
    """压缩配置"""
    # Token 限制
    max_context_tokens: int = 50000
    warn_tokens_percent: float = 0.8  # 80% 时警告
    
    # Micro 压缩配置
    micro_keep_last_results: int = 3
    micro_max_result_chars: int = 500
    
    # Partial 压缩配置
    partial_max_rounds: int = 10
    partial_keep_recent_rounds: int = 3
    
    # Full 压缩配置
    full_summary_prompt: str = (
        "Summarize this conversation concisely, focusing on:\n"
        "1. What was accomplished\n"
        "2. Current state/todo\n"
        "3. Key decisions made"
    )
    
    # 压缩后恢复
    max_restore_files: int = 5
    max_tokens_per_file: int = 5000
    
    # Recent transcript entries (legacy)
    recent_transcript_entries: int = 10
    
    def validate(self) -> None:
        """验证配置有效性"""
        if self.max_context_tokens <= 0:
            raise ValueError('max_context_tokens must be positive')
        if self.warn_tokens_percent <= 0 or self.warn_tokens_percent > 1:
            raise ValueError('warn_tokens_percent must be in (0, 1]')
        if self.micro_keep_last_results < 0:
            raise ValueError('micro_keep_last_results must be non-negative')
        if self.partial_max_rounds <= 0:
            raise ValueError('partial_max_rounds must be positive')
        if self.recent_transcript_entries < 0:
            raise ValueError('recent_transcript_entries must be non-negative')


# Backward compatibility aliases (must be after definitions)
CompressionConfig = CompactConfig

# Legacy alias - create minimal TranscriptEntry class
@dataclass(frozen=True)
class TranscriptEntry:
    kind: str
    content: str
    tool_name: str | None = None
    call_id: str | None = None
    ok: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            'kind': self.kind,
            'content': self.content,
            'tool_name': self.tool_name,
            'call_id': self.call_id,
            'ok': self.ok,
        }

    def render_line(self) -> str:
        if self.kind == 'tool_result':
            status = 'ok' if self.ok else 'error'
            return f'tool_result[{self.tool_name or "unknown"}:{self.call_id or "-"}:{status}] {self.content}'
        return f'{self.kind}: {self.content}'


@dataclass
class CompactState:
    """压缩状态"""
    strategy: CompactStrategy = CompactStrategy.NONE
    reason: CompactReason = CompactReason.MANUAL
    
    compaction_count: int = 0
    total_tokens_saved: int = 0
    
    # 摘要信息
    summary: str = ''
    archived_count: int = 0
    
    # 时间戳
    last_compact_time: Optional[datetime] = None


@dataclass
class CompactResult:
    """压缩结果"""
    ok: bool
    strategy: CompactStrategy
    tokens_before: int
    tokens_after: int
    messages: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ''
    error: Optional[str] = None


@dataclass
class MessageGroup:
    """消息分组 - 按 API 轮次"""
    round_id: int
    messages: List[Dict[str, Any]]
    token_count: int = 0


# ============== Token Estimation ==============

def estimate_tokens(parts: Iterable[str]) -> int:
    """估算 token 数量（原始实现）"""
    total_chars = sum(len(part) for part in parts if part)
    return max(1, total_chars // 4) if total_chars else 0


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """估算消息列表的 token 数量"""
    total = 0
    for msg in messages:
        # 计算角色
        total += estimate_tokens([msg.get('role', '')])
        
        # 计算内容
        content = msg.get('content', '')
        if isinstance(content, str):
            total += estimate_tokens([content])
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += estimate_tokens([str(block.get('text', ''))])
    
    # 加上工具开销（估计）
    total += len(messages) * 10
    return total


# ============== Micro Compression ==============

def micro_compact_messages(
    messages: List[Dict[str, Any]],
    *,
    keep_last_results: int = 3,
    max_result_chars: int = 500,
) -> List[Dict[str, Any]]:
    """
    微压缩 - 保留最后 N 个工具结果，截断早期的结果内容。
    
    参考 Claude Code microCompact.ts
    """
    if not messages:
        return messages
    
    result: List[Dict[str, Any]] = []
    tool_result_indices: List[int] = []
    
    # 找到所有 tool_result 消息
    for i, msg in enumerate(messages):
        if msg.get('role') == 'assistant':
            content = msg.get('content', [])
            if isinstance(content, list):
                for j, block in enumerate(content):
                    if isinstance(block, dict) and block.get('type') == 'tool_result':
                        tool_result_indices.append((i, j))
    
    # 保留最后 N 个完整结果
    keep_count = min(keep_last_results, len(tool_result_indices))
    keep_indices = set(tool_result_indices[-keep_count:]) if keep_count > 0 else set()
    
    # 处理消息
    for i, msg in enumerate(messages):
        if msg.get('role') != 'assistant':
            result.append(msg)
            continue
        
        content = msg.get('content', [])
        if not isinstance(content, list):
            result.append(msg)
            continue
        
        new_content: List[Dict[str, Any]] = []
        for j, block in enumerate(content):
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            
            if block.get('type') != 'tool_result':
                new_content.append(block)
                continue
            
            # 检查是否保留
            if (i, j) in keep_indices:
                new_content.append(block)
            else:
                # 截断内容
                tool_name = block.get('tool_use_id', 'tool')
                truncated = {
                    **block,
                    'content': f'[Earlier {tool_name} result truncated]',
                }
                new_content.append(truncated)
        
        result.append({**msg, 'content': new_content})
    
    return result


# ============== Grouping ==============

def group_messages_by_rounds(
    messages: List[Dict[str, Any]],
) -> List[MessageGroup]:
    """
    按 API 轮次对消息进行分组。
    
    参考 Claude Code grouping.ts
    """
    groups: List[MessageGroup] = []
    current_group: List[Dict[str, Any]] = []
    current_round = 0
    
    for msg in messages:
        role = msg.get('role', '')
        
        if role == 'user':
            # 用户消息开始新轮次
            if current_group:
                groups.append(MessageGroup(
                    round_id=current_round,
                    messages=current_group,
                    token_count=estimate_messages_tokens(current_group),
                ))
                current_round += 1
            current_group = [msg]
        
        elif role == 'assistant':
            # Assistant 消息可能包含 tool_use
            current_group.append(msg)
            
            # 检查是否有 tool_use 块
            content = msg.get('content', [])
            has_tool_use = False
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        has_tool_use = True
                        break
            
            # 如果有 tool_use，等待 tool_result 后才算一轮结束
            if has_tool_use:
                continue
        
        elif role == 'tool':
            # Tool 结果消息
            current_group.append(msg)
        
        else:
            current_group.append(msg)
    
    # 添加最后一组
    if current_group:
        groups.append(MessageGroup(
            round_id=current_round,
            messages=current_group,
            token_count=estimate_messages_tokens(current_group),
        ))
    
    return groups


# ============== Partial Compression ==============

def partial_compact_messages(
    messages: List[Dict[str, Any]],
    *,
    max_rounds: int = 10,
    keep_recent_rounds: int = 3,
    summary: str = '',
) -> List[Dict[str, Any]]:
    """
    部分压缩 - 归档较早的轮次，保留最近的。
    
    参考 Claude Code sessionMemoryCompact.ts
    """
    groups = group_messages_by_rounds(messages)
    
    if len(groups) <= max_rounds:
        return messages
    
    # 归档早期轮次
    archive_groups = groups[:-keep_recent_rounds]
    keep_groups = groups[-keep_recent_rounds:]
    
    # 生成归档摘要消息
    archive_summary = summary or _generate_archive_summary(archive_groups)
    
    result: List[Dict[str, Any]] = [
        {
            'role': 'system',
            'content': f'[Earlier conversation summarized]\n\n{archive_summary}',
        }
    ]
    
    # 添加保留的消息
    for group in keep_groups:
        result.extend(group.messages)
    
    return result


def _generate_archive_summary(groups: List[MessageGroup]) -> str:
    """生成归档摘要"""
    if not groups:
        return 'No previous conversation.'
    
    lines = [f'Earlier ({len(groups)} rounds):']
    for group in groups:
        # 提取关键信息
        tool_count = 0
        for msg in group.messages:
            content = msg.get('content', [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        tool_count += 1
        
        lines.append(f'- Round {group.round_id}: {tool_count} tool calls, {group.token_count} tokens')

    return '\n'.join(lines)


def micro_compact_entries(
    entries: Tuple[TranscriptEntry, ...],
    *,
    keep_last_results: int,
) -> Tuple[TranscriptEntry, ...]:
    """Compact older tool results while preserving the most recent entries verbatim."""
    if not entries:
        return entries

    tool_result_indices = [index for index, entry in enumerate(entries) if entry.kind == 'tool_result']
    keep_count = min(keep_last_results, len(tool_result_indices))
    keep_indices = set(tool_result_indices[-keep_count:]) if keep_count > 0 else set()

    compacted: List[TranscriptEntry] = []
    for index, entry in enumerate(entries):
        if entry.kind != 'tool_result' or index in keep_indices:
            compacted.append(entry)
            continue

        tool_name = entry.tool_name or 'tool'
        compacted.append(
            TranscriptEntry(
                kind='tool_result',
                content=f'[Previous: used {tool_name}]',
                tool_name=entry.tool_name,
                call_id=entry.call_id,
                ok=entry.ok,
            )
        )

    return tuple(compacted)


def summarize_entries_deterministically(
    *,
    goal: str,
    previous_summary: str,
    entries: Tuple[TranscriptEntry, ...],
) -> str:
    """Legacy summary function"""
    lines = [
        f'Goal: {goal}',
        f'Previous summary: {previous_summary or "none"}',
        'Recent transcript:',
    ]
    for entry in entries[-12:]:
        lines.append(f'- {entry.render_line()[:240]}')
    return '\n'.join(lines)


def archive_transcript(
    *,
    transcripts_dir: Path,
    compaction_index: int,
    reason: str,
    goal: str,
    previous_summary: str,
    entries: Tuple[TranscriptEntry, ...],
) -> Path:
    """Legacy archive function"""
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    path = transcripts_dir / f'compact_{compaction_index:04d}.json'
    payload = {
        'goal': goal,
        'reason': reason,
        'previous_summary': previous_summary,
        'entries': [entry.to_dict() for entry in entries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


# ============== Full Compression ==============

# 注意: Full 压缩需要调用 LLM 来生成摘要
# 这是一个占位符实现，实际需要集成到 LLM 调用中


def prepare_compact_prompt(
    messages: List[Dict[str, Any]],
    *,
    config: CompactConfig,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    准备压缩提示。
    
    返回 (system_prompt, messages) 供 LLM 生成摘要。
    """
    # 构建系统提示
    system_prompt = config.full_summary_prompt
    
    return system_prompt, messages


# ============== CompactManager ==============

class CompactManager:
    """
    多层压缩管理器。
    
    层级:
    1. Micro - 工具结果截断 (无损)
    2. Partial - 按轮次归档 (轻微损)
    3. Full - LLM 摘要 (有损)
    """
    
    def __init__(
        self,
        config: Optional[CompactConfig] = None,
        summary_provider: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
    ):
        self.config = config or CompactConfig()
        self.summary_provider = summary_provider  # LLM 摘要生成函数
        
        self._state = CompactState()
        self._requested = False
        self._request_reason = ''

    @property
    def requested(self) -> bool:
        """Compatibility shim for harness callers that inspect request state directly."""
        return self._requested

    @property
    def reason(self) -> str:
        return self._request_reason
    
    def request(self, reason: str = '') -> None:
        """请求压缩"""
        self._requested = True
        self._request_reason = reason.strip()
    
    def should_compact(self, messages: List[Dict[str, Any]]) -> bool:
        """检查是否需要压缩"""
        # 检查手动请求
        if self._requested:
            return True
        
        # 检查 token 限制
        tokens = estimate_messages_tokens(messages)
        if tokens >= self.config.max_context_tokens * self.config.warn_tokens_percent:
            return True
        
        # 检查轮次
        groups = group_messages_by_rounds(messages)
        if len(groups) >= self.config.partial_max_rounds:
            return True
        
        return False
    
    def execute_compact(
        self,
        messages: List[Dict[str, Any]],
    ) -> CompactResult:
        """执行压缩"""
        tokens_before = estimate_messages_tokens(messages)
        
        # 选择压缩策略
        strategy = self._choose_strategy(messages)
        
        try:
            if strategy == CompactStrategy.MICRO:
                compacted = self._execute_micro(messages)
            elif strategy == CompactStrategy.PARTIAL:
                compacted = self._execute_partial(messages)
            elif strategy == CompactStrategy.FULL:
                compacted = self._execute_full(messages)
            else:
                return CompactResult(
                    ok=False,
                    strategy=strategy,
                    tokens_before=tokens_before,
                    tokens_after=tokens_before,
                    error='No compression needed',
                )
            
            tokens_after = estimate_messages_tokens(compacted)
            
            # 更新状态
            self._state.compaction_count += 1
            self._state.total_tokens_saved += tokens_before - tokens_after
            self._state.last_compact_time = datetime.now()
            self._state.strategy = strategy
            
            if self._requested:
                self._state.reason = CompactReason.MANUAL
            else:
                self._state.reason = CompactReason.TOKEN_LIMIT
            
            self._requested = False
            self._request_reason = ''
            
            return CompactResult(
                ok=True,
                strategy=strategy,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                messages=compacted,
                summary=self._state.summary,
            )
            
        except Exception as e:
            return CompactResult(
                ok=False,
                strategy=strategy,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                error=str(e),
            )
    
    def _choose_strategy(self, messages: List[Dict[str, Any]]) -> CompactStrategy:
        """选择压缩策略"""
        tokens = estimate_messages_tokens(messages)
        groups = group_messages_by_rounds(messages)
        
        # 80% - Micro 压缩
        if tokens < self.config.max_context_tokens * 0.8:
            return CompactStrategy.MICRO
        
        # 80-95% - Partial 压缩
        if tokens < self.config.max_context_tokens * 0.95:
            return CompactStrategy.PARTIAL
        
        # 95%+ 或无 LLM - Full 压缩（需要 LLM）
        if self.summary_provider:
            return CompactStrategy.FULL
        
        return CompactStrategy.PARTIAL
    
    def _execute_micro(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行微压缩"""
        return micro_compact_messages(
            messages,
            keep_last_results=self.config.micro_keep_last_results,
            max_result_chars=self.config.micro_max_result_chars,
        )
    
    def _execute_partial(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行部分压缩"""
        return partial_compact_messages(
            messages,
            max_rounds=self.config.partial_max_rounds,
            keep_recent_rounds=self.config.partial_keep_recent_rounds,
            summary=self._state.summary,
        )
    
    def _execute_full(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行完全压缩（需要 LLM）"""
        if not self.summary_provider:
            return self._execute_partial(messages)
        
        # 调用 LLM 生成摘要
        system_prompt, compact_messages = prepare_compact_prompt(
            messages,
            config=self.config,
        )
        
        # 注意: 这里需要实际调用 LLM
        # 简化实现
        summary = self._state.summary or '[Summary needed]'
        
        return partial_compact_messages(
            messages,
            max_rounds=self.config.partial_max_rounds,
            keep_recent_rounds=self.config.partial_keep_recent_rounds,
            summary=summary,
        )
    
    @property
    def state(self) -> CompactState:
        """获取压缩状态"""
        return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'compaction_count': self._state.compaction_count,
            'total_tokens_saved': self._state.total_tokens_saved,
            'last_compact_time': self._state.last_compact_time.isoformat() if self._state.last_compact_time else None,
            'strategy': self._state.strategy.name,
            'reason': self._state.reason.name,
        }


# ============== Archive ==============

def archive_compacted_messages(
    messages: List[Dict[str, Any]],
    archive_dir: Path,
    compaction_index: int,
    goal: str,
    summary: str,
) -> Path:
    """归档压缩前的消息"""
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    path = archive_dir / f'compact_{compaction_index:04d}.json'
    payload = {
        'goal': goal,
        'summary': summary,
        'timestamp': datetime.now().isoformat(),
        'messages': messages,
    }
    
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    
    return path
