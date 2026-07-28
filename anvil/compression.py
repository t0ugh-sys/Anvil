"""
CompactManager - Multi-layer Context Compression

参考 Claude Code services/compact/ 的设计，实现多层压缩策略。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .token_estimation import (
    estimate_tokens as _estimate_tokens,
    estimate_messages_tokens as _estimate_messages_tokens,
)

__all__ = [
    'CompactConfig',
    'CompactManager',
    'CompactState',
    'CompactResult',
    'CompactStrategy',
    'CompactReason',
    'estimate_tokens',
    'estimate_messages_tokens',
    'micro_compact_messages',
    'time_based_micro_compact',
    'partial_compact_messages',
    'TranscriptEntry',
    'MessageGroup',
    # Hierarchical Summary
    'SummaryLevel',
    'HierarchicalSummarizer',
    'hierarchical_compact_messages',
    # Prompt Caching
    'CacheSegment',
    'PromptCacheManager',
    'add_cache_control_hints',
]

# Compression thresholds (fraction of max_context_tokens)
PARTIAL_COMPACT_THRESHOLD = 0.8
FULL_COMPACT_THRESHOLD = 0.95
MESSAGE_OVERHEAD_TOKENS = 10


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
# DEPRECATED: use CompactConfig directly. Kept for external consumers.
CompressionConfig = CompactConfig

# Legacy alias - create minimal TranscriptEntry class
@dataclass(frozen=True)
class TranscriptEntry:
    kind: str
    content: str
    tool_name: str | None = None
    call_id: str | None = None
    ok: bool | None = None
    created_at: float = 0.0  # Unix timestamp for time-based compaction

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
    last_compact_time: datetime | None = None


@dataclass
class CompactResult:
    """压缩结果"""
    ok: bool
    strategy: CompactStrategy
    tokens_before: int
    tokens_after: int
    messages: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ''
    error: str | None = None


@dataclass
class MessageGroup:
    """消息分组 - 按 API 轮次"""
    round_id: int
    messages: List[Dict[str, Any]]
    token_count: int = 0


# ============== Token Estimation ==============
# Re-export from token_estimation to avoid duplication

estimate_tokens = _estimate_tokens
estimate_messages_tokens = _estimate_messages_tokens


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


# ============== Time-based Microcompact ==============
# Reference: Claude Code timeBasedMCConfig.ts
# When user returns after a long gap (30-60 min), the server-side prompt cache
# is almost certainly expired. Clear ALL old tool results aggressively.

DEFAULT_GAP_THRESHOLD_S = 30 * 60  # 30 minutes


def time_based_micro_compact(
    entries: Tuple[TranscriptEntry, ...],
    *,
    now_s: float | None = None,
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
) -> Tuple[TranscriptEntry, ...]:
    """Clear ALL old tool results if there's been a long inactivity gap.

    Unlike count-based micro_compact which keeps recent N results,
    this clears everything older than the gap threshold — because after
    30+ minutes of inactivity the prompt cache is expired anyway,
    so keeping old results provides no cache benefit and wastes context.
    """
    if not entries:
        return entries

    _now = now_s if now_s is not None else time.time()

    # Find the most recent timestamp
    latest_ts = max((e.created_at for e in entries if e.created_at > 0), default=0.0)
    if latest_ts <= 0:
        return entries  # No timestamps — skip

    # Check if the gap exceeds threshold
    gap = _now - latest_ts
    if gap < gap_threshold_s:
        return entries  # No significant gap

    # Clear ALL tool results (not just old ones) — cache is expired
    compacted: list[TranscriptEntry] = []
    for entry in entries:
        if entry.kind == 'tool_result':
            tool_name = entry.tool_name or 'tool'
            compacted.append(TranscriptEntry(
                kind='tool_result',
                content=f'[Previous: used {tool_name}] (cache expired after {int(gap)}s gap)',
                tool_name=entry.tool_name,
                call_id=entry.call_id,
                ok=entry.ok,
                created_at=entry.created_at,
            ))
        else:
            compacted.append(entry)
    return tuple(compacted)


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

    Cache-aware mode: When prompt_cache_manager is set, compression
    preserves the cacheable prefix to maximize prompt cache hits.
    """

    def __init__(
        self,
        config: Optional[CompactConfig] = None,
        summary_provider: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
        prompt_cache_manager: Optional['PromptCacheManager'] = None,
    ):
        self.config = config or CompactConfig()
        self.summary_provider = summary_provider  # LLM 摘要生成函数
        self.prompt_cache_manager = prompt_cache_manager

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
        if tokens < self.config.max_context_tokens * PARTIAL_COMPACT_THRESHOLD:
            return CompactStrategy.MICRO

        # 80-95% - Partial 压缩
        if tokens < self.config.max_context_tokens * FULL_COMPACT_THRESHOLD:
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
        """执行部分压缩（缓存感知）。

        When prompt_cache_manager is set, only compresses messages AFTER
        the cache boundary to preserve cache hits.
        """
        if self.prompt_cache_manager is not None:
            return self._cache_aware_partial(messages)
        return partial_compact_messages(
            messages,
            max_rounds=self.config.partial_max_rounds,
            keep_recent_rounds=self.config.partial_keep_recent_rounds,
            summary=self._state.summary,
        )

    def _cache_aware_partial(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cache-aware partial compaction.

        Uses PromptCacheManager to identify the stable cacheable prefix,
        then only compacts messages in the dynamic suffix. This ensures
        the API-level prompt cache stays valid across compaction.
        """
        assert self.prompt_cache_manager is not None
        prefix, suffix = self.prompt_cache_manager.split_for_caching(messages)

        if not suffix:
            # Everything is in the prefix — nothing to compact
            return messages

        # Compact only the suffix
        compacted_suffix = partial_compact_messages(
            suffix,
            max_rounds=self.config.partial_max_rounds,
            keep_recent_rounds=self.config.partial_keep_recent_rounds,
            summary=self._state.summary,
        )

        # Reassemble: preserved prefix + compacted suffix
        return prefix + compacted_suffix
    
    def _execute_full(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行完全压缩（需要 LLM）"""
        if not self.summary_provider:
            return self._execute_partial(messages)

        system_prompt, compact_messages = prepare_compact_prompt(
            messages,
            config=self.config,
        )

        summary = self.summary_provider(system_prompt, compact_messages)
        self._state.summary = summary

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


# ============== Importance Scoring ==============
# Based on Zero2Agent article #20: Context Compression & Optimization
# Hybrid scoring: score = w_recency * recency + w_relevance * relevance + w_importance * importance


# Keywords that signal important content (decisions, commitments, facts)
_DECISION_KEYWORDS = frozenset({
    'decided', 'decision', 'chose', 'chosen', 'agreed', 'approved',
    'confirmed', 'final', 'conclusion', 'resolved', 'must', 'shall',
    'requirement', 'constraint', 'blocked', 'critical', 'priority',
    '决定', '确认', '必须', '关键', '结论', '阻塞', '优先',
})

_QUESTION_KEYWORDS = frozenset({
    '?', 'why', 'how', 'what', 'when', 'where', 'which',
    'should', 'could', 'would',
    '为什么', '怎么', '什么', '何时', '哪个', '是否',
})

_ERROR_KEYWORDS = frozenset({
    'error', 'failed', 'failure', 'exception', 'traceback',
    'bug', 'broken', 'crash', 'regression',
    '错误', '失败', '异常', '崩溃',
})


def score_message_importance(
    message: Dict[str, Any],
    *,
    position: int = 0,
    total_messages: int = 1,
    recency_weight: float = 0.4,
    relevance_weight: float = 0.3,
    importance_weight: float = 0.3,
) -> float:
    """Score a message's importance for context filtering.

    Inspired by Zero2Agent hybrid scoring formula:
        score = w_recency * recency + w_relevance * relevance + w_importance * importance

    Args:
        message: The message dict to score.
        position: Index of this message in the list (0 = oldest).
        total_messages: Total number of messages.
        recency_weight: Weight for recency score.
        relevance_weight: Weight for relevance score.
        importance_weight: Weight for importance score.

    Returns:
        Score between 0.0 and 1.0.
    """
    # --- Recency score (newer = higher) ---
    if total_messages > 1:
        recency = position / (total_messages - 1)
    else:
        recency = 1.0

    # --- Content importance score ---
    content = _extract_text_content(message)
    role = message.get('role', '')
    content_lower = content.lower()

    importance = 0.0

    # System messages are always important
    if role == 'system':
        importance += 0.5

    # User messages slightly more important than assistant
    if role == 'user':
        importance += 0.15
    elif role == 'assistant':
        importance += 0.1

    # Decision keywords boost
    decision_hits = sum(1 for kw in _DECISION_KEYWORDS if kw in content_lower)
    importance += min(0.3, decision_hits * 0.1)

    # Error messages are important for debugging
    error_hits = sum(1 for kw in _ERROR_KEYWORDS if kw in content_lower)
    importance += min(0.2, error_hits * 0.1)

    # Questions are moderately important
    question_hits = sum(1 for kw in _QUESTION_KEYWORDS if kw in content_lower)
    importance += min(0.1, question_hits * 0.05)

    # Tool results with errors are more important than successful ones
    if role == 'tool':
        is_error = message.get('is_error', False)
        if is_error:
            importance += 0.2

    # Longer content tends to be more information-dense
    if len(content) > 500:
        importance += 0.1

    importance = min(1.0, importance)

    # --- Relevance score (heuristic: tool calls near the end are relevant) ---
    # Without embeddings, use a simple heuristic: messages with tool_use
    # blocks are operationally relevant
    relevance = 0.0
    if role == 'assistant':
        tool_calls = message.get('tool_calls', [])
        if tool_calls:
            relevance += 0.3
    if role == 'tool':
        relevance += 0.2  # tool results are contextually relevant

    # Combine scores
    score = (
        recency_weight * recency
        + relevance_weight * relevance
        + importance_weight * importance
    )

    return min(1.0, score)


def _extract_text_content(message: Dict[str, Any]) -> str:
    """Extract text content from a message (handles string and block formats)."""
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    parts.append(block.get('text', ''))
                elif block.get('type') == 'tool_result':
                    parts.append(str(block.get('content', '')))
        return ' '.join(parts)
    return str(content) if content else ''


def filter_messages_by_importance(
    messages: List[Dict[str, Any]],
    *,
    min_score: float = 0.2,
    always_keep_recent: int = 6,
    always_keep_system: bool = True,
) -> List[Dict[str, Any]]:
    """Filter messages by importance score, keeping the most relevant ones.

    Always preserves:
    - System messages (if always_keep_system=True)
    - The most recent N messages (always_keep_recent)

    Args:
        messages: List of message dicts.
        min_score: Minimum importance score to keep (0.0-1.0).
        always_keep_recent: Number of recent messages to always keep.
        always_keep_system: Whether to always keep system messages.

    Returns:
        Filtered list of messages, preserving order.
    """
    if not messages:
        return messages

    total = len(messages)

    # Score all messages
    scored = []
    for i, msg in enumerate(messages):
        score = score_message_importance(msg, position=i, total_messages=total)
        scored.append((i, msg, score))

    # Always keep system messages
    keep_indices: set = set()
    for i, msg, score in scored:
        if always_keep_system and msg.get('role') == 'system':
            keep_indices.add(i)

    # Always keep recent messages
    for i in range(max(0, total - always_keep_recent), total):
        keep_indices.add(i)

    # Keep messages above threshold
    for i, msg, score in scored:
        if score >= min_score:
            keep_indices.add(i)

    # Reconstruct in order
    return [msg for i, msg, _ in scored if i in keep_indices]


# ============== Hierarchical Summary ==============
# Based on Zero2Agent article #20: Multi-level context summarization
# L1: Per-round summaries (tool calls & outcomes)
# L2: Block summaries (group of rounds on related topics)
# L3: Global summary (entire conversation arc)


@dataclass
class SummaryLevel:
    """A single level in the hierarchical summary."""
    level: int  # 1, 2, or 3
    content: str
    round_range: Tuple[int, int]  # (start_round, end_round)
    token_count: int = 0


class HierarchicalSummarizer:
    """Multi-level hierarchical summarization for long conversations.

    Produces three levels of summaries:
    - L1 (Per-round): One-line summary per tool-use round
    - L2 (Block): Groups of 5-10 L1 summaries into thematic blocks
    - L3 (Global): Single paragraph covering the entire conversation

    The LLM can then use the appropriate level depending on how much
    context it needs — L3 for overview, L2 for recent topic context,
    L1 for detailed per-round recall.
    """

    def __init__(
        self,
        summary_provider: Optional[Callable[[str, str], str]] = None,
        l1_per_round: bool = True,
        l2_block_size: int = 5,
    ):
        """
        Args:
            summary_provider: Callable(system_prompt, text) -> summary.
                If None, uses deterministic (non-LLM) extraction.
            l1_per_round: Whether to generate L1 per-round summaries.
            l2_block_size: Number of L1 summaries per L2 block.
        """
        self.summary_provider = summary_provider
        self.l1_per_round = l1_per_round
        self.l2_block_size = l2_block_size

    def summarize(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_l1_tokens: int = 2000,
        max_l2_tokens: int = 500,
        max_l3_tokens: int = 300,
    ) -> List[SummaryLevel]:
        """Generate hierarchical summaries for the given messages.

        Returns a list of SummaryLevel objects, ordered L1 -> L2 -> L3.
        If LLM provider is not available, uses deterministic extraction.
        """
        groups = group_messages_by_rounds(messages)
        if not groups:
            return []

        levels: List[SummaryLevel] = []

        # --- L1: Per-round summaries ---
        l1_summaries: List[Tuple[int, str]] = []
        for group in groups:
            summary = self._summarize_round(group)
            l1_summaries.append((group.round_id, summary))

        if l1_summaries:
            first_round = l1_summaries[0][0]
            last_round = l1_summaries[-1][0]
            l1_content = '\n'.join(
                f'Round {rid}: {s}' for rid, s in l1_summaries
            )
            # Truncate if too long
            if len(l1_content) > max_l1_tokens * 4:  # ~4 chars/token
                l1_content = l1_content[:max_l1_tokens * 4] + '...[truncated]'
            levels.append(SummaryLevel(
                level=1,
                content=l1_content,
                round_range=(first_round, last_round),
                token_count=estimate_tokens(l1_content),
            ))

        # --- L2: Block summaries ---
        if len(l1_summaries) > self.l2_block_size:
            l2_blocks = self._group_into_blocks(l1_summaries)
            l2_content_parts: List[str] = []
            for block_start, block_rounds in l2_blocks:
                block_text = '\n'.join(
                    f'  Round {rid}: {s}' for rid, s in block_rounds
                )
                if self.summary_provider:
                    prompt = (
                        'Summarize these conversation rounds in 1-2 sentences:\n'
                        f'{block_text}'
                    )
                    block_summary = self.summary_provider('You are a summarizer.', prompt)
                else:
                    block_summary = self._deterministic_block_summary(block_rounds)
                block_end = block_rounds[-1][0]
                l2_content_parts.append(
                    f'Rounds {block_start}-{block_end}: {block_summary}'
                )

            l2_content = '\n'.join(l2_content_parts)
            if len(l2_content) > max_l2_tokens * 4:
                l2_content = l2_content[:max_l2_tokens * 4] + '...[truncated]'
            levels.append(SummaryLevel(
                level=2,
                content=l2_content,
                round_range=(l1_summaries[0][0], l1_summaries[-1][0]),
                token_count=estimate_tokens(l2_content),
            ))

        # --- L3: Global summary ---
        l3_source = l1_content if not levels or len(l1_summaries) <= self.l2_block_size else l2_content
        if self.summary_provider:
            prompt = (
                'Summarize this entire conversation in one concise paragraph '
                '(max 200 words), focusing on goals, key decisions, and current state:\n'
                f'{l3_source}'
            )
            l3_content = self.summary_provider('You are a summarizer.', prompt)
        else:
            l3_content = self._deterministic_global_summary(l1_summaries)

        if len(l3_content) > max_l3_tokens * 4:
            l3_content = l3_content[:max_l3_tokens * 4] + '...[truncated]'
        levels.append(SummaryLevel(
            level=3,
            content=l3_content,
            round_range=(l1_summaries[0][0], l1_summaries[-1][0]),
            token_count=estimate_tokens(l3_content),
        ))

        return levels

    def _summarize_round(self, group: MessageGroup) -> str:
        """Generate a one-line summary for a single round."""
        tool_names: List[str] = []
        tool_outcomes: List[str] = []
        user_text = ''
        assistant_text = ''

        for msg in group.messages:
            role = msg.get('role', '')
            content = msg.get('content', [])

            if role == 'user':
                text = _extract_text_content(msg)
                if text:
                    user_text = text[:100]

            elif role == 'assistant':
                text = _extract_text_content(msg)
                if text:
                    assistant_text = text[:100]
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'tool_use':
                            tool_names.append(block.get('name', '?'))

            elif role == 'tool':
                is_error = msg.get('is_error', False)
                tool_outcomes.append('error' if is_error else 'ok')

        parts: List[str] = []
        if user_text:
            parts.append(user_text[:60])
        if tool_names:
            outcomes = ', '.join(tool_outcomes) if tool_outcomes else ''
            parts.append(f'tools: [{", ".join(tool_names)}] ({outcomes})')
        if assistant_text and not tool_names:
            parts.append(assistant_text[:60])

        return ' | '.join(parts) if parts else '(empty round)'

    def _group_into_blocks(
        self,
        l1_summaries: List[Tuple[int, str]],
    ) -> List[Tuple[int, List[Tuple[int, str]]]]:
        """Group L1 summaries into blocks of l2_block_size."""
        blocks: List[Tuple[int, List[Tuple[int, str]]]] = []
        for i in range(0, len(l1_summaries), self.l2_block_size):
            chunk = l1_summaries[i:i + self.l2_block_size]
            blocks.append((chunk[0][0], chunk))
        return blocks

    def _deterministic_block_summary(
        self,
        block_rounds: List[Tuple[int, str]],
    ) -> str:
        """Non-LLM block summary: extract tool usage stats."""
        tool_counts: Dict[str, int] = {}
        error_count = 0
        for _, summary in block_rounds:
            if 'error' in summary.lower():
                error_count += 1
            # Extract tool names from "tools: [x, y]" pattern
            import re
            match = re.search(r'tools: \[([^\]]+)\]', summary)
            if match:
                for t in match.group(1).split(','):
                    t = t.strip()
                    if t:
                        tool_counts[t] = tool_counts.get(t, 0) + 1

        parts = [f'{len(block_rounds)} rounds']
        if tool_counts:
            top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:3]
            parts.append(f'top tools: {", ".join(f"{t}({c})" for t, c in top_tools)}')
        if error_count:
            parts.append(f'{error_count} errors')
        return '; '.join(parts)

    def _deterministic_global_summary(
        self,
        l1_summaries: List[Tuple[int, str]],
    ) -> str:
        """Non-LLM global summary."""
        total = len(l1_summaries)
        error_rounds = sum(1 for _, s in l1_summaries if 'error' in s.lower())
        tool_rounds = sum(1 for _, s in l1_summaries if 'tools:' in s)
        return (
            f'{total} rounds total, {tool_rounds} with tool calls, '
            f'{error_rounds} with errors.'
        )


def hierarchical_compact_messages(
    messages: List[Dict[str, Any]],
    *,
    summarizer: HierarchicalSummarizer,
    keep_recent_rounds: int = 3,
    summary_level: int = 2,  # Which level to inject (1, 2, or 3)
) -> List[Dict[str, Any]]:
    """Apply hierarchical summarization to compress messages.

    Replaces older rounds with the appropriate summary level,
    keeping the most recent rounds intact.

    Args:
        messages: Full message list.
        summarizer: Configured HierarchicalSummarizer instance.
        keep_recent_rounds: Number of recent rounds to preserve verbatim.
        summary_level: Which summary level to inject (1=detailed, 3=brief).

    Returns:
        Compressed message list with summary prepended.
    """
    groups = group_messages_by_rounds(messages)
    if len(groups) <= keep_recent_rounds:
        return messages

    # Generate summaries
    levels = summarizer.summarize(messages)
    if not levels:
        return messages

    # Find the requested summary level (or closest available)
    target_level = next(
        (lv for lv in levels if lv.level == summary_level),
        levels[-1],  # fallback to most compressed
    )

    # Build result: summary + recent rounds
    result: List[Dict[str, Any]] = [
        {
            'role': 'system',
            'content': f'[Hierarchical Summary L{target_level.level}]\n{target_level.content}',
        }
    ]
    for group in groups[-keep_recent_rounds:]:
        result.extend(group.messages)

    return result


# ============== Prompt Caching ==============
# Based on Zero2Agent article #20 & API provider caching patterns
# Maintains a stable prefix to maximize cache hits with providers
# that support prompt caching (e.g. Claude, OpenAI).


@dataclass
class CacheSegment:
    """A segment of the prompt that can be cached."""
    content_hash: str  # SHA-256 of the content
    token_count: int
    messages: List[Dict[str, Any]]
    created_at: float = 0.0
    hit_count: int = 0


class PromptCacheManager:
    """Manages prompt caching to reduce costs and latency.

    Maintains a stable prefix of system + early conversation messages
    that rarely change, maximizing cache hits with API providers that
    support prefix caching (Claude, OpenAI).

    Strategy:
    - The system message(s) form the base cache layer (almost always identical)
    - Early conversation rounds that haven't been compacted form the second layer
    - Recent messages change frequently and are NOT cached

    Usage:
        cache = PromptCacheManager()
        # Build cacheable prefix
        prefix, suffix = cache.split_for_caching(messages)
        # Provider can cache prefix; suffix changes each turn
    """

    def __init__(
        self,
        *,
        min_cacheable_tokens: int = 1024,  # Providers need ≥1024 tokens to cache
        cache_suffix_rounds: int = 2,  # Recent rounds NOT cached
    ):
        self.min_cacheable_tokens = min_cacheable_tokens
        self.cache_suffix_rounds = cache_suffix_rounds
        self._segments: List[CacheSegment] = []
        self._cache_hits = 0
        self._cache_misses = 0

    def split_for_caching(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split messages into (cacheable_prefix, dynamic_suffix).

        The prefix is stable across turns (system + early conversation)
        and should be sent with cache_control hints to the provider.
        The suffix changes every turn and is sent fresh.

        Returns:
            (prefix_messages, suffix_messages)
        """
        if not messages:
            return [], []

        groups = group_messages_by_rounds(messages)

        if len(groups) <= self.cache_suffix_rounds + 1:
            # Too few rounds — cache just the system message(s)
            system_msgs = [m for m in messages if m.get('role') == 'system']
            rest = [m for m in messages if m.get('role') != 'system']
            prefix_tokens = estimate_messages_tokens(system_msgs)
            if prefix_tokens >= self.min_cacheable_tokens:
                return system_msgs, rest
            return [], messages

        # Split: everything up to (total - suffix_rounds) goes in prefix
        split_round = len(groups) - self.cache_suffix_rounds
        prefix_msgs: List[Dict[str, Any]] = []
        suffix_msgs: List[Dict[str, Any]] = []

        for i, group in enumerate(groups):
            if i < split_round:
                prefix_msgs.extend(group.messages)
            else:
                suffix_msgs.extend(group.messages)

        prefix_tokens = estimate_messages_tokens(prefix_msgs)

        # If prefix is too small, move more into it
        if prefix_tokens < self.min_cacheable_tokens and split_round > 1:
            # Just put everything in prefix except last round
            prefix_msgs = []
            suffix_msgs = []
            for i, group in enumerate(groups):
                if i < len(groups) - 1:
                    prefix_msgs.extend(group.messages)
                else:
                    suffix_msgs.extend(group.messages)

        # Update segment tracking
        content_hash = self._hash_messages(prefix_msgs)
        self._update_segment(content_hash, prefix_msgs)

        return prefix_msgs, suffix_msgs

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            'segments': len(self._segments),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / total if total > 0 else 0.0,
            'total_cached_tokens': sum(s.token_count for s in self._segments),
        }

    def _hash_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Compute a hash of messages for cache identity."""
        import hashlib
        # Use a deterministic serialization
        parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if isinstance(content, list):
                # Sort blocks for determinism
                for block in content:
                    if isinstance(block, dict):
                        parts.append(f"{role}:{block.get('type', '')}:{hash(str(block))}")
            else:
                parts.append(f"{role}:{hash(str(content))}")
        combined = '|'.join(parts)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]

    def _update_segment(self, content_hash: str, messages: List[Dict[str, Any]]) -> None:
        """Update segment tracking on each call."""
        now = time.time()
        for seg in self._segments:
            if seg.content_hash == content_hash:
                seg.hit_count += 1
                self._cache_hits += 1
                return

        # New segment — cache miss
        self._cache_misses += 1
        self._segments.append(CacheSegment(
            content_hash=content_hash,
            token_count=estimate_messages_tokens(messages),
            messages=messages,
            created_at=now,
        ))

        # Keep only the most recent 5 segments to limit memory
        if len(self._segments) > 5:
            self._segments = sorted(
                self._segments,
                key=lambda s: s.created_at,
                reverse=True,
            )[:5]


def add_cache_control_hints(
    messages: List[Dict[str, Any]],
    *,
    cacheable_prefix_count: int,
) -> List[Dict[str, Any]]:
    """Add cache_control hints to the last message in the cacheable prefix.

    This is provider-specific: Anthropic Claude uses a `cache_control`
    field on content blocks to mark breakpoints. Other providers may
    use different mechanisms.

    Args:
        messages: Full message list.
        cacheable_prefix_count: Number of messages in the cacheable prefix.

    Returns:
        Modified messages with cache_control hints added.
    """
    if cacheable_prefix_count <= 0 or cacheable_prefix_count > len(messages):
        return messages

    result = [dict(msg) for msg in messages]  # shallow copy

    # Add ephemeral cache_control to the last prefix message
    idx = cacheable_prefix_count - 1
    msg = result[idx]
    content = msg.get('content', '')

    if isinstance(content, str):
        # Convert string content to block format with cache_control
        result[idx] = {
            **msg,
            'content': [
                {
                    'type': 'text',
                    'text': content,
                    'cache_control': {'type': 'ephemeral'},
                }
            ],
        }
    elif isinstance(content, list) and content:
        # Add cache_control to the last content block
        new_content = list(content)
        last_block = dict(new_content[-1])
        last_block['cache_control'] = {'type': 'ephemeral'}
        new_content[-1] = last_block
        result[idx] = {**msg, 'content': new_content}

    return result
