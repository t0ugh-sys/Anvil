from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


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


@dataclass(frozen=True)
class CompressionConfig:
    micro_keep_last_results: int = 3
    max_context_tokens: int = 50000
    recent_transcript_entries: int = 8

    def validate(self) -> None:
        if self.micro_keep_last_results < 1:
            raise ValueError('micro_keep_last_results must be >= 1')
        if self.max_context_tokens < 1:
            raise ValueError('max_context_tokens must be >= 1')
        if self.recent_transcript_entries < 1:
            raise ValueError('recent_transcript_entries must be >= 1')


@dataclass
class CompactManager:
    requested: bool = False
    reason: str = ''

    def request(self, reason: str = '') -> None:
        self.requested = True
        self.reason = reason.strip()


def estimate_tokens(parts: Iterable[str]) -> int:
    total_chars = sum(len(part) for part in parts if part)
    return max(1, total_chars // 4) if total_chars else 0


def micro_compact_entries(
    entries: Tuple[TranscriptEntry, ...],
    *,
    keep_last_results: int,
) -> Tuple[TranscriptEntry, ...]:
    result_indices = [index for index, entry in enumerate(entries) if entry.kind == 'tool_result']
    keep_indices = set(result_indices[-keep_last_results:])
    compacted: List[TranscriptEntry] = []
    for index, entry in enumerate(entries):
        if entry.kind != 'tool_result' or index in keep_indices:
            compacted.append(entry)
            continue
        compacted.append(
            TranscriptEntry(
                kind='tool_result',
                tool_name=entry.tool_name,
                call_id=entry.call_id,
                ok=entry.ok,
                content=f'[Previous: used {entry.tool_name or "tool"}]',
            )
        )
    return tuple(compacted)


def summarize_entries_deterministically(
    *,
    goal: str,
    previous_summary: str,
    entries: Tuple[TranscriptEntry, ...],
) -> str:
    interesting_lines: List[str] = []
    for entry in entries[-12:]:
        line = entry.render_line()
        if line:
            interesting_lines.append(f'- {line[:240]}')

    lines = [
        f'Goal: {goal}',
        f'Previous summary: {previous_summary or "none"}',
        'Recent transcript:',
        *interesting_lines,
    ]
    return '\n'.join(lines).strip()


def archive_transcript(
    *,
    transcripts_dir: Path,
    compaction_index: int,
    reason: str,
    goal: str,
    previous_summary: str,
    entries: Tuple[TranscriptEntry, ...],
) -> Path:
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
