from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ['MemoryEntry', 'MemoryResult', 'VectorMemoryStore']

DIMS = 256
_VECTOR_FORMAT = f'{DIMS}f'
_VECTOR_BYTES = struct.calcsize(_VECTOR_FORMAT)

EmbedFn = Callable[[str], List[float]]


@dataclass(frozen=True)
class MemoryEntry:
    key: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ''


@dataclass(frozen=True)
class MemoryResult:
    entry: MemoryEntry
    score: float


def _default_embed(text: str) -> List[float]:
    """Hash bag-of-words embedding — zero external dependencies, 256 dims."""
    tokens = re.findall(r'\w+', text.lower())
    vec = [0.0] * DIMS
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS
        vec[h] += 1.0
        for i in range(len(token) - 1):
            bigram = token[i:i + 2]
            h2 = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % DIMS
            vec[h2] += 0.5
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _pack(vec: List[float]) -> bytes:
    return struct.pack(_VECTOR_FORMAT, *vec)


def _unpack(data: bytes) -> List[float]:
    return list(struct.unpack(_VECTOR_FORMAT, data))


class VectorMemoryStore:
    """SQLite-backed semantic memory with local hash-embedding (zero external deps).

    Supports three tiers — pass tier='long' (default), 'project', or 'short'.
    All tiers share the same db file; query by tier to scope results.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        embed_fn: Optional[EmbedFn] = None,
        tier: str = 'long',
    ) -> None:
        self.db_path = db_path
        self.tier = tier
        self._embed: EmbedFn = embed_fn or _default_embed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                '''CREATE TABLE IF NOT EXISTS memories (
                       key        TEXT PRIMARY KEY,
                       text       TEXT NOT NULL,
                       vector     BLOB NOT NULL,
                       tier       TEXT NOT NULL,
                       metadata   TEXT NOT NULL DEFAULT '{}',
                       created_at TEXT NOT NULL DEFAULT ''
                   )'''
            )
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tier ON memories (tier)')
            conn.commit()

    def add(self, key: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        from anvil.config.run_schema import utc_now_iso
        vec = self._embed(text)
        with self._connect() as conn:
            conn.execute(
                '''INSERT OR REPLACE INTO memories
                       (key, text, vector, tier, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (key, text, _pack(vec), self.tier, json.dumps(metadata or {}), utc_now_iso()),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute('DELETE FROM memories WHERE key = ?', (key,))
            conn.commit()

    def get(self, key: str) -> Optional[MemoryEntry]:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT key, text, metadata, created_at FROM memories WHERE key = ?', (key,)
            ).fetchone()
        if row is None:
            return None
        return MemoryEntry(
            key=row['key'],
            text=row['text'],
            metadata=json.loads(row['metadata']),
            created_at=row['created_at'],
        )

    def search(
        self, query: str, top_k: int = 5, *, tier: Optional[str] = None
    ) -> List[MemoryResult]:
        q_vec = self._embed(query)
        target_tier = tier if tier is not None else self.tier
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT key, text, vector, metadata, created_at FROM memories WHERE tier = ?',
                (target_tier,),
            ).fetchall()
        scored: List[Tuple[float, MemoryEntry]] = []
        for row in rows:
            score = _cosine(q_vec, _unpack(row['vector']))
            entry = MemoryEntry(
                key=row['key'],
                text=row['text'],
                metadata=json.loads(row['metadata']),
                created_at=row['created_at'],
            )
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [MemoryResult(entry=e, score=s) for s, e in scored[:top_k]]

    def count(self, *, tier: Optional[str] = None) -> int:
        target_tier = tier if tier is not None else self.tier
        with self._connect() as conn:
            return conn.execute(
                'SELECT COUNT(*) FROM memories WHERE tier = ?', (target_tier,)
            ).fetchone()[0]

    def all_entries(self, *, tier: Optional[str] = None) -> List[MemoryEntry]:
        target_tier = tier if tier is not None else self.tier
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT key, text, metadata, created_at FROM memories WHERE tier = ?',
                (target_tier,),
            ).fetchall()
        return [
            MemoryEntry(
                key=row['key'],
                text=row['text'],
                metadata=json.loads(row['metadata']),
                created_at=row['created_at'],
            )
            for row in rows
        ]

    def inject_into_prompt(self, query: str, top_k: int = 5) -> str:
        """Return formatted relevant memories suitable for system prompt injection."""
        results = self.search(query, top_k=top_k)
        if not results:
            return ''
        lines = ['Relevant memories from previous sessions:']
        for i, r in enumerate(results, 1):
            lines.append(f'{i}. [{r.entry.key}] {r.entry.text}')
        return '\n'.join(lines)
