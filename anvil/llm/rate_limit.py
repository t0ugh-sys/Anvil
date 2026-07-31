from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict

__all__ = ['RateLimitTracker', 'RateLimitSnapshot']


@dataclass
class RateLimitSnapshot:
    """Snapshot of rate limit state from API response headers."""
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_requests: datetime | None = None
    reset_tokens: datetime | None = None
    limit_requests: int | None = None
    limit_tokens: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RateLimitTracker:
    """Tracks API rate limit state from response headers.

    Supports Anthropic-style headers:
    - anthropic-ratelimit-requests-remaining
    - anthropic-ratelimit-tokens-remaining
    - anthropic-ratelimit-requests-reset
    - anthropic-ratelimit-tokens-reset
    - anthropic-ratelimit-requests-limit
    - anthropic-ratelimit-tokens-limit
    """

    def __init__(self) -> None:
        self._snapshots: list[RateLimitSnapshot] = []
        self._latest: RateLimitSnapshot | None = None

    def record_headers(self, headers: Dict[str, str]) -> None:
        """Extract and record rate limit info from response headers."""
        snapshot = RateLimitSnapshot()

        # Anthropic headers
        if 'anthropic-ratelimit-requests-remaining' in headers:
            try:
                snapshot.remaining_requests = int(headers['anthropic-ratelimit-requests-remaining'])
            except (ValueError, TypeError):
                pass

        if 'anthropic-ratelimit-tokens-remaining' in headers:
            try:
                snapshot.remaining_tokens = int(headers['anthropic-ratelimit-tokens-remaining'])
            except (ValueError, TypeError):
                pass

        if 'anthropic-ratelimit-requests-limit' in headers:
            try:
                snapshot.limit_requests = int(headers['anthropic-ratelimit-requests-limit'])
            except (ValueError, TypeError):
                pass

        if 'anthropic-ratelimit-tokens-limit' in headers:
            try:
                snapshot.limit_tokens = int(headers['anthropic-ratelimit-tokens-limit'])
            except (ValueError, TypeError):
                pass

        if 'anthropic-ratelimit-requests-reset' in headers:
            try:
                snapshot.reset_requests = datetime.fromisoformat(
                    headers['anthropic-ratelimit-requests-reset'].replace('Z', '+00:00')
                )
            except (ValueError, TypeError):
                pass

        if 'anthropic-ratelimit-tokens-reset' in headers:
            try:
                snapshot.reset_tokens = datetime.fromisoformat(
                    headers['anthropic-ratelimit-tokens-reset'].replace('Z', '+00:00')
                )
            except (ValueError, TypeError):
                pass

        self._snapshots.append(snapshot)
        self._latest = snapshot

    @property
    def latest(self) -> RateLimitSnapshot | None:
        """Get the most recent rate limit snapshot."""
        return self._latest

    def summary(self) -> dict:
        """Return current rate limit state as a dict."""
        if self._latest is None:
            return {
                'has_data': False,
                'remaining_requests': None,
                'remaining_tokens': None,
                'limit_requests': None,
                'limit_tokens': None,
                'reset_requests': None,
                'reset_tokens': None,
            }

        snap = self._latest
        return {
            'has_data': True,
            'remaining_requests': snap.remaining_requests,
            'remaining_tokens': snap.remaining_tokens,
            'limit_requests': snap.limit_requests,
            'limit_tokens': snap.limit_tokens,
            'reset_requests': snap.reset_requests.isoformat() if snap.reset_requests else None,
            'reset_tokens': snap.reset_tokens.isoformat() if snap.reset_tokens else None,
        }
