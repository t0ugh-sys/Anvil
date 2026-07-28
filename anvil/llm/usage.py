from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


PRICING_FILE = Path(__file__).parent / "pricing.json"


@dataclass
class TokenUsageRecord:
    """A single API call's token usage."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    model: str = ''

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of input tokens served from cache (0.0-1.0)."""
        total = self.input_tokens + self.cache_read_input_tokens
        if total <= 0:
            return 0.0
        return self.cache_read_input_tokens / total


class TokenUsageTracker:
    """Tracks cumulative token usage across API calls.

    Each Anthropic API response includes a ``usage`` field with exact
    token counts. This tracker accumulates them for cost monitoring.

    Usage::

        tracker = TokenUsageTracker()
        invoke = anthropic_invoke_factory(..., usage_tracker=invoke)
        result = invoke(prompt)
        print(tracker.summary())
    """

    def __init__(self) -> None:
        self._records: list[TokenUsageRecord] = []

    def record(self, usage: dict, model: str = '') -> None:
        """Record token usage from an API response."""
        self._records.append(TokenUsageRecord(
            input_tokens=int(usage.get('input_tokens', 0) or 0),
            output_tokens=int(usage.get('output_tokens', 0) or 0),
            cache_creation_input_tokens=int(usage.get('cache_creation_input_tokens', 0) or 0),
            cache_read_input_tokens=int(usage.get('cache_read_input_tokens', 0) or 0),
            model=model,
        ))

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._records)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(r.cache_creation_input_tokens for r in self._records)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(r.cache_read_input_tokens for r in self._records)

    @property
    def call_count(self) -> int:
        return len(self._records)

    def summary(self) -> dict:
        return {
            'calls': self.call_count,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'cache_creation_tokens': self.total_cache_creation_tokens,
            'cache_read_tokens': self.total_cache_read_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'cache_hit_rate': self.cache_hit_rate,
            'estimated_cost_savings': self.estimated_cost_savings,
        }

    def last(self) -> TokenUsageRecord | None:
        return self._records[-1] if self._records else None

    def reset(self) -> None:
        self._records.clear()

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache across all calls (0.0-1.0)."""
        total_cacheable = self.total_input_tokens + self.total_cache_read_tokens
        if total_cacheable <= 0:
            return 0.0
        return self.total_cache_read_tokens / total_cacheable

    @property
    def estimated_cost_savings(self) -> dict:
        """Estimate cost savings from prompt caching.

        Based on Claude API pricing:
        - Cache write: 1.25x standard input price
        - Cache read: 0.1x standard input price (90% discount)
        - Standard input: 1.0x

        Returns dict with estimated costs in abstract units (multiply by
        model-specific per-token price for actual dollars).
        """
        standard_input = self.total_input_tokens
        cache_write = self.total_cache_creation_tokens
        cache_read = self.total_cache_read_tokens

        # Without caching: all tokens at standard rate
        cost_without_cache = standard_input * 1.0

        # With caching: creation at 1.25x, reads at 0.1x, rest at 1.0x
        uncached_tokens = max(0, standard_input - cache_write - cache_read)
        cost_with_cache = (
            uncached_tokens * 1.0
            + cache_write * 1.25
            + cache_read * 0.1
        )

        savings = max(0, cost_without_cache - cost_with_cache)
        return {
            'cost_without_cache': cost_without_cache,
            'cost_with_cache': cost_with_cache,
            'savings': savings,
            'savings_percent': (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0.0,
        }


# Claude API pricing (per million tokens) — fallback when pricing.json is absent.
_CLAUDE_PRICING = {
    'claude-opus-5': {'input': 15.0, 'output': 75.0, 'cache_write': 18.75, 'cache_read': 1.5},
    'claude-opus-4': {'input': 15.0, 'output': 75.0, 'cache_write': 18.75, 'cache_read': 1.5},
    'claude-fable-5': {'input': 3.0, 'output': 15.0, 'cache_write': 3.75, 'cache_read': 0.3},
    'claude-sonnet-5': {'input': 3.0, 'output': 15.0, 'cache_write': 3.75, 'cache_read': 0.3},
    'claude-sonnet-4': {'input': 3.0, 'output': 15.0, 'cache_write': 3.75, 'cache_read': 0.3},
    'claude-haiku-4-5': {'input': 0.80, 'output': 4.0, 'cache_write': 1.0, 'cache_read': 0.08},
    'claude-haiku-3.5': {'input': 0.80, 'output': 4.0, 'cache_write': 1.0, 'cache_read': 0.08},
    'default': {'input': 3.0, 'output': 15.0, 'cache_write': 3.75, 'cache_read': 0.3},
}


def _load_pricing_from_file() -> dict | None:
    """Load pricing table from pricing.json; return None on any failure."""
    try:
        data = json.loads(PRICING_FILE.read_text(encoding='utf-8'))
        models = data.get('models')
        return models if isinstance(models, dict) else None
    except Exception:
        return None


class CostTracker:
    """Track estimated API costs based on Claude pricing.

    Uses model-specific pricing to estimate actual dollar costs.
    Integrates with TokenUsageTracker for token counts.

    Usage::

        cost = CostTracker(model='claude-sonnet-4')
        cost.add_from_tracker(usage_tracker)
        print(cost.summary())
    """

    def __init__(self, model: str = 'claude-sonnet-4'):
        self.model = model
        self._pricing = self._resolve_pricing(model)
        self._calls: list = []

    @staticmethod
    def _resolve_pricing(model: str) -> dict:
        table = _load_pricing_from_file() or _CLAUDE_PRICING
        model_lower = model.lower()
        for key, pricing in table.items():
            if key != 'default' and key in model_lower:
                return pricing
        return table.get('default', _CLAUDE_PRICING['default'])

    def add(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        """Record a single API call's token usage."""
        self._calls.append({
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cache_creation_tokens': cache_creation_tokens,
            'cache_read_tokens': cache_read_tokens,
        })

    def add_from_tracker(self, tracker: TokenUsageTracker) -> None:
        """Import all records from a TokenUsageTracker."""
        for record in tracker._records:
            self.add(
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cache_creation_tokens=record.cache_creation_input_tokens,
                cache_read_tokens=record.cache_read_input_tokens,
            )

    @property
    def total_cost(self) -> float:
        """Total estimated cost in dollars."""
        return sum(self._call_cost(c) for c in self._calls)

    @property
    def cost_without_cache(self) -> float:
        """What the cost would be without any caching."""
        total = 0.0
        for c in self._calls:
            total_input = c['input_tokens'] + c['cache_read_tokens']
            total += (total_input / 1_000_000) * self._pricing['input']
            total += (c['output_tokens'] / 1_000_000) * self._pricing['output']
        return total

    @property
    def savings(self) -> float:
        """Estimated savings from caching."""
        return max(0, self.cost_without_cache - self.total_cost)

    def _call_cost(self, call: dict) -> float:
        """Calculate cost for a single call."""
        cost = 0.0
        # Regular input tokens (excluding cache-related)
        regular_input = max(0, call['input_tokens'] - call['cache_creation_tokens'] - call['cache_read_tokens'])
        cost += (regular_input / 1_000_000) * self._pricing['input']
        cost += (call['cache_creation_tokens'] / 1_000_000) * self._pricing['cache_write']
        cost += (call['cache_read_tokens'] / 1_000_000) * self._pricing['cache_read']
        cost += (call['output_tokens'] / 1_000_000) * self._pricing['output']
        return cost

    def summary(self) -> dict:
        """Return cost summary."""
        total_input = sum(c['input_tokens'] for c in self._calls)
        total_output = sum(c['output_tokens'] for c in self._calls)
        total_cache_create = sum(c['cache_creation_tokens'] for c in self._calls)
        total_cache_read = sum(c['cache_read_tokens'] for c in self._calls)
        return {
            'model': self.model,
            'calls': len(self._calls),
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_cache_creation_tokens': total_cache_create,
            'total_cache_read_tokens': total_cache_read,
            'total_cost_usd': round(self.total_cost, 6),
            'cost_without_cache_usd': round(self.cost_without_cache, 6),
            'savings_usd': round(self.savings, 6),
            'savings_percent': round(
                (self.savings / self.cost_without_cache * 100) if self.cost_without_cache > 0 else 0.0,
                2,
            ),
            'pricing_per_million': self._pricing,
        }
