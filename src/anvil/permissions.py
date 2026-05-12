from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .policies import Capability


class PermissionMode(str):
    allow = 'allow'
    deny = 'deny'
    ask = 'ask'


@dataclass(frozen=True)
class PermissionRequest:
    tool_name: str
    arguments: Dict[str, object]
    workspace_root: Path
    capabilities: Tuple[Capability, ...]
    cache_key: str


@dataclass(frozen=True)
class PermissionDecision:
    mode: str
    reason: str
    cache_key: str
    cached: bool = False

    @property
    def allowed(self) -> bool:
        return self.mode == PermissionMode.allow


# ============== Permission Rules ==============

@dataclass(frozen=True)
class PermissionRule:
    """A single permission rule: tool pattern -> mode."""
    tool_pattern: str  # Exact tool name or '*' for all
    mode: str  # 'allow', 'deny', 'ask'
    source: str = ''  # Where this rule came from (user, project, policy)


@dataclass(frozen=True)
class PermissionRuleSet:
    """A set of rules from a single source."""
    source: str
    rules: Tuple[PermissionRule, ...] = ()

    def find_rule(self, tool_name: str) -> PermissionRule | None:
        """Find a matching rule for the given tool name."""
        # Exact match first, then wildcard
        for rule in self.rules:
            if rule.tool_pattern == tool_name:
                return rule
        for rule in self.rules:
            if rule.tool_pattern == '*':
                return rule
        return None


# Rule source precedence (lowest to highest)
RULE_SOURCE_PRECEDENCE = ('user', 'project', 'session', 'policy')


def merge_rule_sets(rule_sets: Sequence[PermissionRuleSet]) -> Tuple[PermissionRule, ...]:
    """Merge rule sets by precedence (later sources override earlier)."""
    merged: Dict[str, PermissionRule] = {}
    for rule_set in rule_sets:
        for rule in rule_set.rules:
            merged[rule.tool_pattern] = rule
    return tuple(merged.values())


# ============== Denial Tracking ==============

class DenialTracker:
    """Track consecutive denials to fall back to interactive prompting."""

    def __init__(self, threshold: int = 3) -> None:
        self._consecutive_denials: Dict[str, int] = {}
        self._threshold = threshold

    def record_denial(self, tool_name: str) -> None:
        self._consecutive_denials[tool_name] = self._consecutive_denials.get(tool_name, 0) + 1

    def record_allow(self, tool_name: str) -> None:
        self._consecutive_denials.pop(tool_name, None)

    def should_prompt(self, tool_name: str) -> bool:
        """Check if we should fall back to interactive prompting."""
        return self._consecutive_denials.get(tool_name, 0) >= self._threshold

    def reset(self, tool_name: str) -> None:
        self._consecutive_denials.pop(tool_name, None)


# ============== Permission Pipeline ==============

class PermissionPipeline:
    """Multi-step permission evaluation with precedence cascade.

    Inspired by Claude Code's hasPermissionsToUseTool:
    Step 1: Check deny rules (fail-fast)
    Step 2: Check ask rules
    Step 3: Check tool-specific permissions
    Step 4: Check mode-based bypass
    Step 5: Check always-allow rules
    """

    def __init__(
        self,
        *,
        mode_name: str = 'balanced',
        rule_sets: Sequence[PermissionRuleSet] = (),
        cache: Mapping[str, str] | None = None,
        denial_tracker: DenialTracker | None = None,
    ) -> None:
        normalized = mode_name.strip().lower() or 'balanced'
        if normalized not in {'strict', 'balanced', 'unsafe'}:
            raise ValueError(f'unknown permission mode: {mode_name}')
        self.mode_name = normalized
        self._rule_sets = list(rule_sets)
        self._merged_rules = merge_rule_sets(rule_sets)
        self._cache: Dict[str, str] = dict(cache or {})
        self._denial_tracker = denial_tracker or DenialTracker()

    @property
    def cache(self) -> Dict[str, str]:
        return dict(self._cache)

    def cache_key_for(self, tool_name: str, capabilities: Iterable[Capability]) -> str:
        required = sorted(item.value for item in capabilities)
        suffix = ','.join(required) or 'none'
        return f'{tool_name}:{suffix}'

    def build_request(
        self,
        *,
        tool_name: str,
        arguments: Dict[str, object],
        workspace_root: Path,
        capabilities: Tuple[Capability, ...],
    ) -> PermissionRequest:
        cache_key = self.cache_key_for(tool_name, capabilities)
        return PermissionRequest(
            tool_name=tool_name,
            arguments=arguments,
            workspace_root=workspace_root,
            capabilities=capabilities,
            cache_key=cache_key,
        )

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        # Check cache first
        cached_mode = self._cache.get(request.cache_key)
        if cached_mode in {PermissionMode.allow, PermissionMode.deny, PermissionMode.ask}:
            return PermissionDecision(
                mode=cached_mode,
                reason=f'cached decision for {request.tool_name}',
                cache_key=request.cache_key,
                cached=True,
            )

        # Step 1: Check deny rules (fail-fast)
        deny_rule = self._find_rule_for_mode(request.tool_name, PermissionMode.deny)
        if deny_rule:
            self._denial_tracker.record_denial(request.tool_name)
            return PermissionDecision(
                mode=PermissionMode.deny,
                reason=f'denied by {deny_rule.source} rule',
                cache_key=request.cache_key,
            )

        # Step 2: Check ask rules
        ask_rule = self._find_rule_for_mode(request.tool_name, PermissionMode.ask)
        if ask_rule:
            return PermissionDecision(
                mode=PermissionMode.ask,
                reason=f'ask required by {ask_rule.source} rule',
                cache_key=request.cache_key,
            )

        # Step 3: Check allow rules
        allow_rule = self._find_rule_for_mode(request.tool_name, PermissionMode.allow)
        if allow_rule:
            self._denial_tracker.record_allow(request.tool_name)
            self._cache[request.cache_key] = PermissionMode.allow
            return PermissionDecision(
                mode=PermissionMode.allow,
                reason=f'allowed by {allow_rule.source} rule',
                cache_key=request.cache_key,
            )

        # Step 4: Mode-based default
        mode = self._default_mode_for(request.capabilities)
        reason = self._reason_for(mode, request.tool_name, request.capabilities)
        if mode != PermissionMode.ask:
            self._cache[request.cache_key] = mode
        if mode == PermissionMode.deny:
            self._denial_tracker.record_denial(request.tool_name)
        else:
            self._denial_tracker.record_allow(request.tool_name)
        return PermissionDecision(mode=mode, reason=reason, cache_key=request.cache_key, cached=False)

    def record_decision(self, cache_key: str, mode: str) -> None:
        if mode in {PermissionMode.allow, PermissionMode.deny, PermissionMode.ask}:
            self._cache[cache_key] = mode

    def _find_rule_for_mode(self, tool_name: str, mode: str) -> PermissionRule | None:
        """Find a rule matching the tool name with the given mode."""
        for rule in self._merged_rules:
            if rule.mode == mode and (rule.tool_pattern == tool_name or rule.tool_pattern == '*'):
                return rule
        return None

    def _default_mode_for(self, capabilities: Tuple[Capability, ...]) -> str:
        if self.mode_name == 'unsafe':
            return PermissionMode.allow

        read_safe = {Capability.read, Capability.memory}
        if all(item in read_safe for item in capabilities):
            return PermissionMode.allow

        if self.mode_name == 'strict':
            return PermissionMode.deny
        return PermissionMode.ask

    def _reason_for(self, mode: str, tool_name: str, capabilities: Tuple[Capability, ...]) -> str:
        capability_names = ','.join(sorted(item.value for item in capabilities)) or 'none'
        if mode == PermissionMode.allow:
            return f'{tool_name} allowed in {self.mode_name} mode'
        if mode == PermissionMode.deny:
            return f'{tool_name} denied in {self.mode_name} mode ({capability_names})'
        return f'{tool_name} requires approval in {self.mode_name} mode ({capability_names})'


# Keep PermissionManager as alias for backward compatibility
PermissionManager = PermissionPipeline
