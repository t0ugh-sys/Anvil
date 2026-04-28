from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

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


class PermissionManager:
    def __init__(
        self,
        *,
        mode_name: str,
        cache: Mapping[str, str] | None = None,
    ) -> None:
        normalized = mode_name.strip().lower() or 'balanced'
        if normalized not in {'strict', 'balanced', 'unsafe'}:
            raise ValueError(f'unknown permission mode: {mode_name}')
        self.mode_name = normalized
        self._cache: Dict[str, str] = dict(cache or {})

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
        cached_mode = self._cache.get(request.cache_key)
        if cached_mode in {PermissionMode.allow, PermissionMode.deny, PermissionMode.ask}:
            return PermissionDecision(
                mode=cached_mode,
                reason=f'cached decision for {request.tool_name}',
                cache_key=request.cache_key,
                cached=True,
            )

        mode = self._default_mode_for(request.capabilities)
        reason = self._reason_for(mode, request.tool_name, request.capabilities)
        if mode != PermissionMode.ask:
            self._cache[request.cache_key] = mode
        return PermissionDecision(mode=mode, reason=reason, cache_key=request.cache_key, cached=False)

    def record_decision(self, cache_key: str, mode: str) -> None:
        if mode in {PermissionMode.allow, PermissionMode.deny, PermissionMode.ask}:
            self._cache[cache_key] = mode

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
