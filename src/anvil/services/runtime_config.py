from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..session import SessionStore

VALID_PROVIDERS = {'mock', 'openai_compatible', 'anthropic', 'gemini'}
VALID_WIRE_APIS = {'chat_completions', 'responses'}


@dataclass
class InteractiveRuntimeConfig:
    provider: str
    model: str
    base_url: str
    wire_api: str

    def to_dict(self) -> dict[str, object]:
        return {
            'provider': self.provider,
            'model': self.model,
            'base_url': self.base_url,
            'wire_api': self.wire_api,
        }


class RuntimeConfigManager:
    def __init__(self, *, session_store: SessionStore, config: InteractiveRuntimeConfig) -> None:
        self.session_store = session_store
        self.config = config

    @classmethod
    def from_args(cls, *, session_store: SessionStore, args: Any) -> 'RuntimeConfigManager':
        saved = session_store.state.runtime_config
        provider = str(saved.get('provider') or getattr(args, 'provider', 'mock'))
        model = str(saved.get('model') or getattr(args, 'model', 'mock-model'))
        base_url = str(saved.get('base_url') or getattr(args, 'base_url', ''))
        wire_api = str(saved.get('wire_api') or getattr(args, 'wire_api', 'chat_completions'))
        manager = cls(
            session_store=session_store,
            config=InteractiveRuntimeConfig(
                provider=provider,
                model=model,
                base_url=base_url,
                wire_api=wire_api,
            ),
        )
        manager.persist()
        return manager

    def persist(self) -> None:
        self.session_store.update_runtime_config(self.config.to_dict())

    def set_provider(self, provider: str) -> str:
        candidate = provider.strip()
        if candidate not in VALID_PROVIDERS:
            raise ValueError(f'unknown provider: {candidate}')
        self.config.provider = candidate
        if candidate == 'mock' and not self.config.model:
            self.config.model = 'mock-model'
        self.persist()
        return self.summary()

    def set_model(self, model: str) -> str:
        candidate = model.strip()
        if not candidate:
            raise ValueError('model must not be empty')
        self.config.model = candidate
        self.persist()
        return self.summary()

    def set_wire_api(self, wire_api: str) -> str:
        candidate = wire_api.strip()
        if candidate not in VALID_WIRE_APIS:
            raise ValueError(f'unknown wire api: {candidate}')
        self.config.wire_api = candidate
        self.persist()
        return self.summary()

    def set_base_url(self, base_url: str) -> str:
        self.config.base_url = base_url.strip()
        self.persist()
        return self.summary()

    def apply_to_args(self, args: Any) -> None:
        args.provider = self.config.provider
        args.model = self.config.model
        args.base_url = self.config.base_url
        args.wire_api = self.config.wire_api

    def summary(self) -> str:
        return (
            'runtime_config:\n'
            f'provider: {self.config.provider}\n'
            f'model: {self.config.model}\n'
            f'wire_api: {self.config.wire_api}\n'
            f'base_url: {self.config.base_url or "(empty)"}'
        )
