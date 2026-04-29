"""
High-level API for Anvil

Provides a simple programmatic interface for using Anvil.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from anvil.core.agent import AnvilAgent
from anvil.core.types import StopConfig
from anvil.llm.providers import build_invoke_from_args
from anvil.steps.json_loop import JsonLoopState, make_json_decision_step

from .errors import (
    validate_goal,
    validate_max_steps,
    validate_model,
    validate_provider,
    validate_strategy,
    validate_temperature,
)


@dataclass
class AgentConfig:
    """Configuration for Anvil."""
    provider: str = "mock"
    model: str = "mock-model"
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_steps: int = 20
    timeout_s: float = 60.0
    history_window: int = 3
    strategy: str = "json_llm"
    workspace: Path = field(default_factory=Path.cwd)
    fallback_model: list[str] = field(default_factory=list)
    wire_api: str = "chat_completions"
    provider_timeout_s: float = 60.0
    provider_debug: bool = False
    provider_header: list[str] = field(default_factory=list)
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    retry_http_code: list[int] = field(default_factory=list)
    
    def validate(self) -> "AgentConfig":
        """Validate configuration."""
        self.provider = validate_provider(self.provider)
        self.model = validate_model(self.model)
        self.temperature = validate_temperature(self.temperature)
        self.max_steps = validate_max_steps(self.max_steps)
        self.strategy = validate_strategy(self.strategy)
        self.workspace = Path(self.workspace)
        return self


@dataclass
class AgentResult:
    """Result from running an agent."""
    success: bool
    output: str
    steps: int
    stop_reason: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "metadata": self.metadata,
        }


class AnvilAPI:
    """High-level API for Anvil."""
    
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        # Using Any for flexibility - actual decider has 5 parameters
        self._invoke_fn: Any = None
    
    def set_provider(self, invoke_fn: Callable[..., str]) -> "AnvilAPI":
        """Set custom LLM provider function."""
        self._invoke_fn = invoke_fn
        return self

    def _build_provider_invoke(self, *, mode: str) -> Callable[[str], str]:
        if self._invoke_fn is not None:
            return self._invoke_fn

        self.config.validate()
        args = argparse.Namespace(
            provider=self.config.provider,
            model=self.config.model,
            fallback_model=list(self.config.fallback_model),
            base_url=self.config.base_url,
            wire_api=self.config.wire_api,
            api_key_env=self.config.api_key_env,
            temperature=self.config.temperature,
            provider_timeout_s=self.config.provider_timeout_s,
            provider_debug=self.config.provider_debug,
            provider_header=list(self.config.provider_header),
            max_retries=self.config.max_retries,
            retry_backoff_s=self.config.retry_backoff_s,
            retry_http_code=list(self.config.retry_http_code),
        )
        return build_invoke_from_args(args, mode=mode)
    
    def run(self, goal: str) -> AgentResult:
        """Run the agent with a goal."""
        # Validate goal
        goal = validate_goal(goal)
        
        try:
            step = make_json_decision_step(
                self._build_provider_invoke(mode="json"),
                history_window=self.config.history_window,
            )
            
            # Create agent
            stop = StopConfig(
                max_steps=self.config.max_steps,
                max_elapsed_s=self.config.timeout_s,
            )
            agent = AnvilAgent(step=step, stop=stop)
            
            # Run
            result = agent.run(
                goal=goal,
                initial_state=JsonLoopState(),
            )
            
            return AgentResult(
                success=result.done,
                output=result.final_output or "",
                steps=result.steps,
                stop_reason=result.stop_reason.value,
                error=None,
                metadata={"elapsed_s": result.elapsed_s},
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                steps=0,
                stop_reason="error",
                error=str(e),
            )
    
    def run_coding(self, goal: str) -> AgentResult:
        """Run the coding agent with a goal."""
        from anvil.coding_agent import run_coding_agent, build_coding_step
        
        goal = validate_goal(goal)
        
        try:
            invoke = self._build_provider_invoke(mode="coding")
            
            build_coding_step(
                invoke,
                workspace_root=self.config.workspace,
            )
            
            stop = StopConfig(
                max_steps=self.config.max_steps,
                max_elapsed_s=self.config.timeout_s,
            )
            
            result = run_coding_agent(
                goal=goal,
                decider=invoke,
                workspace_root=self.config.workspace,
                stop=stop,
            )
            
            return AgentResult(
                success=result.done,
                output=result.final_output or "",
                steps=result.steps,
                stop_reason=result.stop_reason.value,
                error=None,
                metadata={"elapsed_s": result.elapsed_s},
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                steps=0,
                stop_reason="error",
                error=str(e),
            )


# Convenience functions
def create_agent(
    provider: str = "mock",
    model: str = "mock-model",
    **kwargs: Any,
) -> AnvilAPI:
    """Create a configured agent."""
    config = AgentConfig(
        provider=provider,
        model=model,
        **kwargs,
    )
    return AnvilAPI(config)


def run_goal(
    goal: str,
    provider: str = "mock",
    model: str = "mock-model",
    max_steps: int = 20,
    **kwargs: Any,
) -> AgentResult:
    """Quickly run a goal with minimal setup."""
    agent = create_agent(
        provider=provider,
        model=model,
        max_steps=max_steps,
        **kwargs,
    )
    return agent.run(goal)


# Example usage:
"""
from anvil.api import run_goal

# Simple usage
result = run_goal("Hello, world!")
print(result.success, result.output)

# Custom provider
def my_provider(prompt: str) -> str:
    # Call your LLM here
    return '{"answer": "response", "done": true}'

agent = create_agent()
agent.set_provider(my_provider)
result = agent.run("Analyze this data")
"""
