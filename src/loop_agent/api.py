"""
High-level API for LoopAgent

Provides a simple programmatic interface for using LoopAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loop_agent.agent_protocol import ToolResult
from loop_agent.core.agent import LoopAgent
from loop_agent.core.types import RunResult, StopConfig
from loop_agent.coding_agent import CodingAgentState, DeciderFn
from loop_agent.steps.json_loop import JsonLoopState, make_json_decision_step

from .errors import ValidationError, validate_goal, validate_max_steps, validate_temperature


@dataclass
class AgentConfig:
    """Configuration for LoopAgent."""
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
    
    def validate(self) -> "AgentConfig":
        """Validate configuration."""
        self.provider = self.provider
        self.model = self.model
        self.temperature = validate_temperature(self.temperature)
        self.max_steps = validate_max_steps(self.max_steps)
        self.goal = validate_goal(self.goal) if hasattr(self, 'goal') else ""
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


class LoopAgentAPI:
    """High-level API for LoopAgent."""
    
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        # Using Any for flexibility - actual decider has 5 parameters
        self._invoke_fn: Any = None
    
    def set_provider(self, invoke_fn: Callable[..., str]) -> "LoopAgentAPI":
        """Set custom LLM provider function."""
        self._invoke_fn = invoke_fn
        return self
    
    def run(self, goal: str) -> AgentResult:
        """Run the agent with a goal."""
        # Validate goal
        goal = validate_goal(goal)
        
        try:
            if self._invoke_fn:
                # Use custom provider
                step = make_json_decision_step(
                    self._invoke_fn,
                    history_window=self.config.history_window
                )
            else:
                # Use mock provider
                from loop_agent.llm.providers import _mock_invoke_factory
                invoke_fn = _mock_invoke_factory(
                    model=self.config.model,
                    mode="json"
                )
                step = make_json_decision_step(
                    invoke_fn,
                    history_window=self.config.history_window
                )
            
            # Create agent
            stop = StopConfig(
                max_steps=self.config.max_steps,
                max_elapsed_s=self.config.timeout_s,
            )
            agent = LoopAgent(step=step, stop=stop)
            
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
        from loop_agent.coding_agent import run_coding_agent, build_coding_step
        from loop_agent.tools import ToolContext
        
        goal = validate_goal(goal)
        
        try:
            if not self._invoke_fn:
                raise ValueError("Coding agent requires a provider function")
            
            step = build_coding_step(
                self._invoke_fn,
                workspace_root=self.config.workspace,
            )
            
            stop = StopConfig(
                max_steps=self.config.max_steps,
                max_elapsed_s=self.config.timeout_s,
            )
            
            result = run_coding_agent(
                goal=goal,
                decider=self._invoke_fn,
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
) -> LoopAgentAPI:
    """Create a configured agent."""
    config = AgentConfig(
        provider=provider,
        model=model,
        **kwargs,
    )
    return LoopAgentAPI(config)


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
from loop_agent.api import run_goal

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
