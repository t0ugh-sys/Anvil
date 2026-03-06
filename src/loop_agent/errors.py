"""
Error handling and validation utilities for LoopAgent

Provides consistent error handling and input validation across the project.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class LoopAgentError(Exception):
    """Base exception for LoopAgent."""
    code: str = "LOOP_AGENT_ERROR"
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(LoopAgentError):
    """Configuration related errors."""
    code = "CONFIG_ERROR"


class ProviderError(LoopAgentError):
    """LLM provider related errors."""
    code = "PROVIDER_ERROR"


class ToolError(LoopAgentError):
    """Tool execution related errors."""
    code = "TOOL_ERROR"


class ValidationError(LoopAgentError):
    """Input validation errors."""
    code = "VALIDATION_ERROR"


class MemoryError(LoopAgentError):
    """Memory/storage related errors."""
    code = "MEMORY_ERROR"


class SkillError(LoopAgentError):
    """Skill loading/execution errors."""
    code = "SKILL_ERROR"


# Error codes for programmatic handling
class ErrorCode(Enum):
    """Error codes for programmatic handling."""
    
    # Configuration errors (1000-1099)
    CONFIG_MISSING = 1001
    CONFIG_INVALID = 1002
    CONFIG_NOT_FOUND = 1003
    
    # Provider errors (2000-2099)
    PROVIDER_NOT_FOUND = 2001
    PROVIDER_AUTH_FAILED = 2002
    PROVIDER_RATE_LIMITED = 2003
    PROVIDER_TIMEOUT = 2004
    PROVIDER_INVALID_RESPONSE = 2005
    
    # Tool errors (3000-3099)
    TOOL_NOT_FOUND = 3001
    TOOL_EXECUTION_FAILED = 3002
    TOOL_TIMEOUT = 3003
    TOOL_INVALID_ARGS = 3004
    
    # Validation errors (4000-4099)
    VALIDATION_FAILED = 4001
    INVALID_GOAL = 4002
    INVALID_MODEL = 4003
    INVALID_STRATEGY = 4004
    
    # Memory errors (5000-5099)
    MEMORY_NOT_FOUND = 5001
    MEMORY_CORRUPTED = 5002
    MEMORY_WRITE_FAILED = 5003
    
    # Skill errors (6000-6099)
    SKILL_NOT_FOUND = 6001
    SKILL_LOAD_FAILED = 6002
    SKILL_REGISTRATION_FAILED = 6003


def format_error(error: Exception) -> dict[str, Any]:
    """Format an exception into a dictionary for JSON output."""
    if isinstance(error, LoopAgentError):
        return {
            "error": error.message,
            "code": error.code,
            "details": error.details,
        }
    return {
        "error": str(error),
        "code": "UNKNOWN_ERROR",
        "details": {},
    }


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    if isinstance(error, ProviderError):
        code = error.details.get("status_code")
        # Retry on rate limit or temporary errors
        if code in (429, 500, 502, 503, 504):
            return True
    return False


# Validation functions
def validate_goal(goal: str) -> str:
    """Validate and sanitize goal input."""
    if not goal or not goal.strip():
        raise ValidationError("Goal cannot be empty", {"field": "goal"})
    
    goal = goal.strip()
    
    if len(goal) > 10000:
        raise ValidationError(
            "Goal too long (max 10000 characters)",
            {"field": "goal", "length": len(goal)}
        )
    
    return goal


def validate_model(model: str) -> str:
    """Validate model name."""
    if not model or not model.strip():
        raise ValidationError("Model cannot be empty", {"field": "model"})
    
    # Basic sanitization - remove potentially dangerous characters
    model = model.strip()
    if any(c in model for c in ['\n', '\r', '\0']):
        raise ValidationError(
            "Model name contains invalid characters",
            {"field": "model"}
        )
    
    return model


def validate_temperature(temperature: float) -> float:
    """Validate temperature parameter."""
    if not isinstance(temperature, (int, float)):
        raise ValidationError(
            "Temperature must be a number",
            {"field": "temperature", "value": temperature}
        )
    
    if temperature < 0 or temperature > 2:
        raise ValidationError(
            "Temperature must be between 0 and 2",
            {"field": "temperature", "value": temperature}
        )
    
    return float(temperature)


def validate_max_steps(max_steps: int) -> int:
    """Validate max_steps parameter."""
    if not isinstance(max_steps, int):
        raise ValidationError(
            "max_steps must be an integer",
            {"field": "max_steps", "value": max_steps}
        )
    
    if max_steps < 1:
        raise ValidationError(
            "max_steps must be at least 1",
            {"field": "max_steps", "value": max_steps}
        )
    
    if max_steps > 1000:
        raise ValidationError(
            "max_steps too large (max 1000)",
            {"field": "max_steps", "value": max_steps}
        )
    
    return max_steps


def validate_provider(provider: str) -> str:
    """Validate provider name."""
    valid_providers = {"mock", "openai_compatible", "anthropic", "gemini"}
    
    if provider not in valid_providers:
        raise ValidationError(
            f"Invalid provider. Must be one of: {', '.join(valid_providers)}",
            {"field": "provider", "value": provider, "valid": list(valid_providers)}
        )
    
    return provider


def validate_strategy(strategy: str) -> str:
    """Validate strategy name."""
    if not strategy or not strategy.strip():
        raise ValidationError("Strategy cannot be empty", {"field": "strategy"})
    
    # Currently valid strategies (can be extended)
    # The actual validation happens in the registry
    return strategy.strip()


def sanitize_path(path: str) -> str:
    """Sanitize a file path to prevent path traversal."""
    import os
    
    # Remove null bytes
    path = path.replace('\0', '')
    
    # Normalize path
    path = os.path.normpath(path)
    
    # Check for path traversal attempts
    if '..' in path:
        raise ValidationError(
            "Path traversal not allowed",
            {"field": "path", "value": path}
        )
    
    return path
