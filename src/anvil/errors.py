"""
Error handling and validation utilities for Anvil

Provides consistent error handling and input validation across the project.
"""

from __future__ import annotations

from typing import Any


class AnvilError(Exception):
    """Base exception for Anvil."""
    code: str = "ANVIL_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(AnvilError):
    """Input validation errors."""
    code = "VALIDATION_ERROR"


def format_error(error: Exception) -> dict[str, Any]:
    """Format an exception into a dictionary for JSON output."""
    if isinstance(error, AnvilError):
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

    return strategy.strip()
