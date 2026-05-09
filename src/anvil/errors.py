"""
Error handling and validation utilities for Anvil

Provides consistent error handling and input validation across the project.
Implements a structured error hierarchy inspired by Claude Code's error system.
"""

from __future__ import annotations

from typing import Any


# ============== Error Hierarchy ==============

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


class AbortError(AnvilError):
    """Operation was cancelled or aborted."""
    code = "ABORT_ERROR"


class ShellError(AnvilError):
    """Shell command execution errors with stdout/stderr/exit code."""
    code = "SHELL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        stdout: str = '',
        stderr: str = '',
        exit_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class ToolValidationError(AnvilError):
    """Tool input validation errors with structured field info."""
    code = "TOOL_VALIDATION_ERROR"

    def __init__(
        self,
        tool_name: str,
        *,
        missing: tuple[str, ...] = (),
        unexpected: tuple[str, ...] = (),
        type_mismatches: tuple[str, ...] = (),
        details: dict[str, Any] | None = None,
    ):
        self.tool_name = tool_name
        self.missing = missing
        self.unexpected = unexpected
        self.type_mismatches = type_mismatches
        issues = []
        if missing:
            issues.extend(f'Missing required parameter: `{p}`' for p in missing)
        if unexpected:
            issues.extend(f'Unexpected parameter: `{p}`' for p in unexpected)
        if type_mismatches:
            issues.extend(type_mismatches)
        message = f'{tool_name} validation failed:\n' + '\n'.join(f'  - {i}' for i in issues)
        super().__init__(message, details)


# ============== Error Classification ==============

def is_abort_error(error: Exception) -> bool:
    """Check if an error is an abort/cancellation signal."""
    if isinstance(error, AbortError):
        return True
    # Handle KeyboardInterrupt
    if isinstance(error, KeyboardInterrupt):
        return True
    return False


def format_tool_error(tool_name: str, error: Exception) -> str:
    """Format a tool error into a structured message for model self-correction.

    Returns a message that tells the model exactly what went wrong,
    so it can adjust its approach.
    """
    if isinstance(error, ToolValidationError):
        return str(error)
    if isinstance(error, ShellError):
        parts = [f'{tool_name} command failed (exit code {error.exit_code})']
        if error.stderr:
            parts.append(f'stderr: {error.stderr[:500]}')
        if error.stdout:
            parts.append(f'stdout: {error.stdout[:500]}')
        return '\n'.join(parts)
    if isinstance(error, ValidationError):
        return f'{tool_name}: {error.message}'
    return f'{tool_name} error: {str(error)}'


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
