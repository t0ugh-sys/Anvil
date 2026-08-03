"""Schema validation for Anvil's layered configuration.

Layered config (see :mod:`anvil.config.layered`) merges built-in defaults
with user/project/local YAML/JSON files, environment variables, and CLI
args. None of those layers are checked against a schema today, so a typo
in a hand-edited config file (e.g. ``tempreture: 0.5``) is silently
dropped by the merge instead of surfacing an error -- the mistake only
shows up later as a runtime ``KeyError`` or a value that quietly falls
back to its default.

This module defines the known top-level fields and their constraints, and
a ``validate_config`` function that returns a list of human-readable
errors for a merged config dict (empty list = valid). Callers decide what
to do with the errors; :func:`validate_or_exit` is a ready-made fail-fast
helper that prints them and exits with status 1.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    'ConfigFieldError',
    'KNOWN_FIELDS',
    'validate_config',
    'validate_or_exit',
]

_VALID_PROVIDERS = frozenset({'mock', 'openai_compatible', 'anthropic', 'gemini'})
_VALID_PERMISSION_MODES = frozenset({'strict', 'balanced', 'unsafe'})

# alias kept short for readability in the validator bodies below
ConfigFieldError = Optional[str]


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _check_model(value: Any) -> ConfigFieldError:
    if not isinstance(value, str) or not value.strip():
        return f"'model' must be a non-empty string, got {value!r}"
    return None


def _check_provider(value: Any) -> ConfigFieldError:
    if value not in _VALID_PROVIDERS:
        return f"'provider' must be one of {sorted(_VALID_PROVIDERS)}, got {value!r}"
    return None


def _check_temperature(value: Any) -> ConfigFieldError:
    if _is_bool(value) or not isinstance(value, (int, float)):
        return f"'temperature' must be a number, got {type(value).__name__}"
    if value < 0 or value > 2:
        return f"'temperature' must be between 0 and 2, got {value}"
    return None


def _check_max_steps(value: Any) -> ConfigFieldError:
    if _is_bool(value) or not isinstance(value, int):
        return f"'max_steps' must be an integer, got {type(value).__name__}"
    if value < 1 or value > 1000:
        return f"'max_steps' must be between 1 and 1000, got {value}"
    return None


def _check_permission_mode(value: Any) -> ConfigFieldError:
    if value not in _VALID_PERMISSION_MODES:
        return f"'permission_mode' must be one of {sorted(_VALID_PERMISSION_MODES)}, got {value!r}"
    return None


def _check_history_window(value: Any) -> ConfigFieldError:
    if _is_bool(value) or not isinstance(value, int):
        return f"'history_window' must be an integer, got {type(value).__name__}"
    if value < 0:
        return f"'history_window' must be >= 0, got {value}"
    return None


def _check_max_tokens(value: Any) -> ConfigFieldError:
    if _is_bool(value) or not isinstance(value, int):
        return f"'max_tokens' must be an integer, got {type(value).__name__}"
    if value < 1 or value > 200_000:
        return f"'max_tokens' must be between 1 and 200000, got {value}"
    return None


# Known top-level config fields, keyed by name -> validator. These mirror
# anvil.config.layered.BUILTIN_DEFAULTS -- the one place that currently
# enumerates every field the layered config system understands.
KNOWN_FIELDS: Dict[str, Callable[[Any], ConfigFieldError]] = {
    'model': _check_model,
    'provider': _check_provider,
    'temperature': _check_temperature,
    'max_steps': _check_max_steps,
    'permission_mode': _check_permission_mode,
    'history_window': _check_history_window,
    'max_tokens': _check_max_tokens,
}


def validate_config(values: Dict[str, Any]) -> List[str]:
    """Validate a merged layered-config dict.

    Returns a list of human-readable error strings; an empty list means
    the config is valid. Reports two kinds of problems: unrecognized
    field names (most often a typo in a user config file) and known
    fields whose value fails its type/range check.
    """
    errors: List[str] = []
    for key, value in values.items():
        validator = KNOWN_FIELDS.get(key)
        if validator is None:
            errors.append(f"unknown config field: '{key}' (check for a typo)")
            continue
        message = validator(value)
        if message:
            errors.append(message)
    return errors


def validate_or_exit(values: Dict[str, Any], *, stream=None) -> None:
    """Validate ``values`` and exit the process with status 1 if invalid.

    Intended for use at startup, right after building a merged config, so
    a bad config fails fast with a clear list instead of crashing deep
    inside a run with a ``KeyError``.
    """
    errors = validate_config(values)
    if not errors:
        return
    out = stream if stream is not None else sys.stderr
    print('Invalid configuration:', file=out)
    for error in errors:
        print(f'  - {error}', file=out)
    sys.exit(1)
