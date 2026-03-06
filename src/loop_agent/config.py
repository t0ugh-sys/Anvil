"""
Configuration system for LoopAgent

Supports YAML, JSON, and .env configuration files.

Usage:
    # Create config.yaml
    # Run with config: loopagent --config config.yaml
    
    # Or use default locations:
    # - ./loopagent.yaml
    # - ./loopagent.json
    # - ~/.loopagent/config.yaml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


PathLike = Union[str, Path]


def load_yaml_config(path: PathLike) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    YAML support is optional. Install with: `pip install pyyaml`.
    """

    if yaml is None:
        raise ModuleNotFoundError('missing optional dependency: pyyaml')

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_json_config(path: PathLike) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_env_config(path: PathLike) -> Dict[str, Any]:
    """Load configuration from .env file."""
    config = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Map to config keys
                if key.startswith('LOOPAGENT_'):
                    config[key[10:].lower()] = value
                elif key.startswith('OPENAI_'):
                    config[key.lower()] = value
                elif key.startswith('ANTHROPIC_'):
                    config[key.lower()] = value
                elif key.startswith('GEMINI_'):
                    config[key.lower()] = value
    return config


# Default config locations
DEFAULT_CONFIG_LOCATIONS = [
    './loopagent.yaml',
    './loopagent.yml',
    './loopagent.json',
    './.loopagent.yaml',
    './.loopagent.yml',
    './.loopagent.json',
]


def find_default_config() -> Optional[Path]:
    """Find default config file."""
    for loc in DEFAULT_CONFIG_LOCATIONS:
        path = Path(loc)
        if path.exists():
            return path
    # Check home directory
    home_config = Path.home() / '.loopagent' / 'config.yaml'
    if home_config.exists():
        return home_config
    return None


def load_config(config_path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Load configuration from file or find default."""
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')
    else:
        path = find_default_config()
        if not path:
            return {}
    
    suffix = path.suffix.lower()
    if suffix in ('.yaml', '.yml'):
        return load_yaml_config(path)
    elif suffix == '.json':
        return load_json_config(path)
    elif suffix in ('.env',):
        return load_env_config(path)
    else:
        raise ValueError(f'Unsupported config format: {suffix}')


def merge_config(args_config: Dict[str, Any], config_file: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI args with config file. CLI args take precedence."""
    merged = config_file.copy()
    merged.update(args_config)
    return merged


# Config schema for documentation
CONFIG_SCHEMA = {
    'provider': 'LLM provider (mock, openai_compatible, anthropic, gemini)',
    'model': 'Model name',
    'base_url': 'API base URL for openai_compatible provider',
    'api_key_env': 'Environment variable name for API key',
    'temperature': 'LLM temperature (0.0-2.0)',
    'max_steps': 'Maximum steps for agent loop',
    'timeout_s': 'Timeout in seconds',
    'strategy': 'Agent strategy (demo, json_stub, json_llm)',
    'history_window': 'Number of historical steps to include',
    'memory_dir': 'Directory for memory storage',
    'skills': 'Comma-separated list of skills to load',
    'provider_timeout_s': 'Provider API timeout',
    'max_retries': 'Maximum retries for API calls',
    'retry_backoff_s': 'Backoff time for retries',
}
