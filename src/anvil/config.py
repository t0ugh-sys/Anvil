"""
Configuration system for Anvil

Supports YAML, JSON, and .env configuration files.

Usage:
    # Create config.yaml
    # Run with config: anvil --config config.yaml
    
    # Or use default locations:
    # - ./anvil.yaml
    # - ./anvil.json
    # - ~/.anvil/config.yaml
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
                if key.startswith('ANVIL_'):
                    config[key[6:].lower()] = value
                elif key.startswith('OPENAI_'):
                    config[key.lower()] = value
                elif key.startswith('ANTHROPIC_'):
                    config[key.lower()] = value
                elif key.startswith('GEMINI_'):
                    config[key.lower()] = value
    return config


# Default config locations
DEFAULT_CONFIG_LOCATIONS = [
    './anvil.yaml',
    './anvil.yml',
    './anvil.json',
    './.anvil.yaml',
    './.anvil.yml',
    './.anvil.json',
]


def find_default_config() -> Optional[Path]:
    """Find default config file."""
    for loc in DEFAULT_CONFIG_LOCATIONS:
        path = Path(loc)
        if path.exists():
            return path
    # Check home directory
    home_candidates = [
        Path.home() / '.anvil' / 'config.yaml',
    ]
    for home_config in home_candidates:
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
