"""
Layered configuration system for Anvil

Implements Claude Code's merge-on-read pattern with multiple config sources:
1. Built-in defaults (lowest precedence)
2. User config (~/.anvil/config.yaml)
3. Project config (.anvil.yaml in workspace root)
4. Local config (.anvil.local.yaml, typically gitignored)
5. Environment variables (ANVIL_*)
6. CLI arguments (highest precedence)

Each layer is optional. Later layers override earlier ones via deep merge.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

PathLike = Union[str, Path]


# ============== Built-in Defaults ==============

BUILTIN_DEFAULTS: Dict[str, Any] = {
    'model': 'gpt-4o-mini',
    'provider': 'openai_compatible',
    'temperature': 0.0,
    'max_steps': 50,
    'permission_mode': 'balanced',
    'history_window': 10,
    'max_tokens': 4096,
}


# ============== Config Layer ==============

@dataclass(frozen=True)
class ConfigLayer:
    """A single configuration layer with source metadata."""
    name: str
    source: str  # File path or 'builtin'/'env'/'cli'
    values: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


# ============== Deep Merge ==============

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dicts. Override values take precedence.

    For nested dicts, merges recursively. For lists and scalars, override wins.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ============== Config Loaders ==============

def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    if yaml is None:
        raise ModuleNotFoundError('missing optional dependency: pyyaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_env_vars(prefix: str = 'ANVIL_') -> Dict[str, Any]:
    """Load environment variables with given prefix.

    Maps ANVIL_MODEL -> model, ANVIL_TEMPERATURE -> temperature, etc.
    Numeric values are converted to int/float.
    """
    config: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            config[config_key] = _coerce_env_value(value)
    return config


def _coerce_env_value(value: str) -> Any:
    """Coerce string env value to appropriate Python type."""
    # Boolean
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # String
    return value


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load config file by extension."""
    suffix = path.suffix.lower()
    if suffix in ('.yaml', '.yml'):
        return load_yaml_file(path)
    elif suffix == '.json':
        return load_json_file(path)
    else:
        raise ValueError(f'Unsupported config format: {suffix}')


# ============== Layered Config ==============

class LayeredConfig:
    """Merge-on-read configuration with multiple layers.

    Layers are ordered by precedence (lowest first):
    1. builtin: Built-in defaults
    2. user: User config (~/.anvil/config.yaml)
    3. project: Project config (.anvil.yaml in workspace)
    4. local: Local config (.anvil.local.yaml, gitignored)
    5. env: Environment variables (ANVIL_*)
    6. cli: CLI arguments

    Thread-safe: all reads and writes are protected by a lock.
    """

    def __init__(self) -> None:
        self._layers: Dict[str, ConfigLayer] = {}
        self._merged_cache: Dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._dirty = True

    def add_layer(self, name: str, source: str, values: Dict[str, Any]) -> None:
        """Add or replace a configuration layer."""
        with self._lock:
            self._layers[name] = ConfigLayer(name=name, source=source, values=values)
            self._merged_cache = None
            self._dirty = True

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key path.

        Example: config.get('model.temperature')
        """
        merged = self._get_merged()
        parts = key.split('.')
        current = merged
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return default
            else:
                return default
        return current

    def get_flat(self, key: str, default: Any = None) -> Any:
        """Get a config value by flat key (no dot notation)."""
        merged = self._get_merged()
        return merged.get(key, default)

    def set_in_layer(self, layer_name: str, key: str, value: Any) -> None:
        """Set a value in a specific layer (write-through)."""
        with self._lock:
            if layer_name not in self._layers:
                raise ValueError(f'Unknown layer: {layer_name}')
            layer = self._layers[layer_name]
            new_values = {**layer.values, key: value}
            self._layers[layer_name] = ConfigLayer(
                name=layer.name,
                source=layer.source,
                values=new_values,
            )
            self._merged_cache = None
            self._dirty = True

    def to_dict(self) -> Dict[str, Any]:
        """Return merged config as dict."""
        return self._get_merged().copy()

    def layers(self) -> Tuple[ConfigLayer, ...]:
        """Return all layers in order."""
        return tuple(self._layers.values())

    def _get_merged(self) -> Dict[str, Any]:
        """Get merged config, using cache if available."""
        with self._lock:
            if self._merged_cache is not None:
                return self._merged_cache
            merged: Dict[str, Any] = {}
            for layer in self._layers.values():
                merged = deep_merge(merged, layer.values)
            self._merged_cache = merged
            self._dirty = False
            return merged


# ============== Config Discovery ==============

# Standard config file names (in order of precedence within a layer)
PROJECT_CONFIG_NAMES = ('.anvil.yaml', '.anvil.yml', '.anvil.json', 'anvil.yaml', 'anvil.yml', 'anvil.json')
LOCAL_CONFIG_NAMES = ('.anvil.local.yaml', '.anvil.local.yml', '.anvil.local.json')
USER_CONFIG_DIR = Path.home() / '.anvil'
USER_CONFIG_NAMES = ('config.yaml', 'config.yml', 'config.json')


def find_config_file(candidates: Sequence[str], search_dir: Path | None = None) -> Path | None:
    """Find first existing config file from candidates."""
    if search_dir is not None:
        for name in candidates:
            path = search_dir / name
            if path.exists():
                return path
    else:
        for name in candidates:
            path = Path(name)
            if path.exists():
                return path
    return None


def find_user_config() -> Path | None:
    """Find user-level config (~/.anvil/config.yaml)."""
    if USER_CONFIG_DIR.exists():
        for name in USER_CONFIG_NAMES:
            path = USER_CONFIG_DIR / name
            if path.exists():
                return path
    return None


def find_project_config(workspace_root: Path) -> Path | None:
    """Find project-level config in workspace root."""
    return find_config_file(PROJECT_CONFIG_NAMES, workspace_root)


def find_local_config(workspace_root: Path) -> Path | None:
    """Find local config (gitignored) in workspace root."""
    return find_config_file(LOCAL_CONFIG_NAMES, workspace_root)


# ============== Factory ==============

def build_layered_config(
    workspace_root: Path | None = None,
    cli_args: Dict[str, Any] | None = None,
    env_prefix: str = 'ANVIL_',
) -> LayeredConfig:
    """Build a LayeredConfig with all discovered layers.

    Args:
        workspace_root: Project workspace root for project/local configs
        cli_args: CLI arguments (highest precedence)
        env_prefix: Prefix for environment variables

    Returns:
        Fully populated LayeredConfig
    """
    config = LayeredConfig()

    # Layer 1: Built-in defaults
    config.add_layer('builtin', 'builtin', BUILTIN_DEFAULTS.copy())

    # Layer 2: User config (~/.anvil/)
    user_path = find_user_config()
    if user_path:
        try:
            user_values = load_config_file(user_path)
            config.add_layer('user', str(user_path), user_values)
        except Exception:
            pass  # Silently skip invalid user config

    # Layer 3: Project config
    if workspace_root:
        project_path = find_project_config(workspace_root)
        if project_path:
            try:
                project_values = load_config_file(project_path)
                config.add_layer('project', str(project_path), project_values)
            except Exception:
                pass

    # Layer 4: Local config (gitignored)
    if workspace_root:
        local_path = find_local_config(workspace_root)
        if local_path:
            try:
                local_values = load_config_file(local_path)
                config.add_layer('local', str(local_path), local_values)
            except Exception:
                pass

    # Layer 5: Environment variables
    env_values = load_env_vars(env_prefix)
    if env_values:
        config.add_layer('env', 'environment', env_values)

    # Layer 6: CLI arguments
    if cli_args:
        # Filter out None values
        filtered = {k: v for k, v in cli_args.items() if v is not None}
        if filtered:
            config.add_layer('cli', 'command-line', filtered)

    return config


# ============== Backward Compatibility ==============

def load_config(config_path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Load configuration from a single file (backward compatible)."""
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')
        return load_config_file(path)
    return {}


def merge_config(args_config: Dict[str, Any], config_file: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI args with config file (backward compatible)."""
    return deep_merge(config_file, args_config)
