from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from anvil.layered_config import (
    BUILTIN_DEFAULTS,
    LayeredConfig,
    build_layered_config,
    deep_merge,
    find_config_file,
    find_local_config,
    find_project_config,
    find_user_config,
    load_env_vars,
)


class DeepMergeTests(unittest.TestCase):
    def test_should_merge_flat_dicts(self) -> None:
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        result = deep_merge(base, override)
        self.assertEqual(result, {'a': 1, 'b': 3, 'c': 4})

    def test_should_deep_merge_nested_dicts(self) -> None:
        base = {'a': {'x': 1, 'y': 2}, 'b': 3}
        override = {'a': {'y': 20, 'z': 30}}
        result = deep_merge(base, override)
        self.assertEqual(result, {'a': {'x': 1, 'y': 20, 'z': 30}, 'b': 3})

    def test_should_override_lists_not_merge(self) -> None:
        base = {'a': [1, 2, 3]}
        override = {'a': [4, 5]}
        result = deep_merge(base, override)
        self.assertEqual(result, {'a': [4, 5]})

    def test_should_not_mutate_inputs(self) -> None:
        base = {'a': {'x': 1}}
        override = {'a': {'y': 2}}
        deep_merge(base, override)
        self.assertEqual(base, {'a': {'x': 1}})
        self.assertEqual(override, {'a': {'y': 2}})


class LayeredConfigTests(unittest.TestCase):
    def test_should_create_empty_config(self) -> None:
        config = LayeredConfig()
        self.assertEqual(config.to_dict(), {})

    def test_should_add_and_read_layer(self) -> None:
        config = LayeredConfig()
        config.add_layer('test', 'test.yaml', {'model': 'gpt-4', 'temperature': 0.5})
        self.assertEqual(config.get_flat('model'), 'gpt-4')
        self.assertEqual(config.get_flat('temperature'), 0.5)

    def test_should_merge_layers_by_precedence(self) -> None:
        config = LayeredConfig()
        config.add_layer('defaults', 'builtin', {'model': 'gpt-4', 'temperature': 0.0})
        config.add_layer('user', 'user.yaml', {'model': 'claude-3'})
        self.assertEqual(config.get_flat('model'), 'claude-3')
        self.assertEqual(config.get_flat('temperature'), 0.0)

    def test_should_support_dot_notation(self) -> None:
        config = LayeredConfig()
        config.add_layer('test', 'test.yaml', {'model': {'temperature': 0.5}})
        self.assertEqual(config.get('model.temperature'), 0.5)

    def test_should_return_default_for_missing_key(self) -> None:
        config = LayeredConfig()
        config.add_layer('test', 'test.yaml', {'model': 'gpt-4'})
        self.assertEqual(config.get('missing', 'default'), 'default')
        self.assertIsNone(config.get('missing'))

    def test_should_invalidate_cache_on_add_layer(self) -> None:
        config = LayeredConfig()
        config.add_layer('first', 'first.yaml', {'a': 1})
        self.assertEqual(config.get_flat('a'), 1)
        config.add_layer('second', 'second.yaml', {'a': 2})
        self.assertEqual(config.get_flat('a'), 2)

    def test_should_set_value_in_layer(self) -> None:
        config = LayeredConfig()
        config.add_layer('test', 'test.yaml', {'model': 'gpt-4'})
        config.set_in_layer('test', 'model', 'claude-3')
        self.assertEqual(config.get_flat('model'), 'claude-3')

    def test_should_raise_on_set_in_unknown_layer(self) -> None:
        config = LayeredConfig()
        with self.assertRaises(ValueError):
            config.set_in_layer('unknown', 'key', 'value')

    def test_should_return_all_layers(self) -> None:
        config = LayeredConfig()
        config.add_layer('first', 'first.yaml', {'a': 1})
        config.add_layer('second', 'second.yaml', {'b': 2})
        layers = config.layers()
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0].name, 'first')
        self.assertEqual(layers[1].name, 'second')

    def test_should_be_thread_safe(self) -> None:
        import threading
        config = LayeredConfig()
        config.add_layer('test', 'test.yaml', {'counter': 0})
        errors = []

        def increment():
            try:
                for _ in range(100):
                    current = config.get_flat('counter', 0)
                    config.set_in_layer('test', 'counter', current + 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


class LoadEnvVarsTests(unittest.TestCase):
    def test_should_load_prefixed_env_vars(self) -> None:
        with patch.dict(os.environ, {'ANVIL_MODEL': 'gpt-4', 'ANVIL_TEMPERATURE': '0.5'}):
            result = load_env_vars('ANVIL_')
            self.assertEqual(result['model'], 'gpt-4')
            self.assertEqual(result['temperature'], 0.5)

    def test_should_coerce_boolean_values(self) -> None:
        with patch.dict(os.environ, {'ANVIL_DEBUG': 'true', 'ANVIL_VERBOSE': 'no'}):
            result = load_env_vars('ANVIL_')
            self.assertTrue(result['debug'])
            self.assertFalse(result['verbose'])

    def test_should_coerce_integer_values(self) -> None:
        with patch.dict(os.environ, {'ANVIL_MAX_STEPS': '50'}):
            result = load_env_vars('ANVIL_')
            self.assertEqual(result['max_steps'], 50)

    def test_should_ignore_non_prefixed_vars(self) -> None:
        with patch.dict(os.environ, {'OTHER_VAR': 'value', 'ANVIL_MODEL': 'gpt-4'}):
            result = load_env_vars('ANVIL_')
            self.assertNotIn('other_var', result)
            self.assertEqual(result['model'], 'gpt-4')


class ConfigDiscoveryTests(unittest.TestCase):
    def test_should_find_config_in_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / '.anvil.yaml'
            config_path.write_text('model: gpt-4')
            result = find_config_file(('.anvil.yaml',), Path(tmpdir))
            self.assertEqual(result, config_path)

    def test_should_return_none_if_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_config_file(('.anvil.yaml',), Path(tmpdir))
            self.assertIsNone(result)

    def test_should_find_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / '.anvil.yaml'
            config_path.write_text('model: gpt-4')
            result = find_project_config(Path(tmpdir))
            self.assertEqual(result, config_path)

    def test_should_find_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / '.anvil.local.yaml'
            config_path.write_text('api_key: secret')
            result = find_local_config(Path(tmpdir))
            self.assertEqual(result, config_path)


class BuildLayeredConfigTests(unittest.TestCase):
    def test_should_build_with_defaults_only(self) -> None:
        config = build_layered_config()
        self.assertEqual(config.get_flat('model'), BUILTIN_DEFAULTS['model'])

    def test_should_include_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / '.anvil.yaml'
            config_path.write_text(json.dumps({'model': 'claude-3'}))
            config = build_layered_config(workspace_root=Path(tmpdir))
            self.assertEqual(config.get_flat('model'), 'claude-3')

    def test_should_include_local_over_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / '.anvil.yaml'
            project_path.write_text(json.dumps({'model': 'gpt-4'}))
            local_path = Path(tmpdir) / '.anvil.local.yaml'
            local_path.write_text(json.dumps({'model': 'claude-3'}))
            config = build_layered_config(workspace_root=Path(tmpdir))
            self.assertEqual(config.get_flat('model'), 'claude-3')

    def test_should_include_env_over_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / '.anvil.yaml'
            config_path.write_text(json.dumps({'model': 'gpt-4'}))
            with patch.dict(os.environ, {'ANVIL_MODEL': 'claude-3'}):
                config = build_layered_config(workspace_root=Path(tmpdir))
                self.assertEqual(config.get_flat('model'), 'claude-3')

    def test_should_include_cli_over_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / '.anvil.yaml'
            config_path.write_text(json.dumps({'model': 'gpt-4'}))
            with patch.dict(os.environ, {'ANVIL_MODEL': 'claude-3'}):
                config = build_layered_config(
                    workspace_root=Path(tmpdir),
                    cli_args={'model': 'gemini-pro'},
                )
                self.assertEqual(config.get_flat('model'), 'gemini-pro')

    def test_should_filter_none_cli_args(self) -> None:
        config = build_layered_config(cli_args={'model': 'gpt-4', 'temperature': None})
        self.assertEqual(config.get_flat('model'), 'gpt-4')
        # None values should be filtered out
        self.assertEqual(config.get_flat('temperature'), BUILTIN_DEFAULTS['temperature'])


if __name__ == '__main__':
    unittest.main()
