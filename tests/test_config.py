from __future__ import annotations

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from loop_agent.config import DEFAULT_CONFIG_LOCATIONS, find_default_config, load_env_config


class ConfigTests(unittest.TestCase):
    def test_should_prefer_anvil_default_locations(self) -> None:
        self.assertEqual(DEFAULT_CONFIG_LOCATIONS[0], './anvil.yaml')
        self.assertIn('./loopagent.yaml', DEFAULT_CONFIG_LOCATIONS)

    def test_should_find_legacy_project_config_when_present(self) -> None:
        cwd = Path('tests/.tmp') / 'anvil-config-project'
        shutil.rmtree(cwd, ignore_errors=True)
        cwd.mkdir(parents=True, exist_ok=True)
        old_cwd = Path.cwd()
        try:
            os.chdir(cwd)
            legacy = Path('loopagent.yaml')
            legacy.write_text('provider: mock\n', encoding='utf-8')
            found = find_default_config()
            self.assertEqual(found, legacy)
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(cwd, ignore_errors=True)

    def test_should_find_anvil_home_config_before_legacy_home_config(self) -> None:
        fake_home = Path('tests/.tmp') / 'anvil-config-home'
        shutil.rmtree(fake_home, ignore_errors=True)
        fake_home.mkdir(parents=True, exist_ok=True)
        try:
            anvil_home = fake_home / '.anvil'
            legacy_home = fake_home / '.loopagent'
            anvil_home.mkdir(parents=True, exist_ok=True)
            legacy_home.mkdir(parents=True, exist_ok=True)
            (legacy_home / 'config.yaml').write_text('provider: mock\n', encoding='utf-8')
            (anvil_home / 'config.yaml').write_text('provider: mock\n', encoding='utf-8')
            with patch('pathlib.Path.home', return_value=fake_home):
                found = find_default_config()
            self.assertEqual(found, anvil_home / 'config.yaml')
        finally:
            shutil.rmtree(fake_home, ignore_errors=True)

    def test_should_parse_anvil_env_prefix(self) -> None:
        tmp = Path('tests/.tmp') / 'anvil-config-env'
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        env_file = tmp / 'config.env'
        try:
            env_file.write_text('ANVIL_PROVIDER=mock\nANVIL_MODEL=test-model\n', encoding='utf-8')
            payload = load_env_config(env_file)
            self.assertEqual(payload['provider'], 'mock')
            self.assertEqual(payload['model'], 'test-model')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
