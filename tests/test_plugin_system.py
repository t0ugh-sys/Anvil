"""Tests for the plugin system — entry-point discovery and SkillLoader wiring."""
from __future__ import annotations

import sys
import types
import unittest
from typing import Any, Callable
from unittest.mock import MagicMock, patch

from anvil.infra.skills import (
    Skill,
    SkillBase,
    SkillLoader,
    discover_plugins,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _GoodPlugin:
    """Minimal SkillBase-compliant plugin class."""
    name = "good_plugin"
    description = "A test plugin"

    def get_tools(self) -> dict[str, Callable]:
        return {}

    def on_activate(self, ctx: dict[str, Any]) -> None:
        pass


class _BadPlugin:
    """Missing on_activate — NOT SkillBase-compliant."""
    name = "bad_plugin"
    description = "Fails SkillBase check"

    def get_tools(self) -> dict[str, Callable]:
        return {}


def _make_ep(name: str, cls: type | None, *, raise_on_load: bool = False):
    """Return a mock entry-point object."""
    ep = MagicMock()
    ep.name = name
    if raise_on_load:
        ep.load.side_effect = ImportError("simulated load failure")
    else:
        ep.load.return_value = cls
    return ep


# ---------------------------------------------------------------------------
# discover_plugins
# ---------------------------------------------------------------------------

class TestDiscoverPlugins(unittest.TestCase):

    def test_returns_empty_when_no_entry_points(self):
        with patch("importlib.metadata.entry_points", return_value=[]):
            result = discover_plugins()
        self.assertEqual(result, {})

    def test_returns_compliant_plugin(self):
        eps = [_make_ep("good_plugin", _GoodPlugin)]
        with patch("importlib.metadata.entry_points", return_value=eps):
            result = discover_plugins()
        self.assertIn("good_plugin", result)
        self.assertIs(result["good_plugin"], _GoodPlugin)

    def test_skips_non_compliant_class(self):
        eps = [_make_ep("bad_plugin", _BadPlugin)]
        with patch("importlib.metadata.entry_points", return_value=eps):
            result = discover_plugins()
        self.assertNotIn("bad_plugin", result)

    def test_skips_unloadable_entry_point(self):
        eps = [_make_ep("broken", None, raise_on_load=True)]
        with patch("importlib.metadata.entry_points", return_value=eps):
            result = discover_plugins()
        self.assertEqual(result, {})

    def test_skips_entry_point_that_raises_on_instantiation(self):
        class _Crasher:
            name = "crasher"
            description = "raises on init"

            def __init__(self):
                raise RuntimeError("boom")

            def get_tools(self):
                return {}

            def on_activate(self, ctx):
                pass

        eps = [_make_ep("crasher", _Crasher)]
        with patch("importlib.metadata.entry_points", return_value=eps):
            result = discover_plugins()
        self.assertEqual(result, {})

    def test_returns_multiple_compliant_plugins(self):
        class _PluginB(_GoodPlugin):
            name = "plugin_b"

        eps = [_make_ep("good_plugin", _GoodPlugin), _make_ep("plugin_b", _PluginB)]
        with patch("importlib.metadata.entry_points", return_value=eps):
            result = discover_plugins()
        self.assertIn("good_plugin", result)
        self.assertIn("plugin_b", result)


# ---------------------------------------------------------------------------
# SkillLoader._load_external — entry-point path
# ---------------------------------------------------------------------------

class TestSkillLoaderEntryPoints(unittest.TestCase):

    def setUp(self):
        self.loader = SkillLoader()

    def test_loads_skill_via_entry_point(self):
        eps = {"good_plugin": _make_ep("good_plugin", _GoodPlugin)}
        with patch(
            "importlib.metadata.entry_points",
            return_value=list(eps.values()),
        ):
            loaded = self.loader._load_external("good_plugin")
        self.assertTrue(loaded)
        self.assertIn("good_plugin", self.loader.list_loaded())

    def test_entry_point_takes_priority_over_namespace(self):
        """Entry-point path must be tried before anvil_skills.*."""
        eps = [_make_ep("my_skill", _GoodPlugin)]
        call_order: list[str] = []

        orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def tracking_import(name, *args, **kwargs):
            if 'anvil_skills' in name:
                call_order.append('namespace')
            return orig_import(name, *args, **kwargs)

        with patch("importlib.metadata.entry_points", return_value=eps) as mock_ep:
            loaded = self.loader._load_external("my_skill")

        mock_ep.assert_called_once()
        self.assertTrue(loaded)
        # namespace import must not have been reached
        self.assertNotIn('namespace', call_order)

    def test_falls_back_to_namespace_when_no_entry_point(self):
        """When entry-point lookup misses, anvil_skills.* is tried."""
        fake_module = types.ModuleType("anvil_skills.fallback")
        fake_module.Skill = type("Skill", (Skill,), {"name": "fallback", "description": "fb"})

        with patch("importlib.metadata.entry_points", return_value=[]):
            with patch.dict(sys.modules, {"anvil_skills.fallback": fake_module}):
                loaded = self.loader._load_external("fallback")
        self.assertTrue(loaded)

    def test_returns_false_when_both_paths_miss(self):
        with patch("importlib.metadata.entry_points", return_value=[]):
            result = self.loader._load_external("nonexistent_xyz")
        self.assertFalse(result)

    def test_rejects_invalid_name(self):
        with self.assertRaises(ValueError):
            self.loader._load_external("../../etc/passwd")

    def test_instance_is_stored_not_class(self):
        eps = [_make_ep("good_plugin", _GoodPlugin)]
        with patch("importlib.metadata.entry_points", return_value=eps):
            self.loader._load_external("good_plugin")
        skill = self.loader._loaded_skills.get("good_plugin")
        self.assertIsInstance(skill, _GoodPlugin)

    def test_loads_instance_when_entry_point_returns_instance(self):
        """Entry point may return a pre-built instance rather than a class."""
        instance = _GoodPlugin()
        ep = _make_ep("good_plugin", None)
        ep.load.return_value = instance  # not a type — already instantiated

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            loaded = self.loader._load_external("good_plugin")
        self.assertTrue(loaded)
        self.assertIs(self.loader._loaded_skills["good_plugin"], instance)


# ---------------------------------------------------------------------------
# SkillBase protocol structural check
# ---------------------------------------------------------------------------

class TestSkillBaseProtocol(unittest.TestCase):

    def test_compliant_class_passes_isinstance(self):
        self.assertIsInstance(_GoodPlugin(), SkillBase)

    def test_non_compliant_class_fails_isinstance(self):
        self.assertNotIsInstance(_BadPlugin(), SkillBase)

    def test_builtin_skill_subclass_does_not_satisfy_protocol(self):
        # Skill base class lacks on_activate — should not satisfy SkillBase
        plain = Skill()
        self.assertNotIsInstance(plain, SkillBase)


if __name__ == "__main__":
    unittest.main()
