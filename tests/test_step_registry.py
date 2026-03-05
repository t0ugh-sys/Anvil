from __future__ import annotations

import argparse
import unittest

import _bootstrap  # noqa: F401

from loop_agent.steps.registry import build_default_registry


class StepRegistryTests(unittest.TestCase):
    def test_should_contain_builtin_strategies(self) -> None:
        registry = build_default_registry()
        self.assertEqual(registry.names(), ['demo', 'json_stub'])

    def test_should_create_demo_bundle(self) -> None:
        registry = build_default_registry()
        args = argparse.Namespace(history_window=3)
        step, state = registry.create('demo', args)
        self.assertIsNotNone(step)
        self.assertEqual(type(state).__name__, 'DemoState')

    def test_should_raise_for_unknown_strategy(self) -> None:
        registry = build_default_registry()
        args = argparse.Namespace(history_window=3)
        with self.assertRaises(ValueError):
            registry.create('unknown', args)


if __name__ == '__main__':
    unittest.main()

