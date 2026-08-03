from __future__ import annotations

import io
import unittest

import _bootstrap  # noqa: F401

from anvil.config.layered import BUILTIN_DEFAULTS, build_layered_config
from anvil.config.schema import KNOWN_FIELDS, validate_config, validate_or_exit


class ValidateConfigTests(unittest.TestCase):
    def test_should_accept_builtin_defaults(self) -> None:
        self.assertEqual(validate_config(dict(BUILTIN_DEFAULTS)), [])

    def test_should_accept_empty_config(self) -> None:
        self.assertEqual(validate_config({}), [])

    def test_should_flag_unknown_field(self) -> None:
        errors = validate_config({'tempreture': 0.5})
        self.assertEqual(len(errors), 1)
        self.assertIn('tempreture', errors[0])

    def test_should_flag_invalid_provider(self) -> None:
        errors = validate_config({'provider': 'not-a-provider'})
        self.assertEqual(len(errors), 1)
        self.assertIn('provider', errors[0])

    def test_should_flag_temperature_out_of_range(self) -> None:
        errors = validate_config({'temperature': 5.0})
        self.assertEqual(len(errors), 1)
        self.assertIn('temperature', errors[0])

    def test_should_flag_non_numeric_temperature(self) -> None:
        errors = validate_config({'temperature': 'hot'})
        self.assertEqual(len(errors), 1)

    def test_should_reject_bool_for_int_field(self) -> None:
        errors = validate_config({'max_steps': True})
        self.assertEqual(len(errors), 1)
        self.assertIn('max_steps', errors[0])

    def test_should_flag_invalid_permission_mode(self) -> None:
        errors = validate_config({'permission_mode': 'yolo'})
        self.assertEqual(len(errors), 1)
        self.assertIn('permission_mode', errors[0])

    def test_should_flag_max_tokens_out_of_range(self) -> None:
        errors = validate_config({'max_tokens': 0})
        self.assertEqual(len(errors), 1)
        self.assertIn('max_tokens', errors[0])

    def test_should_collect_multiple_errors(self) -> None:
        errors = validate_config({'provider': 'bogus', 'max_steps': -1})
        self.assertEqual(len(errors), 2)

    def test_known_fields_cover_builtin_defaults(self) -> None:
        self.assertEqual(set(KNOWN_FIELDS), set(BUILTIN_DEFAULTS))


class ValidateOrExitTests(unittest.TestCase):
    def test_should_not_exit_for_valid_config(self) -> None:
        stream = io.StringIO()
        validate_or_exit(dict(BUILTIN_DEFAULTS), stream=stream)
        self.assertEqual(stream.getvalue(), '')

    def test_should_exit_for_invalid_config(self) -> None:
        stream = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            validate_or_exit({'provider': 'bogus'}, stream=stream)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn('provider', stream.getvalue())


class BuildLayeredConfigValidateTests(unittest.TestCase):
    def test_should_not_exit_when_validate_true_and_config_valid(self) -> None:
        config = build_layered_config(validate=True)
        self.assertEqual(config.get_flat('model'), BUILTIN_DEFAULTS['model'])

    def test_should_exit_when_validate_true_and_cli_args_invalid(self) -> None:
        with self.assertRaises(SystemExit):
            build_layered_config(cli_args={'provider': 'bogus'}, validate=True)


if __name__ == '__main__':
    unittest.main()
