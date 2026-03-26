from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from loop_agent.ops.doctor import run_provider_doctor


class DoctorTests(unittest.TestCase):
    def test_should_return_structured_report(self) -> None:
        payload = run_provider_doctor(
            base_url='https://example.com/v1',
            model='gpt-5.3-codex',
            wire_api='responses',
            timeout_s=1.0,
            api_key_present=False,
            extra_headers=[],
        )
        self.assertIn('dns', payload)
        self.assertIn('tcp_443', payload)
        self.assertIn('probe_base', payload)
        self.assertIn('probe_endpoint', payload)
        self.assertEqual(payload['wire_api'], 'responses')


if __name__ == '__main__':
    unittest.main()
