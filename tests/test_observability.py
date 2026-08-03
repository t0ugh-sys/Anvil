"""Tests for observability — tracing and structured JSON logging."""
from __future__ import annotations

import json
import logging
import io
import unittest
from unittest.mock import MagicMock, patch

from anvil.observability.tracing import (
    NoOpSpan,
    NoOpTracer,
    get_tracer,
    trace_tool_call,
    trace_llm_invoke,
)
from anvil.observability.logging import (
    JSONFormatter,
    StructuredLogger,
    configure_json_logging,
    get_logger,
)


# ---------------------------------------------------------------------------
# NoOpSpan
# ---------------------------------------------------------------------------

class TestNoOpSpan(unittest.TestCase):

    def test_set_attribute_returns_self(self):
        span = NoOpSpan()
        ret = span.set_attribute('key', 'value')
        self.assertIs(ret, span)

    def test_record_exception_does_not_raise(self):
        span = NoOpSpan()
        span.record_exception(ValueError('boom'))

    def test_add_event_does_not_raise(self):
        span = NoOpSpan()
        span.add_event('my_event', {'k': 'v'})

    def test_context_manager_yields_self(self):
        span = NoOpSpan()
        with span as s:
            self.assertIs(s, span)

    def test_context_manager_suppresses_nothing(self):
        with self.assertRaises(RuntimeError):
            with NoOpSpan():
                raise RuntimeError('test')


# ---------------------------------------------------------------------------
# NoOpTracer
# ---------------------------------------------------------------------------

class TestNoOpTracer(unittest.TestCase):

    def test_start_as_current_span_yields_noop(self):
        tracer = NoOpTracer()
        with tracer.start_as_current_span('op') as span:
            self.assertIsInstance(span, NoOpSpan)

    def test_start_span_returns_noop(self):
        tracer = NoOpTracer()
        self.assertIsInstance(tracer.start_span('op'), NoOpSpan)

    def test_attributes_accepted(self):
        tracer = NoOpTracer()
        with tracer.start_as_current_span('op', attributes={'a': 1}) as span:
            self.assertIsNotNone(span)

    def test_exception_propagates_out(self):
        tracer = NoOpTracer()
        with self.assertRaises(ValueError):
            with tracer.start_as_current_span('op'):
                raise ValueError('propagate me')


# ---------------------------------------------------------------------------
# get_tracer
# ---------------------------------------------------------------------------

class TestGetTracer(unittest.TestCase):

    def test_returns_tracer_without_otel(self):
        # OTel may or may not be installed; either way get_tracer must return
        # something with start_as_current_span
        tracer = get_tracer('test')
        self.assertTrue(hasattr(tracer, 'start_as_current_span'))

    def test_noop_tracer_when_otel_missing(self):
        with patch.dict('sys.modules', {'opentelemetry': None, 'opentelemetry.trace': None}):
            import importlib
            import anvil.observability.tracing as mod
            importlib.reload(mod)
            tracer = mod.get_tracer('test')
            # Should be a NoOpTracer or equivalent
            with tracer.start_as_current_span('x') as span:
                span.set_attribute('k', 'v')


# ---------------------------------------------------------------------------
# trace_tool_call decorator
# ---------------------------------------------------------------------------

class TestTraceToolCall(unittest.TestCase):

    def _make_result(self, ok: bool):
        r = MagicMock()
        r.ok = ok
        return r

    def test_wraps_sync_function(self):
        @trace_tool_call
        def my_tool(ctx, args):
            return self._make_result(True)

        result = my_tool(None, {})
        self.assertTrue(result.ok)

    def test_preserves_function_name(self):
        @trace_tool_call
        def read_file(ctx, args):
            return self._make_result(True)

        self.assertEqual(read_file.__name__, 'read_file')

    def test_sync_exception_propagates(self):
        @trace_tool_call
        def broken_tool(ctx, args):
            raise ValueError('tool error')

        with self.assertRaises(ValueError):
            broken_tool(None, {})

    def test_wraps_async_function(self):
        import asyncio

        @trace_tool_call
        async def async_tool(ctx, args):
            return self._make_result(True)

        result = asyncio.run(async_tool(None, {}))
        self.assertTrue(result.ok)

    def test_async_exception_propagates(self):
        import asyncio

        @trace_tool_call
        async def broken_async(ctx, args):
            raise RuntimeError('async boom')

        with self.assertRaises(RuntimeError):
            asyncio.run(broken_async(None, {}))


# ---------------------------------------------------------------------------
# trace_llm_invoke decorator
# ---------------------------------------------------------------------------

class TestTraceLlmInvoke(unittest.TestCase):

    def test_wraps_sync_function(self):
        @trace_llm_invoke
        def invoke(prompt, model='gpt-4'):
            return 'response'

        self.assertEqual(invoke('hello'), 'response')

    def test_preserves_function_name(self):
        @trace_llm_invoke
        def anthropic_invoke(prompt):
            return ''

        self.assertEqual(anthropic_invoke.__name__, 'anthropic_invoke')

    def test_sync_exception_propagates(self):
        @trace_llm_invoke
        def failing_invoke(prompt):
            raise ConnectionError('network')

        with self.assertRaises(ConnectionError):
            failing_invoke('hello')

    def test_wraps_async_function(self):
        import asyncio

        @trace_llm_invoke
        async def async_invoke(prompt):
            return 'async response'

        result = asyncio.run(async_invoke('hello'))
        self.assertEqual(result, 'async response')


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------

class TestJSONFormatter(unittest.TestCase):

    def _make_record(self, msg: str, **extra) -> logging.LogRecord:
        record = logging.makeLogRecord({
            'name': 'test.logger',
            'levelname': 'INFO',
            'msg': msg,
        })
        record._structured = extra
        return record

    def test_output_is_valid_json(self):
        fmt = JSONFormatter()
        line = fmt.format(self._make_record('hello'))
        parsed = json.loads(line)
        self.assertIsInstance(parsed, dict)

    def test_required_fields_present(self):
        fmt = JSONFormatter()
        line = fmt.format(self._make_record('test msg'))
        d = json.loads(line)
        for key in ('ts', 'level', 'logger', 'msg'):
            self.assertIn(key, d)

    def test_level_is_lowercase(self):
        fmt = JSONFormatter()
        line = fmt.format(self._make_record('x'))
        d = json.loads(line)
        self.assertEqual(d['level'], 'info')

    def test_extra_fields_included(self):
        fmt = JSONFormatter()
        line = fmt.format(self._make_record('read', path='README.md', bytes=1024))
        d = json.loads(line)
        self.assertEqual(d['path'], 'README.md')
        self.assertEqual(d['bytes'], 1024)

    def test_ts_is_utc_iso(self):
        fmt = JSONFormatter()
        line = fmt.format(self._make_record('x'))
        d = json.loads(line)
        self.assertTrue(d['ts'].endswith('Z'))


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------

class TestStructuredLogger(unittest.TestCase):

    def _capture(self) -> tuple[StructuredLogger, io.StringIO]:
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JSONFormatter())
        name = f'test_structured_{id(buf)}'
        inner = logging.getLogger(name)
        inner.handlers.clear()
        inner.addHandler(handler)
        inner.setLevel(logging.DEBUG)
        inner.propagate = False
        log = StructuredLogger(name)
        return log, buf

    def test_info_emits_json(self):
        log, buf = self._capture()
        log.info('started', tool='read_file')
        line = buf.getvalue().strip()
        d = json.loads(line)
        self.assertEqual(d['msg'], 'started')
        self.assertEqual(d['tool'], 'read_file')

    def test_error_emits_json(self):
        log, buf = self._capture()
        log.error('failed', code=500)
        d = json.loads(buf.getvalue().strip())
        self.assertEqual(d['level'], 'error')
        self.assertEqual(d['code'], 500)

    def test_debug_emits_json(self):
        log, buf = self._capture()
        log.debug('trace point', step=3)
        d = json.loads(buf.getvalue().strip())
        self.assertEqual(d['level'], 'debug')
        self.assertEqual(d['step'], 3)

    def test_warning_emits_json(self):
        log, buf = self._capture()
        log.warning('slow response', ms=500)
        d = json.loads(buf.getvalue().strip())
        self.assertEqual(d['level'], 'warning')

    def test_exception_includes_exc_field(self):
        log, buf = self._capture()
        log.exception('caught', exc=ValueError('oops'))
        d = json.loads(buf.getvalue().strip())
        self.assertIn('exc', d)
        self.assertIn('ValueError', d['exc'])

    def test_get_logger_returns_structured_logger(self):
        log = get_logger('anvil.tools')
        self.assertIsInstance(log, StructuredLogger)


# ---------------------------------------------------------------------------
# configure_json_logging
# ---------------------------------------------------------------------------

class TestConfigureJsonLogging(unittest.TestCase):

    def test_installs_json_handler(self):
        import anvil.observability.logging as log_mod
        orig = log_mod._CONFIGURED
        log_mod._CONFIGURED = False
        buf = io.StringIO()
        try:
            configure_json_logging(stream=buf, service='anvil_test_cfg')
            logger = logging.getLogger('anvil_test_cfg')
            logger.info('probe')
            output = buf.getvalue()
            self.assertTrue(output.strip())
            d = json.loads(output.strip())
            self.assertEqual(d['msg'], 'probe')
        finally:
            log_mod._CONFIGURED = orig
            logging.getLogger('anvil_test_cfg').handlers.clear()

    def test_idempotent_second_call(self):
        import anvil.observability.logging as log_mod
        orig = log_mod._CONFIGURED
        log_mod._CONFIGURED = True
        try:
            buf = io.StringIO()
            configure_json_logging(stream=buf, service='anvil_test_idem')
            # Handler should NOT have been added since already configured
            self.assertEqual(len(logging.getLogger('anvil_test_idem').handlers), 0)
        finally:
            log_mod._CONFIGURED = orig


if __name__ == '__main__':
    unittest.main()
