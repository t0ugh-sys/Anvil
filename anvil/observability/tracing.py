"""Tracing module — zero-hard-dependency OpenTelemetry wrapper.

When the ``opentelemetry-api`` package is installed the real OTel tracer is
used; otherwise every call is a fast no-op so production code that imports
this module never breaks in environments without OTel installed.

Usage::

    from anvil.observability.tracing import get_tracer, trace_tool_call, trace_llm_invoke

    tracer = get_tracer('anvil.tools')

    with tracer.start_as_current_span('my_operation') as span:
        span.set_attribute('key', 'value')
        ...

    @trace_tool_call
    def my_tool(ctx, args):
        ...
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

__all__ = [
    'Span',
    'NoOpSpan',
    'NoOpTracer',
    'get_tracer',
    'trace_tool_call',
    'trace_llm_invoke',
]

_OTEL_AVAILABLE: bool = False
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Span, Tracer  # noqa: F401
    _OTEL_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# No-op implementations (used when OTel is not installed)
# ---------------------------------------------------------------------------

class NoOpSpan:
    """Minimal span that satisfies the OTel Span interface without doing anything."""

    def set_attribute(self, key: str, value: Any) -> 'NoOpSpan':
        return self

    def set_status(self, status: Any, description: str = '') -> 'NoOpSpan':
        return self

    def record_exception(self, exc: BaseException, **_: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass

    def end(self, end_time: float | None = None) -> None:
        pass

    def __enter__(self) -> 'NoOpSpan':
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class NoOpTracer:
    """Tracer that always returns ``NoOpSpan`` — zero overhead."""

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: dict | None = None,
        **_: Any,
    ) -> Generator[NoOpSpan, None, None]:
        span = NoOpSpan()
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span

    def start_span(self, name: str, **_: Any) -> NoOpSpan:
        return NoOpSpan()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_tracer(name: str, *, version: str = '') -> Any:
    """Return an OTel tracer if available, otherwise a ``NoOpTracer``."""
    if _OTEL_AVAILABLE:
        return _otel_trace.get_tracer(name, version or None)
    return NoOpTracer()


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def trace_tool_call(fn: Callable) -> Callable:
    """Wrap a tool handler with a tracing span.

    Attaches ``tool.name``, ``tool.ok``, and elapsed time as span attributes.
    Works on both sync and async functions.
    """
    import asyncio
    import inspect

    tracer = get_tracer('anvil.tools')

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async_wrapper(ctx: Any, args: Any, *a: Any, **kw: Any) -> Any:
            tool_name = getattr(fn, '__name__', 'unknown_tool')
            with tracer.start_as_current_span(
                f'tool/{tool_name}',
                attributes={'tool.name': tool_name},
            ) as span:
                t0 = time.perf_counter()
                try:
                    result = await fn(ctx, args, *a, **kw)
                    span.set_attribute('tool.ok', getattr(result, 'ok', True))
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    raise
                finally:
                    span.set_attribute('tool.elapsed_ms', round((time.perf_counter() - t0) * 1000, 2))
        return _async_wrapper

    @functools.wraps(fn)
    def _sync_wrapper(ctx: Any, args: Any, *a: Any, **kw: Any) -> Any:
        tool_name = getattr(fn, '__name__', 'unknown_tool')
        with tracer.start_as_current_span(
            f'tool/{tool_name}',
            attributes={'tool.name': tool_name},
        ) as span:
            t0 = time.perf_counter()
            try:
                result = fn(ctx, args, *a, **kw)
                span.set_attribute('tool.ok', getattr(result, 'ok', True))
                return result
            except Exception as exc:
                span.record_exception(exc)
                raise
            finally:
                span.set_attribute('tool.elapsed_ms', round((time.perf_counter() - t0) * 1000, 2))
    return _sync_wrapper


def trace_llm_invoke(fn: Callable) -> Callable:
    """Wrap an LLM invocation with a tracing span.

    Attaches ``llm.model`` (if the first positional arg carries a ``model``
    attribute, or if the keyword arg ``model`` is present) and elapsed time.
    """
    import asyncio
    import inspect

    tracer = get_tracer('anvil.llm')

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async_llm_wrapper(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get('model', '') or (
                getattr(args[0], 'model', '') if args else ''
            )
            with tracer.start_as_current_span(
                'llm/invoke',
                attributes={'llm.model': str(model)},
            ) as span:
                t0 = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    raise
                finally:
                    span.set_attribute('llm.elapsed_ms', round((time.perf_counter() - t0) * 1000, 2))
        return _async_llm_wrapper

    @functools.wraps(fn)
    def _sync_llm_wrapper(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get('model', '') or (
            getattr(args[0], 'model', '') if args else ''
        )
        with tracer.start_as_current_span(
            'llm/invoke',
            attributes={'llm.model': str(model)},
        ) as span:
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as exc:
                span.record_exception(exc)
                raise
            finally:
                span.set_attribute('llm.elapsed_ms', round((time.perf_counter() - t0) * 1000, 2))
    return _sync_llm_wrapper
