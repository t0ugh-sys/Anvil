"""Structured JSON logging — stdlib only, zero new dependencies.

Replaces the mix of bare ``print`` and ``logging`` calls scattered through
the codebase with a single consistent structured logger.  All records are
emitted as newline-delimited JSON (NDJSON) so log aggregators (Datadog,
CloudWatch, ELK) can parse them without extra config.

Usage::

    from anvil.observability.logging import get_logger

    log = get_logger('anvil.tools')
    log.info('tool called', tool='read_file', elapsed_ms=12.3)
    log.error('tool failed', tool='shell', error='permission denied')
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

__all__ = [
    'JSONFormatter',
    'get_logger',
    'configure_json_logging',
    'StructuredLogger',
]

_SERVICE_NAME = 'anvil'
_CONFIGURED = False


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Fields always present: ``ts`` (ISO-8601), ``level``, ``logger``, ``msg``.
    Extra keyword arguments passed to the StructuredLogger are added as
    top-level keys.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            'ts': self._iso(record.created),
            'level': record.levelname.lower(),
            'logger': record.name,
            'msg': record.getMessage(),
        }
        # Attach any structured fields stored by StructuredLogger
        extra: Dict[str, Any] = getattr(record, '_structured', {})
        payload.update(extra)
        if record.exc_info:
            payload['exc'] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)

    @staticmethod
    def _iso(ts: float) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z'


def configure_json_logging(
    *,
    level: int = logging.INFO,
    stream: Any = None,
    service: str = _SERVICE_NAME,
) -> None:
    """Install a JSON handler on the root ``anvil`` logger.

    Safe to call multiple times — only installs once per process.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    import sys
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger(service)
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


class StructuredLogger:
    """Thin wrapper around a stdlib logger that adds keyword-argument fields.

    Example::

        log = StructuredLogger('anvil.tools')
        log.info('read complete', path='README.md', bytes=4096)
        # → {"ts":"...","level":"info","logger":"anvil.tools",
        #    "msg":"read complete","path":"README.md","bytes":4096}
    """

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _emit(self, level: int, msg: str, **fields: Any) -> None:
        if not self._log.isEnabledFor(level):
            return
        record = self._log.makeRecord(
            name=self._log.name,
            level=level,
            fn='',
            lno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        record._structured = fields  # type: ignore[attr-defined]
        self._log.handle(record)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, msg, **fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit(logging.INFO, msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit(logging.WARNING, msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit(logging.ERROR, msg, **fields)

    def critical(self, msg: str, **fields: Any) -> None:
        self._emit(logging.CRITICAL, msg, **fields)

    def exception(self, msg: str, exc: BaseException, **fields: Any) -> None:
        import traceback
        self._emit(
            logging.ERROR,
            msg,
            exc=traceback.format_exception_only(type(exc), exc)[-1].strip(),
            **fields,
        )


def get_logger(name: str) -> StructuredLogger:
    """Return a ``StructuredLogger`` for the given name."""
    return StructuredLogger(name)
