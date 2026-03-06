"""
Logging system for LoopAgent

Provides structured logging with different levels and outputs.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogOutput(Enum):
    """Log output destinations."""
    STDOUT = "stdout"
    STDERR = "stderr"
    FILE = "file"
    JSONL = "jsonl"


class Logger:
    """Simple structured logger for LoopAgent."""
    
    def __init__(
        self,
        name: str = "loop_agent",
        level: LogLevel = LogLevel.INFO,
        output: LogOutput = LogOutput.STDOUT,
        file_path: str | Path | None = None,
    ):
        self.name = name
        self.level = level
        self.output = output
        self.file_path = Path(file_path) if file_path else None
    
    def _should_log(self, level: LogLevel) -> bool:
        return level.value >= self.level.value
    
    def _format(self, level: LogLevel, message: str, **kwargs: Any) -> str:
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        base = f"[{timestamp}] {level.name} [{self.name}] {message}"
        
        if kwargs:
            import json
            extra = json.dumps(kwargs, ensure_ascii=False)
            base += f" | {extra}"
        
        return base
    
    def _write(self, formatted: str) -> None:
        if self.output == LogOutput.STDOUT:
            print(formatted, file=sys.stdout)
        elif self.output == LogOutput.STDERR:
            print(formatted, file=sys.stderr)
        elif self.output == LogOutput.FILE and self.file_path:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')
        elif self.output == LogOutput.JSONL and self.file_path:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')
    
    def debug(self, message: str, **kwargs: Any) -> None:
        if self._should_log(LogLevel.DEBUG):
            self._write(self._format(LogLevel.DEBUG, message, **kwargs))
    
    def info(self, message: str, **kwargs: Any) -> None:
        if self._should_log(LogLevel.INFO):
            self._write(self._format(LogLevel.INFO, message, **kwargs))
    
    def warning(self, message: str, **kwargs: Any) -> None:
        if self._should_log(LogLevel.WARNING):
            self._write(self._format(LogLevel.WARNING, message, **kwargs))
    
    def error(self, message: str, **kwargs: Any) -> None:
        if self._should_log(LogLevel.ERROR):
            self._write(self._format(LogLevel.ERROR, message, **kwargs))
    
    def critical(self, message: str, **kwargs: Any) -> None:
        if self._should_log(LogLevel.CRITICAL):
            self._write(self._format(LogLevel.CRITICAL, message, **kwargs))
    
    def log_step(self, step: int, action: str, **kwargs: Any) -> None:
        """Log agent step."""
        self.info(f"Step {step}: {action}", **kwargs)
    
    def log_tool(self, tool: str, result: bool, **kwargs: Any) -> None:
        """Log tool execution."""
        status = "ok" if result else "failed"
        self.debug(f"Tool {tool}: {status}", **kwargs)
    
    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log arbitrary event."""
        self.info(f"Event: {event}", **kwargs)


# Global logger instance
_default_logger: Logger | None = None


def get_logger(
    name: str = "loop_agent",
    level: str = "INFO",
    output: str = "stdout",
    file_path: str | Path | None = None,
) -> Logger:
    """Get or create the global logger."""
    global _default_logger
    
    if _default_logger is None:
        level_enum = LogLevel[level.upper()] if isinstance(level, str) else level
        output_enum = LogOutput(output) if isinstance(output, str) else output
        _default_logger = Logger(
            name=name,
            level=level_enum,
            output=output_enum,
            file_path=file_path,
        )
    
    return _default_logger


def set_logger(logger: Logger) -> None:
    """Set the global logger."""
    global _default_logger
    _default_logger = logger
