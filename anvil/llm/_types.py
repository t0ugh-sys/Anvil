from __future__ import annotations

from typing import Awaitable, Callable, Dict, List, Set

InvokeFn = Callable[[str], str]
AsyncInvokeFn = Callable[[str], Awaitable[str]]
ChatInvokeFn = Callable[[List[Dict[str, str]]], str]

DEFAULT_RETRY_HTTP_CODES: Set[int] = {502, 503, 504, 524}
