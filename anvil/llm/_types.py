from __future__ import annotations

from typing import Callable, Dict, List, Set

InvokeFn = Callable[[str], str]
ChatInvokeFn = Callable[[List[Dict[str, str]]], str]

DEFAULT_RETRY_HTTP_CODES: Set[int] = {502, 503, 504, 524}
