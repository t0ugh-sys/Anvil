from __future__ import annotations

import os
import sys


def ensure_src_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(project_root, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


ensure_src_on_path()

