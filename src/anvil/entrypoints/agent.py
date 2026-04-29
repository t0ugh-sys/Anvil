from __future__ import annotations

import sys

from ..agent_cli import build_parser
from ..services.session_runtime import build_interactive_parser, run_interactive_command, should_launch_interactive


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    argv = list(sys.argv[1:] if argv is None else argv)
    if should_launch_interactive(argv):
        parser = build_interactive_parser()
        args = parser.parse_args(argv)
        code = run_interactive_command(args, default_run_id=_default_run_id())
        raise SystemExit(code)
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.handler(args)
    raise SystemExit(code)


def _default_run_id() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


if __name__ == '__main__':
    main()
