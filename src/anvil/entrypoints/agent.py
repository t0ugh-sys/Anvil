from __future__ import annotations

import sys

from ..agent_cli import build_parser
from ..services.session_runtime import build_interactive_parser, run_interactive_command, should_launch_interactive
from ..utils import default_run_id


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[call-arg]
    argv = list(sys.argv[1:] if argv is None else argv)
    if should_launch_interactive(argv):
        parser = build_interactive_parser()
        args = parser.parse_args(argv)
        code = run_interactive_command(args, default_run_id=default_run_id())
        raise SystemExit(code)
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.handler(args)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
