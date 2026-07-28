from __future__ import annotations

import argparse

__all__ = ['register_tools_parser', 'register_skills_parser', 'register_replay_parser', 'register_doctor_parser']


def register_tools_parser(subparsers, *, handler) -> argparse.ArgumentParser:
    tools = subparsers.add_parser(
        'tools',
        help='list tools exposed to the loop',
        description='List tool handlers available to the coding tool-use loop.',
    )
    tools.add_argument('--verbose', action='store_true', help='Show tool descriptions and capability metadata')
    tools.set_defaults(handler=handler)
    return tools


def register_skills_parser(subparsers, *, handler) -> argparse.ArgumentParser:
    skills_parser = subparsers.add_parser(
        'skills',
        help='list available skills',
        description='List optional skills that can extend the loop tool dispatch.',
    )
    skills_parser.set_defaults(handler=handler)
    return skills_parser


def register_replay_parser(subparsers, *, handler) -> argparse.ArgumentParser:
    replay = subparsers.add_parser(
        'replay',
        help='print recorded loop events',
        description='Print a recorded JSONL event stream for a previous tool-use run.',
    )
    replay.add_argument('--events-file', default='')
    replay.add_argument('--session-id', default='')
    replay.add_argument('--sessions-dir', default='.anvil/sessions')
    replay.add_argument('--pretty', action='store_true', help='Render a human-readable event stream')
    replay.add_argument('--limit', type=int, help='Limit pretty replay to the most recent N events')
    replay.set_defaults(handler=handler)
    return replay


def register_doctor_parser(subparsers, *, handler) -> argparse.ArgumentParser:
    doctor = subparsers.add_parser(
        'doctor',
        help='diagnose provider connectivity',
        description='Diagnose provider connectivity before running the tool-use loop.',
    )
    doctor.add_argument('--provider', choices=['openai_compatible'], default='openai_compatible')
    doctor.add_argument('--model', default='gpt-5.3-codex')
    doctor.add_argument('--base-url', required=True)
    doctor.add_argument('--wire-api', choices=['chat_completions', 'responses'], default='responses')
    doctor.add_argument('--api-key-env', default='OPENAI_API_KEY')
    doctor.add_argument('--provider-timeout-s', type=float, default=20.0)
    doctor.add_argument('--provider-header', action='append', default=[])
    doctor.add_argument('--output', choices=['text', 'json'], default='text')
    doctor.set_defaults(handler=handler)
    return doctor
