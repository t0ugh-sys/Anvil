from __future__ import annotations

import unittest
from pathlib import Path

import _bootstrap  # noqa: F401


class LearningPathTests(unittest.TestCase):
    def test_should_include_guided_agent_scripts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = [
            root / 'agents' / 'README.md',
            root / 'agents' / 's01_loop.py',
            root / 'agents' / 's02_protocol.py',
            root / 'agents' / 's03_memory.py',
            root / 'agents' / 's04_skills.py',
            root / 'agents' / 's05_coding.py',
            root / 'agents' / 's06_team.py',
            root / 'agents' / 's_full.py',
        ]
        for path in expected:
            self.assertTrue(path.exists(), msg=str(path))

    def test_should_document_repo_layout_and_skills(self) -> None:
        root = Path(__file__).resolve().parents[1]
        readme = (root / 'README.md').read_text(encoding='utf-8')
        self.assertIn('agents/', readme)
        self.assertIn('skills/', readme)
        self.assertTrue((root / 'docs' / 'repo-layout.md').exists())
        self.assertTrue((root / 'skills' / 'README.md').exists())


if __name__ == '__main__':
    unittest.main()
