from __future__ import annotations

import unittest
from pathlib import Path

import _bootstrap  # noqa: F401


class LearningPathTests(unittest.TestCase):
    def test_should_document_learning_path_in_docs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        learning_path = (root / 'docs' / 'learning-path.md').read_text(encoding='utf-8')
        self.assertIn('src/loop_agent/core/', learning_path)
        self.assertIn('src/loop_agent/tool_use_loop.py', learning_path)

    def test_should_document_repo_layout_and_skills(self) -> None:
        root = Path(__file__).resolve().parents[1]
        readme = (root / 'README.md').read_text(encoding='utf-8')
        self.assertIn('skills/', readme)
        self.assertIn('skills/<name>/SKILL.md', readme)
        self.assertIn('todo_write', readme)
        self.assertIn('todo_reminder', readme)
        self.assertTrue((root / 'docs' / 'repo-layout.md').exists())
        self.assertTrue((root / 'skills' / 'README.md').exists())
        self.assertTrue((root / 'skills' / 'files' / 'SKILL.md').exists())
        self.assertTrue((root / 'skills' / 'commands' / 'SKILL.md').exists())


if __name__ == '__main__':
    unittest.main()
