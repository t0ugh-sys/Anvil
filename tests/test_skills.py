from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.skills import SkillLoader, discover_local_skill_names, skill_metadata


class SkillsTests(unittest.TestCase):
    def test_should_read_skill_metadata_from_frontmatter(self) -> None:
        meta = skill_metadata('files')
        self.assertIsNotNone(meta)
        self.assertEqual(meta['name'], 'files')
        self.assertEqual(meta['description'], 'Read, write, patch, and search files')

    def test_should_load_skill_body_from_skill_md(self) -> None:
        loader = SkillLoader()
        self.assertTrue(loader.load('commands'))
        body = loader.load_body('commands')
        self.assertIsNotNone(body)
        self.assertIn('Provided tools:', body)
        self.assertIn('run_command', body)

    def test_should_discover_local_skill_bundle_from_docs_root(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'skills-{uuid.uuid4().hex}'
        skill_dir = tmp_dir / 'custom-review'
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / 'SKILL.md').write_text(
            '---\nname: custom-review\ndescription: Review local code changes\n---\n\n# custom-review\n\nCheck diffs.',
            encoding='utf-8',
        )
        try:
            names = discover_local_skill_names(tmp_dir)
            self.assertIn('custom-review', names)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_load_local_document_skill_without_registry_entry(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'skills-{uuid.uuid4().hex}'
        skill_dir = tmp_dir / 'custom-review'
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / 'SKILL.md').write_text(
            '---\nname: custom-review\ndescription: Review local code changes\n---\n\n# custom-review\n\nCheck diffs.',
            encoding='utf-8',
        )
        try:
            loader = SkillLoader(docs_root=tmp_dir)
            self.assertTrue(loader.load('custom-review'))
            self.assertEqual(loader.metadata()[0]['description'], 'Review local code changes')
            self.assertIn('Check diffs.', loader.load_body('custom-review') or '')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
