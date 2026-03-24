from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from loop_agent.mailbox import JsonlMailbox, MailMessage


class MailboxTests(unittest.TestCase):
    def test_should_store_and_filter_messages(self) -> None:
        tmp_dir = Path('tests/.tmp') / f'mailbox-{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            mailbox = JsonlMailbox(tmp_dir)
            mailbox.send(MailMessage(id='m1', sender='a', recipient='b', subject='s1', body='body', task_id='t1'))
            mailbox.send(MailMessage(id='m2', sender='a', recipient='c', subject='s2', body='body', task_id='t2'))
            inbox = mailbox.inbox('b')
            self.assertEqual(len(inbox), 1)
            self.assertEqual(inbox[0].id, 'm1')
            thread = mailbox.thread('t1')
            self.assertEqual(len(thread), 1)
            self.assertEqual(thread[0].recipient, 'b')
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
