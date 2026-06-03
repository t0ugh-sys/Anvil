"""Tests for Zero2Agent-inspired improvements: compression, task notifications, plan approval, coordinator."""
from __future__ import annotations

import time

import _bootstrap  # noqa: F401 — adds src/ to sys.path

from anvil.compression import (
    TranscriptEntry,
    micro_compact_entries,
    time_based_micro_compact,
)
from anvil.subagents import TaskNotification
from anvil.team_runtime import TeamMessageType


# ============== Time-based Microcompact ==============


def _entry(kind: str, content: str, *, ts: float = 0.0, tool_name: str | None = None) -> TranscriptEntry:
    return TranscriptEntry(kind=kind, content=content, created_at=ts, tool_name=tool_name)


class TestTimeBasedMicroCompact:
    def test_empty_entries(self):
        assert time_based_micro_compact(()) == ()

    def test_no_timestamps_skips(self):
        entries = (
            _entry('thought', 'thinking'),
            _entry('tool_result', 'big output', tool_name='bash'),
        )
        # No timestamps → no compaction
        assert time_based_micro_compact(entries) == entries

    def test_recent_entries_unchanged(self):
        now = time.time()
        entries = (
            _entry('thought', 'thinking', ts=now - 60),
            _entry('tool_result', 'output', ts=now - 30, tool_name='read_file'),
        )
        result = time_based_micro_compact(entries, now_s=now, gap_threshold_s=1800)
        # Gap is only 30s, threshold is 1800s → no change
        assert result == entries

    def test_old_entries_compacted(self):
        now = time.time()
        entries = (
            _entry('thought', 'thinking', ts=now - 3600),
            _entry('tool_result', 'big output here', ts=now - 3600, tool_name='bash'),
            _entry('thought', 'more thinking', ts=now - 3600),
        )
        result = time_based_micro_compact(entries, now_s=now, gap_threshold_s=1800)
        # Gap is 3600s > 1800s → tool results cleared
        assert len(result) == 3
        assert result[0].kind == 'thought'  # unchanged
        assert result[1].kind == 'tool_result'
        assert 'Previous: used bash' in result[1].content
        assert 'cache expired' in result[1].content
        assert result[2].kind == 'thought'  # unchanged

    def test_preserves_non_tool_entries(self):
        now = time.time()
        entries = (
            _entry('thought', 'my thought', ts=now - 3600),
            _entry('summary', 'a summary', ts=now - 3600),
        )
        result = time_based_micro_compact(entries, now_s=now, gap_threshold_s=1800)
        assert result[0].content == 'my thought'
        assert result[1].content == 'a summary'

    def test_combined_with_count_based(self):
        """Time-based should work after count-based microcompact."""
        now = time.time()
        entries = (
            _entry('tool_result', 'old1', ts=now - 3600, tool_name='a'),
            _entry('tool_result', 'old2', ts=now - 3600, tool_name='b'),
            _entry('tool_result', 'recent', ts=now - 10, tool_name='c'),
        )
        # Count-based first: keep last 1
        after_count = micro_compact_entries(entries, keep_last_results=1)
        # Time-based: gap is only 10s → no further change
        after_time = time_based_micro_compact(after_count, now_s=now, gap_threshold_s=1800)
        # The count-based already compacted old1/old2, time-based sees 10s gap → no change
        assert after_time == after_count


# ============== TaskNotification ==============


class TestTaskNotification:
    def test_to_xml(self):
        n = TaskNotification(
            task_id='t1', agent_id='coder', status='completed',
            summary='Done', result='File created', total_tokens=1500,
            tool_uses=5, duration_ms=3000,
        )
        xml = n.to_xml()
        assert '<task-id>t1</task-id>' in xml
        assert '<status>completed</status>' in xml
        assert '<total_tokens>1500</total_tokens>' in xml
        assert '<tool_uses>5</tool_uses>' in xml
        assert '<duration_ms>3000</duration_ms>' in xml

    def test_to_dict(self):
        n = TaskNotification(
            task_id='t2', agent_id='reviewer', status='failed',
            summary='Error', result='Stack trace...',
        )
        d = n.to_dict()
        assert d['task_id'] == 't2'
        assert d['status'] == 'failed'
        assert d['total_tokens'] == 0

    def test_result_truncated_in_xml(self):
        n = TaskNotification(
            task_id='t3', agent_id='w', status='completed',
            summary='ok', result='x' * 5000,
        )
        xml = n.to_xml()
        assert len(xml) < 5000  # result truncated to 2000

    def test_build_notification_from_result(self):
        from anvil.subagents import SubAgentResult
        r = SubAgentResult(
            agent_id='coder', task_id='t4', success=True,
            stop_reason='completed', final_output='All tests pass',
            payload={'history': ['thought: ok', 'tool[r1] ok', 'tool[r2] ok']},
            duration_ms=1500,
        )
        n = r.build_notification()
        assert n.status == 'completed'
        assert n.tool_uses == 2
        assert n.duration_ms == 1500


# ============== Plan Approval Protocol ==============


class TestPlanApprovalProtocol:
    def test_message_types_exist(self):
        assert hasattr(TeamMessageType, 'plan_approval_request')
        assert hasattr(TeamMessageType, 'plan_approval_response')
        assert TeamMessageType.plan_approval_request.value == 'plan_approval_request'
        assert TeamMessageType.plan_approval_response.value == 'plan_approval_response'

    def test_reject_requires_feedback(self):
        from anvil.team_runtime import PersistentTeamRuntime
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            rt = PersistentTeamRuntime(Path(td))
            try:
                rt.reject_plan('worker1', 'req1', sender='lead', feedback='')
                assert False, 'Should have raised ValueError'
            except ValueError as e:
                assert 'feedback is required' in str(e)

    def test_approve_plan_sends_message(self):
        from anvil.team_runtime import PersistentTeamRuntime
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            rt = PersistentTeamRuntime(Path(td))
            rt.approve_plan('worker1', 'req1', sender='lead', feedback='looks good')
            messages = rt.inbox_store.peek('worker1')
            assert len(messages) == 1
            assert messages[0].message_type == TeamMessageType.plan_approval_response
            assert messages[0].metadata['approved'] is True


# ============== Coordinator Prompt ==============


class TestCoordinatorPrompt:
    def test_coordinator_prompt_exists(self):
        from anvil.prompts import COORDINATOR_SYSTEM_PROMPT
        assert 'coordinator' in COORDINATOR_SYSTEM_PROMPT.lower()
        assert 'spawn_worker' in COORDINATOR_SYSTEM_PROMPT
        assert 'task-notification' in COORDINATOR_SYSTEM_PROMPT

    def test_coordinator_tools_spec(self):
        from anvil.prompts import COORDINATOR_TOOLS_SPEC
        assert 'spawn_worker' in COORDINATOR_TOOLS_SPEC
        assert 'send_message' in COORDINATOR_TOOLS_SPEC
        assert 'stop_worker' in COORDINATOR_TOOLS_SPEC
        # Each tool should have description and parameters
        for name, spec in COORDINATOR_TOOLS_SPEC.items():
            assert 'description' in spec
            assert 'parameters' in spec
            assert 'properties' in spec['parameters']
