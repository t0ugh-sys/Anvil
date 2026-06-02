"""Tests for compression module — microcompact, partial, full, and CompactManager."""
from __future__ import annotations

import json
import time

import _bootstrap  # noqa: F401

from anvil.compression import (
    CompactConfig,
    CompactManager,
    CompactReason,
    CompactResult,
    CompactState,
    CompactStrategy,
    MessageGroup,
    TranscriptEntry,
    archive_compacted_messages,
    group_messages_by_rounds,
    micro_compact_entries,
    micro_compact_messages,
    partial_compact_messages,
    time_based_micro_compact,
)


# ============== micro_compact_messages ==============


class TestMicroCompactMessages:
    def test_empty_messages(self):
        assert micro_compact_messages([]) == []

    def test_no_tool_results_unchanged(self):
        msgs = [{'role': 'user', 'content': 'hello'}]
        assert micro_compact_messages(msgs) == msgs

    def test_keeps_last_n_results(self):
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'a', 'content': 'old output'},
            ]},
            {'role': 'assistant', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'b', 'content': 'new output'},
            ]},
        ]
        result = micro_compact_messages(msgs, keep_last_results=1)
        # First result should be truncated
        assert 'truncated' in result[0]['content'][0]['content']
        # Second result should be kept
        assert result[1]['content'][0]['content'] == 'new output'

    def test_preserves_non_tool_blocks(self):
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'text', 'text': 'thinking'},
                {'type': 'tool_result', 'tool_use_id': 'x', 'content': 'result'},
            ]},
        ]
        result = micro_compact_messages(msgs, keep_last_results=0)
        # Text block preserved, tool_result truncated
        assert result[0]['content'][0]['type'] == 'text'
        assert 'truncated' in result[0]['content'][1]['content']


# ============== micro_compact_entries ==============


class TestMicroCompactEntries:
    def test_empty(self):
        assert micro_compact_entries((), keep_last_results=3) == ()

    def test_keeps_recent(self):
        entries = (
            TranscriptEntry(kind='tool_result', content='old', tool_name='a'),
            TranscriptEntry(kind='tool_result', content='new', tool_name='b'),
        )
        result = micro_compact_entries(entries, keep_last_results=1)
        assert 'Previous: used a' in result[0].content
        assert result[1].content == 'new'

    def test_preserves_non_tool(self):
        entries = (
            TranscriptEntry(kind='thought', content='thinking'),
            TranscriptEntry(kind='tool_result', content='out', tool_name='x'),
        )
        result = micro_compact_entries(entries, keep_last_results=0)
        assert result[0].content == 'thinking'
        assert 'Previous' in result[1].content


# ============== group_messages_by_rounds ==============


class TestGroupMessagesByRounds:
    def test_empty(self):
        assert group_messages_by_rounds([]) == []

    def test_single_user_message(self):
        groups = group_messages_by_rounds([{'role': 'user', 'content': 'hi'}])
        assert len(groups) == 1
        assert groups[0].round_id == 0

    def test_user_assistant_round(self):
        msgs = [
            {'role': 'user', 'content': 'do something'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}]},
        ]
        groups = group_messages_by_rounds(msgs)
        assert len(groups) == 1


# ============== partial_compact_messages ==============


class TestPartialCompact:
    def test_no_compaction_if_few_rounds(self):
        msgs = [{'role': 'user', 'content': 'hi'}]
        assert partial_compact_messages(msgs, max_rounds=10) == msgs

    def test_compacts_old_rounds(self):
        # Build enough rounds to trigger compaction
        msgs = []
        for i in range(5):
            msgs.append({'role': 'user', 'content': f'msg {i}'})
            msgs.append({'role': 'assistant', 'content': f'reply {i}'})
        result = partial_compact_messages(msgs, max_rounds=3, keep_recent_rounds=2)
        # Should have summary + recent rounds
        assert result[0]['role'] == 'system'
        assert 'summarized' in result[0]['content'].lower() or 'earlier' in result[0]['content'].lower()


# ============== CompactConfig ==============


class TestCompactConfig:
    def test_defaults(self):
        config = CompactConfig()
        assert config.max_context_tokens == 50000
        assert config.warn_tokens_percent == 0.8
        config.validate()  # should not raise

    def test_invalid_max_context(self):
        config = CompactConfig(max_context_tokens=-1)
        try:
            config.validate()
            assert False
        except ValueError:
            pass

    def test_invalid_warn_percent(self):
        config = CompactConfig(warn_tokens_percent=0)
        try:
            config.validate()
            assert False
        except ValueError:
            pass


# ============== CompactManager ==============


class TestCompactManager:
    def test_initial_state(self):
        mgr = CompactManager()
        assert mgr.state.compaction_count == 0
        assert not mgr.requested

    def test_request_sets_flag(self):
        mgr = CompactManager()
        mgr.request('manual test')
        assert mgr.requested
        assert mgr.reason == 'manual test'

    def test_should_compact_on_request(self):
        mgr = CompactManager()
        mgr.request('test')
        assert mgr.should_compact([{'role': 'user', 'content': 'short'}])

    def test_execute_compact_micro(self):
        config = CompactConfig(max_context_tokens=100000, warn_tokens_percent=0.95)
        mgr = CompactManager(config=config)
        # Small messages → MICRO strategy
        msgs = [{'role': 'user', 'content': 'hello'}]
        result = mgr.execute_compact(msgs)
        # MICRO on small messages → no change needed
        assert result.strategy == CompactStrategy.MICRO or result.ok

    def test_stats(self):
        mgr = CompactManager()
        stats = mgr.get_stats()
        assert stats['compaction_count'] == 0
        assert stats['total_tokens_saved'] == 0

    def test_choose_strategy_micro(self):
        config = CompactConfig(max_context_tokens=100000)
        mgr = CompactManager(config=config)
        msgs = [{'role': 'user', 'content': 'short'}]
        assert mgr._choose_strategy(msgs) == CompactStrategy.MICRO


# ============== archive_compacted_messages ==============


class TestArchiveCompactedMessages:
    def test_creates_archive_file(self, tmp_path):
        msgs = [{'role': 'user', 'content': 'test'}]
        path = archive_compacted_messages(
            messages=msgs,
            archive_dir=tmp_path / 'archive',
            compaction_index=1,
            goal='test goal',
            summary='test summary',
        )
        assert path.exists()
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['goal'] == 'test goal'
        assert data['summary'] == 'test summary'
        assert len(data['messages']) == 1


# ============== TranscriptEntry ==============


class TestTranscriptEntry:
    def test_to_dict(self):
        entry = TranscriptEntry(kind='thought', content='test', tool_name='bash', ok=True)
        d = entry.to_dict()
        assert d['kind'] == 'thought'
        assert d['tool_name'] == 'bash'
        assert d['ok'] is True

    def test_render_line_tool_result(self):
        entry = TranscriptEntry(kind='tool_result', content='output', tool_name='read', call_id='c1', ok=True)
        line = entry.render_line()
        assert 'read' in line
        assert 'ok' in line

    def test_render_line_thought(self):
        entry = TranscriptEntry(kind='thought', content='thinking')
        assert entry.render_line() == 'thought: thinking'

    def test_created_at_default(self):
        entry = TranscriptEntry(kind='thought', content='x')
        assert entry.created_at == 0.0

    def test_created_at_set(self):
        now = time.time()
        entry = TranscriptEntry(kind='thought', content='x', created_at=now)
        assert entry.created_at == now
