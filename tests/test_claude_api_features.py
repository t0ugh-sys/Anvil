"""Tests for Claude API integration: Prompt Caching, Token Usage Tracking."""
from __future__ import annotations

import _bootstrap  # noqa: F401

from anvil.llm.providers import (
    TokenUsageRecord,
    TokenUsageTracker,
    PromptCache,
    _anthropic_file_tools,
)


# ============== Prompt Caching ==============


class TestAnthropicPromptCaching:
    def test_tools_have_cache_control(self):
        tools = _anthropic_file_tools()
        assert len(tools) == 5
        # Last tool should have cache_control
        last_tool = tools[-1]
        assert 'cache_control' in last_tool
        assert last_tool['cache_control'] == {'type': 'ephemeral'}

    def test_other_tools_no_cache_control(self):
        tools = _anthropic_file_tools()
        for tool in tools[:-1]:
            assert 'cache_control' not in tool

    def test_tool_schemas_intact(self):
        tools = _anthropic_file_tools()
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
            assert 'input_schema' in tool
            assert 'type' in tool['input_schema']
            assert 'properties' in tool['input_schema']
            assert 'required' in tool['input_schema']

    def test_tool_names(self):
        tools = _anthropic_file_tools()
        names = [t['name'] for t in tools]
        assert 'read_file' in names
        assert 'write_file' in names
        assert 'apply_patch' in names
        assert 'search' in names
        assert 'run_command' in names


# ============== Token Usage Tracking ==============


class TestTokenUsageRecord:
    def test_defaults(self):
        r = TokenUsageRecord()
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.total_tokens == 0

    def test_total_tokens(self):
        r = TokenUsageRecord(input_tokens=100, output_tokens=50)
        assert r.total_tokens == 150

    def test_cache_hit_ratio_no_cache(self):
        r = TokenUsageRecord(input_tokens=100, cache_read_input_tokens=0)
        assert r.cache_hit_ratio == 0.0

    def test_cache_hit_ratio_full_cache(self):
        r = TokenUsageRecord(input_tokens=0, cache_read_input_tokens=100)
        assert r.cache_hit_ratio == 1.0

    def test_cache_hit_ratio_partial(self):
        r = TokenUsageRecord(input_tokens=50, cache_read_input_tokens=50)
        assert r.cache_hit_ratio == 0.5

    def test_cache_hit_ratio_zero_total(self):
        r = TokenUsageRecord()
        assert r.cache_hit_ratio == 0.0


class TestTokenUsageTracker:
    def test_initial_state(self):
        t = TokenUsageTracker()
        assert t.call_count == 0
        assert t.total_input_tokens == 0
        assert t.total_output_tokens == 0
        assert t.last() is None

    def test_record_usage(self):
        t = TokenUsageTracker()
        t.record({'input_tokens': 100, 'output_tokens': 50})
        assert t.call_count == 1
        assert t.total_input_tokens == 100
        assert t.total_output_tokens == 50

    def test_record_cache_tokens(self):
        t = TokenUsageTracker()
        t.record({
            'input_tokens': 100,
            'output_tokens': 50,
            'cache_creation_input_tokens': 200,
            'cache_read_input_tokens': 80,
        })
        assert t.total_cache_creation_tokens == 200
        assert t.total_cache_read_tokens == 80

    def test_multiple_records(self):
        t = TokenUsageTracker()
        t.record({'input_tokens': 100, 'output_tokens': 50})
        t.record({'input_tokens': 200, 'output_tokens': 80})
        assert t.call_count == 2
        assert t.total_input_tokens == 300
        assert t.total_output_tokens == 130

    def test_summary(self):
        t = TokenUsageTracker()
        t.record({'input_tokens': 100, 'output_tokens': 50, 'cache_read_input_tokens': 30})
        s = t.summary()
        assert s['calls'] == 1
        assert s['input_tokens'] == 100
        assert s['output_tokens'] == 50
        assert s['cache_read_tokens'] == 30
        assert s['total_tokens'] == 150

    def test_last_record(self):
        t = TokenUsageTracker()
        t.record({'input_tokens': 100, 'output_tokens': 50}, model='claude-3')
        last = t.last()
        assert last is not None
        assert last.model == 'claude-3'
        assert last.input_tokens == 100

    def test_reset(self):
        t = TokenUsageTracker()
        t.record({'input_tokens': 100, 'output_tokens': 50})
        t.reset()
        assert t.call_count == 0
        assert t.last() is None

    def test_handle_missing_fields(self):
        t = TokenUsageTracker()
        t.record({})  # empty usage dict
        assert t.call_count == 1
        assert t.total_input_tokens == 0

    def test_handle_none_values(self):
        t = TokenUsageTracker()
        t.record({'input_tokens': None, 'output_tokens': None})
        assert t.call_count == 1
        assert t.total_input_tokens == 0
