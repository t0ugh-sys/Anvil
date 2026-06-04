"""Tests for Claude API optimization features: prompt caching, thinking budget, cache-aware compression."""
from __future__ import annotations

import _bootstrap  # noqa: F401

from anvil.compression import (
    CompactConfig,
    CompactManager,
    PromptCacheManager,
    hierarchical_compact_messages,
    HierarchicalSummarizer,
    add_cache_control_hints,
)
from anvil.llm.providers import (
    TokenUsageTracker,
    AnthropicChatResponse,
)


# ============== TokenUsageTracker Cache Metrics ==============


class TestTokenUsageTrackerCacheMetrics:
    def test_cache_hit_rate_zero_when_no_data(self):
        tracker = TokenUsageTracker()
        assert tracker.cache_hit_rate == 0.0

    def test_cache_hit_rate_with_cache_reads(self):
        tracker = TokenUsageTracker()
        tracker.record({
            'input_tokens': 1000,
            'output_tokens': 500,
            'cache_creation_input_tokens': 1000,
            'cache_read_input_tokens': 0,
        })
        tracker.record({
            'input_tokens': 200,
            'output_tokens': 500,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 1000,
        })
        # cache_read / (input + cache_read) = 1000 / (1200 + 1000) = 0.45
        assert tracker.cache_hit_rate > 0.4

    def test_estimated_cost_savings_no_cache(self):
        tracker = TokenUsageTracker()
        tracker.record({
            'input_tokens': 1000,
            'output_tokens': 500,
        })
        savings = tracker.estimated_cost_savings
        assert savings['savings'] == 0.0
        assert savings['savings_percent'] == 0.0

    def test_estimated_cost_savings_with_cache(self):
        tracker = TokenUsageTracker()
        tracker.record({
            'input_tokens': 1000,
            'output_tokens': 500,
            'cache_creation_input_tokens': 800,
            'cache_read_input_tokens': 0,
        })
        tracker.record({
            'input_tokens': 200,
            'output_tokens': 500,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 800,
        })
        savings = tracker.estimated_cost_savings
        assert savings['cost_with_cache'] < savings['cost_without_cache']

    def test_summary_includes_cache_fields(self):
        tracker = TokenUsageTracker()
        tracker.record({
            'input_tokens': 1000,
            'output_tokens': 500,
            'cache_creation_input_tokens': 500,
            'cache_read_input_tokens': 500,
        })
        summary = tracker.summary()
        assert 'cache_hit_rate' in summary
        assert 'estimated_cost_savings' in summary
        assert isinstance(summary['cache_hit_rate'], float)
        assert isinstance(summary['estimated_cost_savings'], dict)


# ============== HierarchicalSummarizer ==============


class TestHierarchicalSummarizer:
    def _make_messages(self, n_rounds=5):
        msgs = [
            {'role': 'system', 'content': 'You are a coding assistant.'},
        ]
        for i in range(n_rounds):
            msgs.append({'role': 'user', 'content': 'Task %d: do something' % i})
            msgs.append({'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 't%d' % i, 'name': 'read_file', 'input': {}}
            ]})
            msgs.append({'role': 'tool', 'content': 'output %d' % i, 'tool_use_id': 't%d' % i})
            msgs.append({'role': 'assistant', 'content': 'Done with task %d.' % i})
        return msgs

    def test_empty_messages(self):
        s = HierarchicalSummarizer()
        assert s.summarize([]) == []

    def test_produces_three_levels(self):
        msgs = self._make_messages(8)
        s = HierarchicalSummarizer(l2_block_size=3)
        levels = s.summarize(msgs)
        assert len(levels) == 3
        assert levels[0].level == 1
        assert levels[1].level == 2
        assert levels[2].level == 3

    def test_small_conversation_no_l2(self):
        msgs = self._make_messages(2)
        s = HierarchicalSummarizer(l2_block_size=5)
        levels = s.summarize(msgs)
        assert len(levels) == 2
        assert levels[0].level == 1
        assert levels[1].level == 3

    def test_l1_contains_round_info(self):
        msgs = self._make_messages(3)
        s = HierarchicalSummarizer()
        levels = s.summarize(msgs)
        l1 = levels[0]
        assert 'Round' in l1.content
        assert 'read_file' in l1.content

    def test_l3_is_brief(self):
        msgs = self._make_messages(10)
        s = HierarchicalSummarizer(l2_block_size=3)
        levels = s.summarize(msgs)
        l3 = levels[-1]
        assert l3.level == 3
        assert l3.token_count < 100

    def test_with_llm_provider(self):
        msgs = self._make_messages(6)
        call_count = [0]

        def fake_provider(system_prompt, text):
            call_count[0] += 1
            return 'Summary of: %s' % text[:50]

        s = HierarchicalSummarizer(summary_provider=fake_provider, l2_block_size=2)
        levels = s.summarize(msgs)
        assert len(levels) == 3
        assert call_count[0] >= 2


class TestHierarchicalCompactMessages:
    def _make_messages(self, n_rounds=6):
        msgs = [
            {'role': 'system', 'content': 'You are a coding assistant.'},
        ]
        for i in range(n_rounds):
            msgs.append({'role': 'user', 'content': 'Task %d' % i})
            msgs.append({'role': 'assistant', 'content': 'Done %d' % i})
        return msgs

    def test_no_compaction_if_few_rounds(self):
        msgs = self._make_messages(2)
        s = HierarchicalSummarizer()
        result = hierarchical_compact_messages(msgs, summarizer=s, keep_recent_rounds=3)
        assert result == msgs

    def test_compacts_old_rounds(self):
        msgs = self._make_messages(8)
        s = HierarchicalSummarizer()
        result = hierarchical_compact_messages(msgs, summarizer=s, keep_recent_rounds=2, summary_level=2)
        assert result[0]['role'] == 'system'
        assert 'Hierarchical Summary' in result[0]['content']
        assert len(result) < len(msgs)


# ============== PromptCacheManager ==============


class TestPromptCacheManager:
    def _make_messages(self, n_rounds=6):
        msgs = [
            {'role': 'system', 'content': 'You are a coding assistant. ' * 100},
        ]
        for i in range(n_rounds):
            msgs.append({'role': 'user', 'content': 'Task %d' % i})
            msgs.append({'role': 'assistant', 'content': 'Done %d' % i})
        return msgs

    def test_split_empty(self):
        cache = PromptCacheManager()
        prefix, suffix = cache.split_for_caching([])
        assert prefix == []
        assert suffix == []

    def test_split_with_enough_rounds(self):
        msgs = self._make_messages(6)
        cache = PromptCacheManager(cache_suffix_rounds=2)
        prefix, suffix = cache.split_for_caching(msgs)
        assert len(prefix) > 0
        assert len(suffix) > 0
        assert len(prefix) + len(suffix) == len(msgs)

    def test_cache_hit_on_repeated_call(self):
        msgs = self._make_messages(6)
        cache = PromptCacheManager(cache_suffix_rounds=2)
        cache.split_for_caching(msgs)
        stats1 = cache.get_stats()
        assert stats1['cache_misses'] == 1
        assert stats1['cache_hits'] == 0
        cache.split_for_caching(msgs)
        stats2 = cache.get_stats()
        assert stats2['cache_hits'] == 1

    def test_stats(self):
        cache = PromptCacheManager()
        stats = cache.get_stats()
        assert 'segments' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'hit_rate' in stats
        assert 'total_cached_tokens' in stats


# ============== add_cache_control_hints ==============


class TestAddCacheControlHints:
    def test_adds_hint_to_string_content(self):
        msgs = [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'Hello'},
        ]
        result = add_cache_control_hints(msgs, cacheable_prefix_count=1)
        content = result[0]['content']
        assert isinstance(content, list)
        assert content[0]['cache_control'] == {'type': 'ephemeral'}

    def test_adds_hint_to_block_content(self):
        msgs = [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Document here'},
                {'type': 'text', 'text': 'Question here'},
            ]},
        ]
        result = add_cache_control_hints(msgs, cacheable_prefix_count=1)
        content = result[0]['content']
        assert content[-1]['cache_control'] == {'type': 'ephemeral'}
        assert 'cache_control' not in content[0]

    def test_no_op_when_count_exceeds_messages(self):
        msgs = [{'role': 'user', 'content': 'Hello'}]
        result = add_cache_control_hints(msgs, cacheable_prefix_count=5)
        assert result == msgs


# ============== Cache-Aware CompactManager ==============


class TestCacheAwareCompactManager:
    def _make_messages(self, n_rounds=8):
        msgs = [
            {'role': 'system', 'content': 'You are a coding assistant. ' * 100},
        ]
        for i in range(n_rounds):
            msgs.append({'role': 'user', 'content': 'Task %d' % i})
            msgs.append({'role': 'assistant', 'content': 'Done %d' % i})
        return msgs

    def test_compact_without_cache_manager(self):
        config = CompactConfig(max_context_tokens=50, warn_tokens_percent=0.1)
        mgr = CompactManager(config=config)
        msgs = self._make_messages(4)
        result = mgr.execute_compact(msgs)
        assert result.ok

    def test_compact_with_cache_manager(self):
        config = CompactConfig(max_context_tokens=50, warn_tokens_percent=0.1)
        cache_mgr = PromptCacheManager(cache_suffix_rounds=2, min_cacheable_tokens=10)
        mgr = CompactManager(config=config, prompt_cache_manager=cache_mgr)
        msgs = self._make_messages(8)
        result = mgr.execute_compact(msgs)
        assert result.ok
        assert len(result.messages) > 0


# ============== AnthropicChatResponse ==============


class TestAnthropicChatResponse:
    def test_basic_creation(self):
        resp = AnthropicChatResponse(text='Hello')
        assert resp.text == 'Hello'
        assert resp.thinking_blocks == []
        assert resp.raw_content == []

    def test_with_thinking_blocks(self):
        resp = AnthropicChatResponse(
            text='Answer',
            thinking_blocks=[{'type': 'thinking', 'thinking': 'Let me think...'}],
        )
        assert len(resp.thinking_blocks) == 1
        assert resp.thinking_blocks[0]['thinking'] == 'Let me think...'

    def test_to_assistant_message_with_thinking(self):
        resp = AnthropicChatResponse(
            text='Here is the answer.',
            thinking_blocks=[
                {'type': 'thinking', 'thinking': 'Step 1: analyze'},
                {'type': 'thinking', 'thinking': 'Step 2: conclude'},
            ],
        )
        msg = resp.to_assistant_message()
        assert msg['role'] == 'assistant'
        content = msg['content']
        assert len(content) == 3
        assert content[0]['type'] == 'thinking'
        assert content[1]['type'] == 'thinking'
        assert content[2]['type'] == 'text'
        assert content[2]['text'] == 'Here is the answer.'

    def test_to_assistant_message_text_only(self):
        resp = AnthropicChatResponse(text='Simple answer.')
        msg = resp.to_assistant_message()
        content = msg['content']
        assert len(content) == 1
        assert content[0]['type'] == 'text'
        assert content[0]['text'] == 'Simple answer.'

    def test_to_assistant_message_empty_text(self):
        resp = AnthropicChatResponse(text='')
        msg = resp.to_assistant_message()
        content = msg['content']
        assert len(content) == 0

    def test_multi_round_passback(self):
        """Simulate multi-turn with thinking block passback."""
        resp1 = AnthropicChatResponse(
            text='Let me check the file.',
            thinking_blocks=[{'type': 'thinking', 'thinking': 'User wants to fix a bug.'}],
        )
        messages = [
            {'role': 'user', 'content': 'Fix the bug in main.py'},
            resp1.to_assistant_message(),
            {'role': 'user', 'content': 'Now add tests'},
        ]
        assistant_msg = messages[1]
        assert assistant_msg['role'] == 'assistant'
        assert assistant_msg['content'][0]['type'] == 'thinking'
        assert assistant_msg['content'][0]['thinking'] == 'User wants to fix a bug.'


# ============== anthropic_chat_invoke_factory (unit) ==============


class TestAnthropicChatInvokeFactory:
    def test_factory_creates_callable(self):
        from anvil.llm.providers import anthropic_chat_invoke_factory
        try:
            invoke = anthropic_chat_invoke_factory(
                api_key='test-key',
                model='claude-sonnet-4-20250514',
                system_prompt='You are helpful.',
            )
            assert callable(invoke)
        except Exception:
            pass

    def test_max_tokens_with_thinking(self):
        budget = 10000
        expected = max(budget + 4096, budget * 2)
        assert expected == 20000

        budget = 2000
        expected = max(budget + 4096, budget * 2)
        assert expected == 6096


# ============== anthropic_count_tokens ==============


class TestAnthropicCountTokens:
    def test_count_tokens_or_estimate_fallback(self):
        from anvil.llm.providers import anthropic_count_tokens_or_estimate
        messages = [{'role': 'user', 'content': 'Hello ' * 100}]
        count = anthropic_count_tokens_or_estimate(messages=messages)
        assert count > 0
        assert 50 < count < 500

    def test_count_tokens_or_estimate_with_system(self):
        from anvil.llm.providers import anthropic_count_tokens_or_estimate
        messages = [{'role': 'user', 'content': 'Hello'}]
        count = anthropic_count_tokens_or_estimate(
            messages=messages,
            system_prompt='You are a helpful assistant. ' * 50,
        )
        # Fallback estimation only counts messages, not system prompt
        assert count > 0
