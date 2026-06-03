"""Tests for Zero2Agent round 4 improvements: PII filtering, PromptCache, SecurityMonitor."""
from __future__ import annotations

import time

import _bootstrap  # noqa: F401 — adds src/ to sys.path

from anvil.tools.base import redact_pii
from anvil.llm.providers import PromptCache
from anvil.hooks import SecurityMonitor, SecurityEvent


# ============== PII Redaction ==============


class TestRedactPII:
    def test_email_redacted(self):
        result = redact_pii('Contact alice@example.com for details')
        assert 'alice@example.com' not in result
        assert '[EMAIL REDACTED]' in result

    def test_ssn_redacted(self):
        result = redact_pii('SSN: 123-45-6789')
        assert '123-45-6789' not in result
        assert '[SSN REDACTED]' in result

    def test_credit_card_redacted(self):
        result = redact_pii('Card: 4111 1111 1111 1111')
        assert '4111' not in result
        assert '[CARD REDACTED]' in result

    def test_chinese_id_redacted(self):
        result = redact_pii('ID: 110101199003071234')
        assert '110101199003071234' not in result
        assert '[ID REDACTED]' in result

    def test_chinese_phone_redacted(self):
        result = redact_pii('Phone: 13812345678')
        assert '13812345678' not in result
        assert '[PHONE REDACTED]' in result

    def test_api_key_redacted(self):
        result = redact_pii('Key: sk-abc123def456ghi789jkl012mno345')
        assert 'sk-abc123def456ghi789jkl012mno345' not in result
        assert '[API_KEY REDACTED]' in result

    def test_aws_key_redacted(self):
        result = redact_pii('Key: AKIAIOSFODNN7EXAMPLE')
        assert 'AKIAIOSFODNN7EXAMPLE' not in result
        assert '[AWS_KEY REDACTED]' in result

    def test_private_key_redacted(self):
        result = redact_pii('-----BEGIN RSA PRIVATE KEY-----\nMIIEow...')
        assert 'BEGIN RSA PRIVATE KEY' not in result
        assert '[PRIVATE_KEY REDACTED]' in result

    def test_clean_text_unchanged(self):
        text = 'Fix the bug in main.py at line 42'
        assert redact_pii(text) == text

    def test_empty_input(self):
        assert redact_pii('') == ''
        assert redact_pii(None) is None

    def test_multiple_pii_redacted(self):
        text = 'Email: test@foo.com, Phone: 13900001111'
        result = redact_pii(text)
        assert 'test@foo.com' not in result
        assert '13900001111' not in result
        assert '[EMAIL REDACTED]' in result
        assert '[PHONE REDACTED]' in result


# ============== PromptCache ==============


class TestPromptCache:
    def test_make_key_deterministic(self):
        k1 = PromptCache.make_key('model1', 'hello')
        k2 = PromptCache.make_key('model1', 'hello')
        assert k1 == k2

    def test_make_key_different_model(self):
        k1 = PromptCache.make_key('model1', 'hello')
        k2 = PromptCache.make_key('model2', 'hello')
        assert k1 != k2

    def test_make_key_different_prompt(self):
        k1 = PromptCache.make_key('model1', 'hello')
        k2 = PromptCache.make_key('model1', 'world')
        assert k1 != k2

    def test_get_set_basic(self):
        cache = PromptCache()
        key = cache.make_key('m', 'p')
        assert cache.get(key) is None
        cache.set(key, 'result')
        assert cache.get(key) == 'result'

    def test_lru_eviction(self):
        cache = PromptCache(max_size=2)
        k1 = PromptCache.make_key('m', 'p1')
        k2 = PromptCache.make_key('m', 'p2')
        k3 = PromptCache.make_key('m', 'p3')
        cache.set(k1, 'r1')
        cache.set(k2, 'r2')
        cache.set(k3, 'r3')  # evicts k1
        assert cache.get(k1) is None
        assert cache.get(k2) == 'r2'
        assert cache.get(k3) == 'r3'

    def test_ttl_expiration(self):
        cache = PromptCache(ttl_seconds=0.1)
        key = cache.make_key('m', 'p')
        cache.set(key, 'result')
        assert cache.get(key) == 'result'
        time.sleep(0.15)
        assert cache.get(key) is None

    def test_clear(self):
        cache = PromptCache()
        cache.set('k', 'v')
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0

    def test_stats(self):
        cache = PromptCache(max_size=50, ttl_seconds=1800)
        stats = cache.stats()
        assert stats['max_size'] == 50
        assert stats['ttl_seconds'] == 1800

    def test_thread_safety(self):
        cache = PromptCache(max_size=100)
        import threading

        def writer():
            for i in range(50):
                cache.set(f'k{i}', f'v{i}')

        def reader():
            for i in range(50):
                cache.get(f'k{i}')

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Should not raise any exceptions


# ============== SecurityMonitor ==============


class TestSecurityMonitor:
    def test_initial_state(self):
        m = SecurityMonitor()
        assert m.get_call_count('tool1') == 0
        assert m.is_blocked('tool1') is False

    def test_record_call_counts(self):
        m = SecurityMonitor(window_seconds=10)
        for _ in range(5):
            m.record_call('tool1')
        assert m.get_call_count('tool1') == 5

    def test_alert_on_threshold(self):
        m = SecurityMonitor(window_seconds=10, max_calls_per_tool=3)
        m.record_call('tool1')
        m.record_call('tool1')
        alert = m.record_call('tool1')
        assert alert is not None
        assert 'tool1' in alert
        assert '3' in alert

    def test_no_alert_below_threshold(self):
        m = SecurityMonitor(window_seconds=10, max_calls_per_tool=5)
        for _ in range(4):
            assert m.record_call('tool1') is None

    def test_block_unblock(self):
        m = SecurityMonitor()
        m.block_tool('dangerous_tool')
        assert m.is_blocked('dangerous_tool') is True
        m.unblock_tool('dangerous_tool')
        assert m.is_blocked('dangerous_tool') is False

    def test_events_recorded(self):
        m = SecurityMonitor(window_seconds=10, max_calls_per_tool=2)
        m.record_call('t1')
        m.record_call('t1')
        events = m.get_events()
        assert len(events) == 1
        assert events[0]['event_type'] == 'tool_rate_exceeded'

    def test_events_filtered_by_severity(self):
        m = SecurityMonitor()
        m.block_tool('t1')  # generates 'critical'
        m.get_events(severity='critical')
        critical = m.get_events(severity='critical')
        assert len(critical) == 1
        info = m.get_events(severity='info')
        assert len(info) == 0

    def test_summary(self):
        m = SecurityMonitor(window_seconds=60, max_calls_per_tool=10)
        m.record_call('tool1')
        s = m.summary()
        assert s['window_seconds'] == 60
        assert s['max_calls_per_tool'] == 10
        assert 'tool1' in s['active_tools']

    def test_reset(self):
        m = SecurityMonitor()
        m.record_call('tool1')
        m.block_tool('tool2')
        m.reset()
        assert m.get_call_count('tool1') == 0
        assert m.is_blocked('tool2') is False

    def test_window_sliding(self):
        m = SecurityMonitor(window_seconds=0.1, max_calls_per_tool=3)
        m.record_call('t1')
        m.record_call('t1')
        time.sleep(0.15)
        # Window expired, count should be 0
        assert m.get_call_count('t1') == 0
        # Should not alert since old calls expired
        alert = m.record_call('t1')
        assert alert is None


class TestSecurityEvent:
    def test_to_dict(self):
        e = SecurityEvent('test_event', 'warning', 'tool1', 'some details')
        d = e.to_dict()
        assert d['event_type'] == 'test_event'
        assert d['severity'] == 'warning'
        assert d['tool_name'] == 'tool1'
        assert d['details'] == 'some details'
        assert 'timestamp' in d

    def test_details_truncated(self):
        e = SecurityEvent('test', 'info', details='x' * 1000)
        assert len(e.to_dict()['details']) <= 500
