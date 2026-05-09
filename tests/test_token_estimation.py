from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from anvil.token_estimation import (
    CHARS_PER_TOKEN_DEFAULT,
    CHARS_PER_TOKEN_JSON,
    IMAGE_TOKENS_ESTIMATE,
    HybridTokenCounter,
    TokenUsage,
    _is_json_like,
    estimate_content_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_text_tokens,
    estimate_tokens,
    extract_usage,
)


class EstimateTextTokensTests(unittest.TestCase):
    def test_should_estimate_plain_text(self) -> None:
        # 100 chars / 4 = 25 tokens
        self.assertEqual(estimate_text_tokens('a' * 100), 25)

    def test_should_estimate_json_with_lower_ratio(self) -> None:
        # 100 chars / 2 = 50 tokens
        self.assertEqual(estimate_text_tokens('a' * 100, is_json=True), 50)

    def test_should_return_zero_for_empty(self) -> None:
        self.assertEqual(estimate_text_tokens(''), 0)

    def test_should_return_at_least_one_for_nonempty(self) -> None:
        self.assertEqual(estimate_text_tokens('ab'), 1)


class IsJsonLikeTests(unittest.TestCase):
    def test_should_detect_json_object(self) -> None:
        self.assertTrue(_is_json_like('{"key": "value"}'))

    def test_should_detect_json_array(self) -> None:
        self.assertTrue(_is_json_like('[1, 2, 3]'))

    def test_should_detect_json_code_block(self) -> None:
        self.assertTrue(_is_json_like('```json\n{"a": 1}\n```'))

    def test_should_reject_plain_text(self) -> None:
        self.assertFalse(_is_json_like('hello world'))

    def test_should_reject_empty(self) -> None:
        self.assertFalse(_is_json_like(''))


class EstimateContentTokensTests(unittest.TestCase):
    def test_should_estimate_string_content(self) -> None:
        result = estimate_content_tokens('hello world')
        self.assertGreater(result, 0)

    def test_should_estimate_text_block(self) -> None:
        blocks = [{'type': 'text', 'text': 'hello world'}]
        result = estimate_content_tokens(blocks)
        self.assertGreater(result, 0)

    def test_should_estimate_image_block(self) -> None:
        blocks = [{'type': 'image', 'source': {}}]
        result = estimate_content_tokens(blocks)
        self.assertEqual(result, IMAGE_TOKENS_ESTIMATE)

    def test_should_estimate_tool_use_block(self) -> None:
        blocks = [{'type': 'tool_use', 'name': 'read_file', 'input': {'path': 'test.py'}}]
        result = estimate_content_tokens(blocks)
        self.assertGreater(result, 0)

    def test_should_estimate_thinking_block(self) -> None:
        blocks = [{'type': 'thinking', 'thinking': 'let me think about this'}]
        result = estimate_content_tokens(blocks)
        self.assertGreater(result, 0)

    def test_should_handle_none_content(self) -> None:
        self.assertEqual(estimate_content_tokens(None), 0)


class EstimateMessageTokensTests(unittest.TestCase):
    def test_should_estimate_user_message(self) -> None:
        msg = {'role': 'user', 'content': 'hello'}
        result = estimate_message_tokens(msg)
        self.assertGreater(result, 0)

    def test_should_estimate_assistant_message_with_tool_calls(self) -> None:
        msg = {
            'role': 'assistant',
            'content': 'I will read the file',
            'tool_calls': [{'function': {'name': 'read_file', 'arguments': '{"path": "test.py"}'}}],
        }
        result = estimate_message_tokens(msg)
        self.assertGreater(result, 0)

    def test_should_include_message_overhead(self) -> None:
        msg = {'role': 'user', 'content': ''}
        result = estimate_message_tokens(msg)
        # Should include at least ROLE_OVERHEAD + MESSAGE_OVERHEAD
        self.assertGreater(result, 10)


class EstimateMessagesTokensTests(unittest.TestCase):
    def test_should_estimate_multiple_messages(self) -> None:
        messages = [
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'hi there'},
        ]
        result = estimate_messages_tokens(messages)
        self.assertGreater(result, 0)

    def test_should_return_zero_for_empty(self) -> None:
        self.assertEqual(estimate_messages_tokens([]), 0)


class BackwardCompatibleEstimateTokensTests(unittest.TestCase):
    def test_should_work_like_original(self) -> None:
        # Original: total_chars // 4
        result = estimate_tokens(['a' * 100])
        self.assertEqual(result, 25)

    def test_should_handle_multiple_parts(self) -> None:
        result = estimate_tokens(['a' * 40, 'b' * 40])
        self.assertEqual(result, 20)

    def test_should_return_zero_for_empty(self) -> None:
        self.assertEqual(estimate_tokens([]), 0)
        self.assertEqual(estimate_tokens(['', '']), 0)


class ExtractUsageTests(unittest.TestCase):
    def test_should_extract_usage(self) -> None:
        response = {'usage': {'input_tokens': 100, 'output_tokens': 50}}
        usage = extract_usage(response)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.total_tokens, 150)

    def test_should_handle_missing_usage(self) -> None:
        usage = extract_usage({})
        self.assertEqual(usage.input_tokens, 0)

    def test_should_handle_none_values(self) -> None:
        response = {'usage': {'input_tokens': None, 'output_tokens': None}}
        usage = extract_usage(response)
        self.assertEqual(usage.input_tokens, 0)


class HybridTokenCounterTests(unittest.TestCase):
    def test_should_use_heuristic_initially(self) -> None:
        counter = HybridTokenCounter()
        self.assertFalse(counter.has_api_data)
        messages = [{'role': 'user', 'content': 'a' * 400}]
        result = counter.estimate_messages(messages)
        # 400/4 + role_overhead(4) + role_text(1) + msg_overhead(10) = 115
        self.assertEqual(result, 115)

    def test_should_calibrate_from_api_response(self) -> None:
        counter = HybridTokenCounter()
        counter.update_from_response(
            {'usage': {'input_tokens': 200, 'output_tokens': 50}},
            message_count=2,
        )
        self.assertTrue(counter.has_api_data)

    def test_should_use_calibrated_estimate(self) -> None:
        counter = HybridTokenCounter()
        # Simulate: 1000 chars of content → 200 actual tokens → chars_per_token = 5
        counter.update_from_response(
            {'usage': {'input_tokens': 200, 'output_tokens': 50}},
            message_count=1,
        )
        messages = [{'role': 'user', 'content': 'b' * 1000}]
        result = counter.estimate_messages(messages)
        # Should be close to 200 (calibrated), not 250 (heuristic)
        self.assertGreater(result, 150)
        self.assertLess(result, 300)


if __name__ == '__main__':
    unittest.main()
