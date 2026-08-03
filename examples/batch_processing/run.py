"""
Batch Processing — cost-effective batch code review using Anthropic Batch API.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/batch_processing/run.py

Batch API benefits:
- 50% cost savings vs standard API
- Process up to 10,000 requests per batch
- Results within 24 hours
- Ideal for code review, test generation, documentation
"""

from __future__ import annotations

import os
import sys
import time

from anvil.llm.anthropic.batch import AnthropicBatchClient, BatchRequest


def main() -> None:
    api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if not api_key:
        print('Error: Set ANTHROPIC_API_KEY environment variable', file=sys.stderr)
        sys.exit(1)

    client = AnthropicBatchClient(api_key=api_key, model='claude-sonnet-5')

    # Submit batch of code review requests
    requests = [
        BatchRequest(
            custom_id='review_1',
            prompt='Review this Python function for bugs:\n\ndef add(a, b):\n    return a + b',
            max_tokens=2048,
        ),
        BatchRequest(
            custom_id='review_2',
            prompt='Review this function:\n\ndef divide(a, b):\n    return a / b',
            max_tokens=2048,
        ),
        BatchRequest(
            custom_id='review_3',
            prompt='Suggest improvements:\n\ndef process_list(items):\n    result = []\n    for item in items:\n        result.append(item * 2)\n    return result',
            max_tokens=2048,
        ),
    ]

    print(f'Submitting batch of {len(requests)} requests...')
    batch_id = client.submit(requests)
    print(f'Batch ID: {batch_id}')
    print('Processing (this may take up to 24 hours for large batches)...')

    # Poll for results (in production, use webhooks or check periodically)
    max_polls = 60
    for i in range(max_polls):
        results = client.get_results(batch_id)
        if results:
            break
        if i < max_polls - 1:
            time.sleep(10)
    else:
        print('Batch still processing. Check back later with:')
        print(f'  client.get_results("{batch_id}")')
        sys.exit(0)

    print(f'\nReceived {len(results)} results:')
    for result in results:
        print(f'\n--- {result.custom_id} ---')
        if result.ok:
            print(f'Tokens: {result.input_tokens} in, {result.output_tokens} out')
            print(f'Review:\n{result.text[:200]}...')
        else:
            print(f'Error: {result.error}')


if __name__ == '__main__':
    main()
