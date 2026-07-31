# Batch Processing

Cost-effective batch code review using Anthropic Batch API.

## What it demonstrates

- Using `AnthropicBatchClient` for batch processing
- Submitting multiple `BatchRequest` items
- Polling for results
- 50% cost savings vs standard API

## Prerequisites

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Running

```bash
python examples/batch_processing/run.py
```

## Batch API benefits

- **50% cost reduction** compared to standard API
- Process up to **10,000 requests** per batch
- Results within **24 hours**
- Ideal for:
  - Code review
  - Test generation
  - Documentation writing
  - Refactoring suggestions

## Expected output

```
Submitting batch of 3 requests...
Batch ID: batch_abc123...
Processing (this may take up to 24 hours for large batches)...

Received 3 results:

--- review_1 ---
Tokens: 45 in, 128 out
Review:
The `add` function is simple and correct. It safely adds two numbers...

--- review_2 ---
Tokens: 42 in, 156 out
Review:
The `divide` function has a potential ZeroDivisionError bug...
```

## Production usage

For production, use webhooks instead of polling:

```python
batch_id = client.submit(requests, webhook_url='https://your-api.com/batch-callback')
```
