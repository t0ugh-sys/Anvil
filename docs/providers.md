# LLM Providers

Anvil supports multiple LLM providers through a unified `InvokeFn` interface.
All provider setup goes through `anvil.llm.providers` (a backward-compat re-export
of the `anvil/llm/` subpackage).

## Supported Providers

| Provider | ID | Module |
|---|---|---|
| Anthropic (Claude) | `anthropic` | `anvil/llm/anthropic/` |
| Google Gemini | `gemini` | `anvil/llm/gemini.py` |
| OpenAI-compatible | `openai` / custom base URL | `anvil/llm/openai_compat.py` |
| Mock | `mock` | `anvil/llm/mock.py` |

## Selecting a Provider

Pass `--provider` and `--model` at the command line, or set them in a config file:

```bash
anvil --provider anthropic --model claude-opus-4-5
anvil --provider gemini --model gemini-1.5-pro
anvil --provider openai --model gpt-4o --base-url https://api.openai.com/v1
```

The factory function `build_invoke_from_args(args, mode='coding')` resolves the
provider from `args.provider` and returns a ready-to-call `InvokeFn`.

## Anthropic

`_anthropic_invoke_factory` builds an invoke closure with these capabilities:

- **Prompt caching** (`enable_prompt_caching=True` by default): the system prompt
  is sent with `cache_control: {"type": "ephemeral"}` so Anthropic caches it on
  their side, cutting repeated-turn costs significantly.
- **Extended thinking** (`thinking_budget_tokens > 0`): enables Claude's internal
  chain-of-thought. Forces `temperature=1.0` and adjusts `max_tokens` to satisfy
  the API requirement that `max_tokens > budget_tokens`.
- **Native tool use** (`enable_native_tools=True`): sends tools via the Anthropic
  tool-use API rather than injecting them into the prompt.
- **Stop sequences**: pass a list of strings to halt generation early.
- **Token tracking**: pass a `TokenUsageTracker` to accumulate per-request usage
  including cache hit/miss statistics.
- **Streaming**: `anthropic_stream_invoke_factory` streams the response and
  reassembles it; useful for long outputs.
- **Batch API**: `AnthropicBatchClient` submits batches of `BatchRequest` objects
  and polls for results, saving ~50 % on large workloads.

### Token counting

```python
from anvil.llm.providers import anthropic_count_tokens, anthropic_count_tokens_or_estimate

count = anthropic_count_tokens(prompt, api_key=key, model=model)
# Falls back to a fast heuristic if the API call fails:
count = anthropic_count_tokens_or_estimate(prompt, api_key=key, model=model)
```

## Gemini

`gemini_invoke_factory` / `_gemini_invoke_factory` — calls the Gemini REST API.
Requires a `GEMINI_API_KEY` environment variable or `--api-key`.

## OpenAI-compatible

`openai_compatible_chat_invoke_factory` / `_openai_compatible_invoke_factory` —
sends a standard `/v1/chat/completions` request.  Works with OpenAI, Azure OpenAI,
Groq, Together, Mistral, Ollama (local), and any compatible endpoint.

```bash
anvil --provider openai --base-url http://localhost:11434/v1 --model llama3
```

## Mock

`_mock_invoke_factory` returns a deterministic string without making any network
calls.  Used in unit tests.  Summarizers built from a mock provider are disabled
(return `None`).

## Usage & Cost Tracking

```python
from anvil.llm.usage import TokenUsageTracker, CostTracker

tracker = TokenUsageTracker()
# pass tracker= to _anthropic_invoke_factory
print(tracker.cache_hit_rate())        # float 0.0–1.0
print(tracker.estimated_cost_savings())
print(tracker.summary())               # human-readable dict
```

`CostTracker` reads per-model pricing from `anvil/llm/pricing.json` and adds
`.total_cost` and `.savings` properties.  Pricing can be inspected or updated at
runtime with the `/pricing` slash command.

## Prompt Cache (Client-Side LRU)

`PromptCache` in `anvil/llm/cache.py` is a separate client-side LRU cache that
stores full `(model, prompt) → response` pairs keyed by `sha256[:16]` with a
3 600 s TTL.  It is distinct from Anthropic's server-side prompt caching.
