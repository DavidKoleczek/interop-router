# Provider Guide

This document provides an overview of provider-specific choices that were made.

## Registering Multiple Clients per Provider

`Router.register()` accepts an optional `name` that becomes the prefix in a model reference. It defaults to the provider name. Only the first `/` disambiguates, so the model id itself may contain `/`.

```python
router.register("anthropic", AsyncAnthropic())
router.register("chat_completions", vllm_client, name="vllm")
router.register("chat_completions", openrouter_client, name="openrouter")

"vllm/nvidia/Qwen3.6-27B-NVFP4"  # <name>/<model>, selects a named client
"anthropic/my-model"  # <provider>/<model>, selects that provider's only client
"claude-sonnet-5"  # <model>, a catalog id, selects that provider's only client
"chat_completions/some-model"  # ValueError: two clients registered, name one of them
"nvidia/Qwen3.6-27B-NVFP4"  # ValueError: unknown model, never falls back to a client
```

Names are matched before providers, cannot be empty or contain `/`, and cannot shadow a different provider's name.

## Provider-Specific Parameters

`Router.create()` has a `provider_kwargs` parameter for passing provider-specific keyword arguments. The supported parameters for each provider are listed below.

### Anthropic

#### `cache_control`

Enables [automatic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
Set to `{"type": "ephemeral"}` to cache the last cacheable block in the request.
Cached tokens are reported in `response.usage.input_tokens_details.cached_tokens`.

```python
response = await router.create(
    input=messages,
    model="claude-sonnet-5",
    provider_kwargs={"cache_control": {"type": "ephemeral"}},
)
```

### Chat Completions

`provider_kwargs` is forwarded as `extra_body` on `client.chat.completions.create`. Use this for vendor-specific fields that are not part of the standard Chat Completions request shape.

```python
response = await router.create(
    input=messages,
    model="chat_completions/some-model",
    provider_kwargs={"vendor_specific_field": value},
)
```

Only Responses `type="function"` tools are mapped. Hosted tools (web search, image generation, and similar) are skipped. Responses-only create parameters (`include`, `text`, `truncation`, `background`) are ignored.

Some gateways report upstream failures as HTTP 200 bodies containing an `error` object and no `choices`.
These are surfaced on `RouterResponse.error`: raw code 429 maps to `rate_limit_exceeded`, everything else to `server_error`. The raw error code and any provider `metadata` are included in the error message (for example `[502] Upstream error from ...`).
