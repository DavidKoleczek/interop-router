# Provider Guide

This document provides an overview of provider-specific choices that were made.

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
