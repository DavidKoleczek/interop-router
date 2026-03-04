# Provider Guide

This document providers an overview of provider-specific choices that were made.

## Provider-Specific Parameters

`Router.create()` has `provider_kwargs` parameter for passing provider-specific keyword arguments. The supported parameters for each provider are listed below.

### Anthropic

#### `cache_control`

Enables [automatic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
Set to `{"type": "ephemeral"}` to cache the last cacheable block in the request.
Cached tokens are reported in `response.usage.input_tokens_details.cached_tokens`.

```python
response = await router.create(
    input=messages,
    model="claude-sonnet-4-6",
    provider_kwargs={"cache_control": {"type": "ephemeral"}},
)
```
