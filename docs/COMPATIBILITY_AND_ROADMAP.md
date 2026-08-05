# Compatibility and Roadmap

## Compatibility

This describes the current features from each provider that InteropRouter supports or does not support. This list is not exhaustive. No means not currently planned.

For a list of models that can be used as bare ids (without a provider prefix), see `SupportedModel` in [types.py](../src/interop_router/types.py).
You can also pass `provider/model` (for example `openai/gpt-5.6-terra`) to select a provider explicitly.
`chat_completions` has no bare-id catalog; always use `chat_completions/<api_model_id>` (the id after the first `/` may itself contain `/`).

| Feature | OpenAI | Gemini | Anthropic | Chat Completions |
|---------|--------|--------|-----------|------------------|
| Reasoning* | Yes | Yes | Yes | Partial (`reasoning.effort` only; endpoint-dependent) |
| Image Understanding | Yes | Yes | Yes | Yes |
| Tool Calling with Reasoning | Yes | Yes | Yes | Function tools only |
| Built-in Web Search and Web Fetch| Yes | Yes | Yes | No (hosted tools are skipped) |
| Image Generation Tool | Yes (gpt-image variants) | Yes (Nano Banana variants) | No, Anthropic does not have an image generation model | No |
| Structured Outputs | Planned | Planned | Planned | No |
| Citations | Planned | Planned | Planned | No |
| Other Built-in Tools (Code execution, file search, etc) | TBD | TBD | TBD | No |
| Audio Model Support | No | No | N/A | No |
| Video Generation Model Support | No | No | N/A | No |
| Streaming Support | Yes | Yes | Yes | Yes |
| Token Counting | Yes | Yes | Yes | No |

* First-class providers encrypt or do not allow reasoning to be modified. As such InteropRouter cannot and does not use any non-native reasoning content when switching providers. The `chat_completions` adapter drops prior-turn reasoning items when converting history.


## Known Issues

### Gemini Calls Incorrect Tools when Interoperating with Built-in Web Search

**Affected providers**: Gemini
**Description:** When there are web search results from other providers in the message history, Gemini may either generate tool calls that were not provided, or fail outright.
**Workaround:** Usually the tool is called "web_search" so it could be implemented manually. In an agent loop, filter out this tool call.


## Other Planned Features

- Creating a high-quality llms.txt.
