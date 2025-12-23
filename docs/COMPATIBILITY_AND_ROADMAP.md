# Compatibility and Roadmap

## Compatibility

This describes the current features from each provider that InteropRouter supports or does not support. This list is not exhaustive. No means not currently planned.

For a list of supported models, see `SupportedModel` in [types.py](../src/interop_router/types.py)

| Feature | OpenAI | Gemini | Anthropic |
|---------|--------|--------|-----------|
| Reasoning* | Yes | Yes | Yes |
| Image Understanding | Yes | Yes | Yes |
| Tool Calling with Reasoning | Yes | Yes | Yes |
| Built-in Web Search and Web Fetch| Yes | Yes | Yes |
| Image Generation Tool | Yes (gpt-image variants) | Yes (Nano Banana variants) | No, Anthropic does not have an image generation model |
| Structured Outputs | Planned | Planned | Planned |
| Citations | Planned | Planned | Planned |
| Other Built-in Tools (Code execution, file search, etc) | TBD | TBD | TBD |
| Audio Model Support | No | No | N/A |
| Video Generation Model Support | No | No | N/A |
| Streaming Support | No | No | No |

* All providers encrypt or do not allow reasoning to be modified. As such InteropRouter cannot and does not use any non-native reasoning content when switching providers.


## Known Issues

### Gemini Calls Incorrect Tools when Interoperating with Built-in Web Search

**Affected providers**: Gemini
**Description:** When there are web search results from other providers in the message history, Gemini may either generate tool calls that were not provided, or fail outright.
**Workaround:** Usually the tool is called "web_search" so it could be implemented manually. In an agent loop, filter out this tool call.


## Other Planned Features

- Creating a high-quality llms.txt.
