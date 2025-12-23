<h1 align="center">
    InteropRouter
</h1>
<p align="center">
    <p align="center">Seamlessly call major LLMs and image generation models through a unified interface.
    </p>
</p>
<p align="center">
    <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
    <a href="https://github.com/astral-sh/ty"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" alt="ty"></a>
    <a href="https://pypi.org/project/interop-router/"><img src="https://img.shields.io/pypi/v/interop-router" alt="PyPI"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

InteropRouter is designed to seamlessly interoperate between the most common AI providers at a high level of quality. 
It uses the [OpenAI Responses API](https://platform.openai.com/docs/guides/migrate-to-responses) types as a common denominator for inputs and outputs, allowing you to switch between providers with minimal code changes.
See [examples/](examples/) for detailed notebooks covering interoperability, function calling, image generation, and more.

## Getting Started

### Installation

```bash
# With uv.
uv add interop-router

# With pip.
pip install interop-router
```

### Usage

```python
from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam

from interop_router.router import Router
from interop_router.types import ChatMessage

router = Router()
router.register("openai", AsyncOpenAI())
router.register("gemini", genai.Client())
router.register("anthropic", AsyncAnthropic())

# InteropRouter is strictly typed, so be sure to use OpenAI's Response types for inputs. 
# See https://platform.openai.com/docs/guides/migrate-to-responses and the library source for more details on typing.
messages = [ChatMessage(message=EasyInputMessageParam(role="user", content="Hello!"))]

response = await router.create(input=messages, model="gpt-5.2")
response = await router.create(input=messages, model="gemini-3-flash-preview")
response = await router.create(input=messages, model="claude-sonnet-4-5-20250929")
```

### InteropRouter Design Philosophy

The only goal of InteropRouter is to interoperate between the most common AI providers. To make this goal achievable, we make several trade-offs:
- Only support OpenAI (including Azure OpenAI), Gemini, and Anthropic. Each provider adds a significant amount of possible permutations of features. To maintain high-quality interoperability, we limit the number of providers.
- Only support async APIs, but not streaming token by token as this adds significant complexity, and for agents the latency is not as important.
- We do not support stateful features where possible. These features are contradictory to the goal of seamless swapping between providers.
- We choose the OpenAI Responses API types as the common denominator for creating pivots between providers. The reason is two-fold: a) The Responses API supports most features b) By picking an existing API, we avoid the need to design and maintain our own schema and Responses API support is gained for "free".
- The supported features will be rigorously tested to ensure seamless swapping between providers within a single conversation.


## Development

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [prek](https://github.com/j178/prek/blob/master/README.md#installation)

### Setup

Create uv virtual environment and install dependencies:

```bash
uv sync --frozen --all-extras --all-groups
```

Set up git hooks:

```bash
prek install
```

To update dependencies (updates the lock file):

```bash
uv sync --all-extras --all-groups
```

Run formatting, linting, type checking, and tests in one command:

```bash
uv run ruff format && uv run ruff check --fix && uv run ty check && uv run pytest
```

### Further Information

[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)


## Compatibility and Roadmap

[docs/COMPATIBILITY_AND_ROADMAP.md](docs/COMPATIBILITY_AND_ROADMAP.md)
