"""Tests for token counting, verifying count_tokens results against actual API usage."""

import base64
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam, ResponseInputImageParam, ResponseInputTextParam
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response_input_item_param import FunctionCallOutput, Message
from openai.types.shared_params.reasoning import Reasoning
import pytest

from interop_router.router import Router
from interop_router.types import ChatMessage, ProviderName, SupportedModel

FUNCTION_TOOLS: list[FunctionToolParam] = [
    FunctionToolParam(
        type="function",
        name="get_weather",
        description="Get the current weather for a given location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. San Francisco, USA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use.",
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
        strict=True,
    ),
    FunctionToolParam(
        type="function",
        name="get_stock_price",
        description="Get the current stock price for a given ticker symbol.",
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. AAPL, GOOGL",
                },
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
        strict=True,
    ),
]

TOKEN_COUNT_PROVIDER_MODEL_PARAMS = [
    pytest.param("openai", "gpt-5.2"),
    pytest.param("gemini", "gemini-3-flash-preview"),
    pytest.param("anthropic", "claude-haiku-4-5-20251001"),
]


def get_client(provider: ProviderName) -> AsyncOpenAI | genai.Client | AsyncAnthropic:
    if provider == "openai":
        return AsyncOpenAI()
    if provider == "gemini":
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    if provider == "anthropic":
        return AsyncAnthropic()
    raise ValueError(f"Unknown provider: {provider}")


@pytest.fixture
def router() -> Router:
    return Router()


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_simple(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that count_tokens returns equal to input_tokens reported by create."""
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="What is the capital of France?")),
    ]

    token_count = await router.count_tokens(
        input=messages,
        model=model,
    )

    response = await router.create(
        input=messages,
        model=model,
        max_output_tokens=1024,
    )

    assert response.usage is not None
    assert response.usage.input_tokens == token_count, (
        f"Expected token count {token_count} to match actual input tokens {response.usage.input_tokens}"
    )


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_with_instructions(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that count_tokens accounts for system instructions in addition to messages."""
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="What is the capital of France?")),
    ]
    instructions = "You are a helpful geography assistant. Answer concisely in one sentence."

    token_count = await router.count_tokens(
        input=messages,
        model=model,
        instructions=instructions,
    )

    response = await router.create(
        input=messages,
        model=model,
        instructions=instructions,
        max_output_tokens=1024,
    )

    assert response.usage is not None
    assert response.usage.input_tokens == token_count, (
        f"Expected token count {token_count} to match actual input tokens {response.usage.input_tokens}"
    )


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_multi_turn_with_system(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that count_tokens handles a system message with multiple user/assistant turns."""
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(message=EasyInputMessageParam(role="system", content="You are a helpful geography assistant.")),
        ChatMessage(message=EasyInputMessageParam(role="user", content="What is the capital of France?")),
        ChatMessage(message=EasyInputMessageParam(role="assistant", content="The capital of France is Paris.")),
        ChatMessage(message=EasyInputMessageParam(role="user", content="And what about Germany?")),
    ]

    token_count = await router.count_tokens(
        input=messages,
        model=model,
    )

    response = await router.create(
        input=messages,
        model=model,
        max_output_tokens=1024,
    )

    assert response.usage is not None
    assert response.usage.input_tokens == token_count, (
        f"Expected token count {token_count} to match actual input tokens {response.usage.input_tokens}"
    )


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_system_message_and_instructions(
    router: Router, provider: ProviderName, model: SupportedModel
):
    """Verify token counting with both a system message and instructions alongside a multi-turn conversation."""
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(message=EasyInputMessageParam(role="system", content="You are a helpful geography assistant.")),
        ChatMessage(message=EasyInputMessageParam(role="user", content="What is the capital of France?")),
        ChatMessage(message=EasyInputMessageParam(role="assistant", content="The capital of France is Paris.")),
        ChatMessage(
            message=EasyInputMessageParam(role="user", content="Tell me one interesting fact about that city.")
        ),
        ChatMessage(
            message=EasyInputMessageParam(
                role="assistant",
                content="Paris was originally a Roman city called Lutetia.",
            )
        ),
        ChatMessage(message=EasyInputMessageParam(role="user", content="Now do the same for Berlin.")),
    ]
    instructions = "Reply in one sentence maximum."

    token_count = await router.count_tokens(
        input=messages,
        model=model,
        instructions=instructions,
    )

    response = await router.create(
        input=messages,
        model=model,
        instructions=instructions,
        max_output_tokens=1024,
    )

    assert response.usage is not None
    assert response.usage.input_tokens == token_count, (
        f"Expected token count {token_count} to match actual input tokens {response.usage.input_tokens}"
    )


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_with_image(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that count_tokens correctly accounts for image inputs."""
    client = get_client(provider)
    router.register(provider, client)

    image_path = Path(__file__).parents[1] / "integration" / "data" / "landscape.png"
    image_bytes = image_path.read_bytes()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{base64_image}"

    messages = [
        ChatMessage(message=EasyInputMessageParam(role="system", content="Reply in a few words only.")),
        ChatMessage(
            message=Message(
                role="user",
                content=[
                    ResponseInputTextParam(type="input_text", text="What is in this image?"),
                    ResponseInputImageParam(type="input_image", image_url=data_url, detail="auto"),
                ],
            )
        ),
    ]

    token_count = await router.count_tokens(
        input=messages,
        model=model,
    )

    response = await router.create(
        input=messages,
        model=model,
        max_output_tokens=1024,
    )

    assert response.usage is not None
    assert response.usage.input_tokens == token_count, (
        f"Expected token count {token_count} to match actual input tokens {response.usage.input_tokens}"
    )


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_large_input(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that count_tokens handles a very large input (~1M tokens)."""
    client = get_client(provider)
    router.register(provider, client)

    # Aiming to be over 1 million tokens
    large_text = "The quick brown fox jumps over the lazy dog. " * 150_000

    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content=large_text)),
    ]

    token_count = await router.count_tokens(
        input=messages,
        model=model,
    )

    assert token_count > 200_000, f"Expected at least 200k tokens for large input, got {token_count}"


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_with_tools(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that count_tokens correctly accounts for tool definitions."""
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(role="user", content="What is the weather in San Francisco in celsius?")
        ),
    ]
    instructions = "Use the available tools to answer the user's question."

    token_count = await router.count_tokens(
        input=messages,
        model=model,
        instructions=instructions,
        tools=FUNCTION_TOOLS,
    )

    response = await router.create(
        input=messages,
        model=model,
        instructions=instructions,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        max_output_tokens=1024,
    )

    assert response.usage is not None
    if provider == "gemini":
        # Gemini Developer API countTokens does not support tool definitions in the request,
        # so the count will be strictly less than the actual input tokens.
        assert token_count < response.usage.input_tokens, (
            f"Expected token count {token_count} to be less than actual input tokens {response.usage.input_tokens}"
        )
    else:
        assert response.usage.input_tokens == token_count, (
            f"Expected token count {token_count} to match actual input tokens {response.usage.input_tokens}"
        )


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_with_tool_call_history(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that count_tokens handles a multi-turn conversation containing tool calls and tool outputs."""
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="What's the weather in Tokyo? Use celsius. Also, what's the stock price of AAPL?",
            )
        ),
        ChatMessage(
            message={
                "type": "function_call",
                "call_id": "call_weather_1",
                "name": "get_weather",
                "arguments": '{"location": "Tokyo, Japan", "unit": "celsius"}',
            },
            created_by="openai",
        ),
        ChatMessage(
            message={
                "type": "function_call",
                "call_id": "call_stock_1",
                "name": "get_stock_price",
                "arguments": '{"ticker": "AAPL"}',
            },
            created_by="openai",
        ),
        ChatMessage(
            message=FunctionCallOutput(
                call_id="call_weather_1",
                output='{"temperature": 22, "unit": "celsius", "conditions": "sunny"}',
                type="function_call_output",
            ),
        ),
        ChatMessage(
            message=FunctionCallOutput(
                call_id="call_stock_1",
                output='{"ticker": "AAPL", "price": 178.50, "currency": "USD"}',
                type="function_call_output",
            ),
        ),
        ChatMessage(
            message=EasyInputMessageParam(
                role="assistant",
                content="The weather in Tokyo is 22 degrees celsius and sunny. AAPL is currently at $178.50 USD.",
            ),
            created_by="openai",
        ),
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="What about the weather in London in fahrenheit?",
            ),
        ),
    ]
    instructions = "Use the available tools to answer the user's question."

    token_count = await router.count_tokens(
        input=messages,
        model=model,
        instructions=instructions,
        tools=FUNCTION_TOOLS,
    )

    response = await router.create(
        input=messages,
        model=model,
        instructions=instructions,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        max_output_tokens=1024,
    )

    assert response.usage is not None
    if provider == "gemini":
        # Gemini Developer API countTokens does not support tool definitions in the request,
        # so the count will be strictly less than the actual input tokens.
        assert token_count < response.usage.input_tokens, (
            f"Expected token count {token_count} to be less than actual input tokens {response.usage.input_tokens}"
        )
    else:
        assert response.usage.input_tokens == token_count, (
            f"Expected token count {token_count} to match actual input tokens {response.usage.input_tokens}"
        )


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_with_reasoning(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that count_tokens accounts for reasoning in a multi-turn conversation."""
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="What is 25 * 37?")),
    ]
    reasoning: Reasoning = {"effort": "low", "summary": "auto"}

    # First turn: generate a response with reasoning to get authentic encrypted content.
    response1 = await router.create(
        input=messages,
        model=model,
        reasoning=reasoning,
        include=["reasoning.encrypted_content"],
        max_output_tokens=16_000,
    )
    assert response1.usage is not None

    # Build multi-turn conversation with the reasoning output, then ask a follow-up.
    messages.extend(response1.output)
    messages.append(
        ChatMessage(message=EasyInputMessageParam(role="user", content="Now multiply that result by 2.")),
    )

    token_count = await router.count_tokens(
        input=messages,
        model=model,
        reasoning=reasoning,
    )

    response2 = await router.create(
        input=messages,
        model=model,
        reasoning=reasoning,
        include=["reasoning.encrypted_content"],
        max_output_tokens=16_000,
    )

    assert response2.usage is not None
    assert response2.usage.input_tokens == token_count, (
        f"Expected token count {token_count} to match actual input tokens {response2.usage.input_tokens}"
    )


@pytest.mark.parametrize(("provider", "model"), TOKEN_COUNT_PROVIDER_MODEL_PARAMS)
async def test_count_tokens_with_reasoning_and_tools(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that count_tokens accounts for reasoning combined with tool definitions and tool call history."""
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="What is the weather in New York in fahrenheit?",
            )
        ),
    ]
    reasoning: Reasoning = {"effort": "low", "summary": "auto"}

    # First turn: generate a response with reasoning + tools to get authentic reasoning output.
    response1 = await router.create(
        input=messages,
        model=model,
        reasoning=reasoning,
        include=["reasoning.encrypted_content"],
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        max_output_tokens=16_000,
    )
    assert response1.usage is not None

    # Build multi-turn: append model output (reasoning + function_call), add tool output, then follow-up.
    messages.extend(response1.output)
    for chat_message in response1.output:
        msg = chat_message.message
        if msg.get("type") == "function_call" and msg.get("name") == "get_weather":
            messages.append(
                ChatMessage(
                    message=FunctionCallOutput(
                        call_id=msg.get("call_id", ""),
                        output='{"temperature": 45, "unit": "fahrenheit", "conditions": "cloudy"}',
                        type="function_call_output",
                    ),
                )
            )

    messages.append(
        ChatMessage(message=EasyInputMessageParam(role="user", content="And what about London in celsius?")),
    )

    token_count = await router.count_tokens(
        input=messages,
        model=model,
        reasoning=reasoning,
        tools=FUNCTION_TOOLS,
    )

    response2 = await router.create(
        input=messages,
        model=model,
        reasoning=reasoning,
        include=["reasoning.encrypted_content"],
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        max_output_tokens=16_000,
    )

    assert response2.usage is not None
    if provider == "gemini":
        # Gemini Developer API countTokens does not support tool definitions in the request,
        # so the count will be strictly less than the actual input tokens.
        assert token_count < response2.usage.input_tokens, (
            f"Expected token count {token_count} to be less than actual input tokens {response2.usage.input_tokens}"
        )
    else:
        assert response2.usage.input_tokens == token_count, (
            f"Expected token count {token_count} to match actual input tokens {response2.usage.input_tokens}"
        )
