"""Tests for token counting, verifying count_tokens results against actual API usage."""

import base64
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam, ResponseInputImageParam, ResponseInputTextParam
from openai.types.responses.response_input_item_param import Message
import pytest

from interop_router.router import Router
from interop_router.types import ChatMessage, ProviderName, SupportedModel

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
