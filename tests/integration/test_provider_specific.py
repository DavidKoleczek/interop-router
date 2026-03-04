import os

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
from openai.types.responses.tool_param import ImageGeneration
import pytest

from interop_router.router import Router
from interop_router.types import ChatMessage, SupportedModelAnthropic


@pytest.fixture
def router() -> Router:
    router = Router()
    router.register("openai", AsyncOpenAI())
    router.register("gemini", genai.Client(api_key=os.getenv("GEMINI_API_KEY")))
    router.register("anthropic", AsyncAnthropic())
    return router


# region: Anthropic models

ANTHROPIC_MODELS: list[SupportedModelAnthropic] = ["claude-opus-4-6", "claude-sonnet-4-6"]


@pytest.mark.parametrize("model", ANTHROPIC_MODELS)
async def test_adaptive_thinking_medium(router: Router, model: SupportedModelAnthropic) -> None:
    """Adaptive thinking with medium effort."""
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="What is 27 * 43? Think step by step.")),
    ]

    response = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )

    assert response is not None
    assert response.output


@pytest.mark.parametrize("model", ANTHROPIC_MODELS)
async def test_adaptive_thinking_multi_turn(router: Router, model: SupportedModelAnthropic) -> None:
    """Multi-turn conversation with adaptive thinking preserves thinking blocks."""
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="What is the sum of the first 10 prime numbers? Reason, but just write the number in the end.",
            )
        ),
    ]

    response = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )

    assert response is not None
    assert response.output

    messages.extend(response.output)
    messages.append(
        ChatMessage(
            message=EasyInputMessageParam(
                role="user", content="Now double that result. Reason, but just write the number in the end."
            ),
        )
    )
    response2 = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "xhigh", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    assert response2 is not None
    assert response2.output


@pytest.mark.parametrize("model", ANTHROPIC_MODELS)
async def test_no_reasoning(router: Router, model: SupportedModelAnthropic) -> None:
    """Works without reasoning (thinking disabled)."""
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="Say hello in one word.")),
    ]

    response = await router.create(
        model=model,
        input=messages,
    )

    assert response is not None
    assert response.output


async def test_prompt_caching(router: Router) -> None:
    """Automatic prompt caching via provider_kwargs produces cache read tokens on the second turn."""
    cache_control = {"cache_control": {"type": "ephemeral"}}
    model: SupportedModelAnthropic = "claude-sonnet-4-6"

    # Minimum cacheable length for Sonnet 4.6 is 2048 tokens; use a long system prompt to exceed it.
    padding = "word " * 2200
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(role="system", content=f"You are a helpful assistant. Context: {padding}")
        ),
        ChatMessage(message=EasyInputMessageParam(role="user", content="Say hello in one word.")),
    ]

    response1 = await router.create(model=model, input=messages, provider_kwargs=cache_control)
    assert response1.output

    messages.extend(response1.output)
    messages.append(
        ChatMessage(message=EasyInputMessageParam(role="user", content="Now say goodbye in one word.")),
    )

    response2 = await router.create(model=model, input=messages, provider_kwargs=cache_control)
    assert response2.output
    assert response2.usage is not None
    assert response2.usage.input_tokens_details is not None
    assert response2.usage.input_tokens_details.cached_tokens > 0


# endregion

# region: Gemini models


async def test_image_gen_with_thinking(router: Router) -> None:
    """Image generation with thinking levels using gemini-3.1-flash-image-preview."""
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Create a picture of a nano banana dish in a fancy restaurant with an interop theme?",
            ),
            provider_kwargs={"gemini": {"image_config": {"aspect_ratio": "4:3", "image_size": "1K"}}},
        )
    ]

    image_tool = ImageGeneration(
        type="image_generation",
        model="gemini-3.1-flash-image-preview",
    )

    response = await router.create(
        input=messages,
        model="gemini-3-flash-preview",
        tools=[image_tool],
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    assert response is not None
    assert response.output


# endregion
