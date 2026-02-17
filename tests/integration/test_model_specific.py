import os

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
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
                content="What is the sum of the first 10 prime numbers? Just write the number.",
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
            message=EasyInputMessageParam(role="user", content="Now double that result. Just write the number."),
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


# endregion
