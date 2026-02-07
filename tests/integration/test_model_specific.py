import os

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
import pytest

from interop_router.router import Router
from interop_router.types import ChatMessage


@pytest.fixture
def router() -> Router:
    router = Router()
    router.register("openai", AsyncOpenAI())
    router.register("gemini", genai.Client(api_key=os.getenv("GEMINI_API_KEY")))
    router.register("anthropic", AsyncAnthropic())
    return router


# region: Claude Opus 4.6


OPUS_4_6_MODEL = "claude-opus-4-6"


async def test_opus_4_6_adaptive_thinking_medium(router: Router) -> None:
    """Adaptive thinking with medium effort."""
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="What is 27 * 43? Think step by step.")),
    ]

    response = await router.create(
        model=OPUS_4_6_MODEL,
        input=messages,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )

    assert response is not None
    assert response.output


async def test_opus_4_6_adaptive_thinking_multi_turn(router: Router) -> None:
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
        model=OPUS_4_6_MODEL,
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
        model=OPUS_4_6_MODEL,
        input=messages,
        reasoning={"effort": "xhigh", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    assert response2 is not None
    assert response2.output


async def test_opus_4_6_no_reasoning(router: Router) -> None:
    """Opus 4.6 works without reasoning (thinking disabled)."""
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="Say hello in one word.")),
    ]

    response = await router.create(
        model=OPUS_4_6_MODEL,
        input=messages,
    )

    assert response is not None
    assert response.output


# endregion
