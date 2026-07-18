import os
from typing import cast

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam, ResponseTextConfigParam
from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam
from openai.types.responses.tool_param import ImageGeneration
from openai.types.shared_params.reasoning import Reasoning
import pytest

from interop_router.router import Router
from interop_router.types import ChatMessage, SupportedModelAnthropic, SupportedModelOpenAI


@pytest.fixture
def router() -> Router:
    router = Router()
    router.register("openai", AsyncOpenAI())
    router.register("gemini", genai.Client(api_key=os.getenv("GEMINI_API_KEY")))
    router.register("anthropic", AsyncAnthropic())
    return router


# region: Anthropic models

ANTHROPIC_MODELS: list[SupportedModelAnthropic] = ["claude-opus-4-8", "claude-sonnet-5"]


@pytest.mark.parametrize("model", ANTHROPIC_MODELS)
async def test_adaptive_thinking_summary(router: Router, model: SupportedModelAnthropic) -> None:
    """Requesting a reasoning summary returns non-empty summary text."""
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
    reasoning_items = [
        cast(ResponseReasoningItemParam, output.message)
        for output in response.output
        if output.message.get("type") == "reasoning"
    ]
    assert reasoning_items
    assert any(summary["text"].strip() for reasoning_item in reasoning_items for summary in reasoning_item["summary"])


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


async def test_mid_conversation_system_messages(router: Router) -> None:
    """Opus 4.8 accepts system-level instructions throughout a conversation."""
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="system", content="You are a helpful geography assistant.")),
        ChatMessage(message=EasyInputMessageParam(role="user", content="We will discuss European geography.")),
        ChatMessage(message=EasyInputMessageParam(role="system", content="Keep all later answers concise.")),
        ChatMessage(message=EasyInputMessageParam(role="assistant", content="Understood.")),
        ChatMessage(message=EasyInputMessageParam(role="user", content="Treat country names as case-insensitive.")),
        ChatMessage(message=EasyInputMessageParam(role="developer", content="Use conventional English place names.")),
        ChatMessage(message=EasyInputMessageParam(role="assistant", content="Understood.")),
        ChatMessage(message=EasyInputMessageParam(role="user", content="What is the capital of France?")),
        ChatMessage(message=EasyInputMessageParam(role="system", content="Answer in one short sentence.")),
    ]

    response = await router.create(
        model="claude-opus-4-8",
        input=messages,
    )

    assert response.output


async def test_prompt_caching(router: Router) -> None:
    """Automatic prompt caching via provider_kwargs produces cache read tokens on the second turn."""
    cache_control = {"cache_control": {"type": "ephemeral"}}
    model: SupportedModelAnthropic = "claude-sonnet-5"

    # Use a long system prompt to exceed the minimum cacheable length.
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
        model="gemini-3.1-flash-lite-preview",
        tools=[image_tool],
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    assert response is not None
    assert response.output


# endregion

# region: OpenAI models

OPENAI_MODELS: list[SupportedModelOpenAI] = ["gpt-5.6-terra"]


@pytest.mark.parametrize("model", OPENAI_MODELS)
async def test_verbosity(router: Router, model: SupportedModelOpenAI) -> None:
    """Verbosity parameter controls response length."""
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="Explain what a binary tree is.")),
    ]

    response = await router.create(
        model=model,
        input=messages,
        text=ResponseTextConfigParam(verbosity="low"),
    )

    assert response is not None
    assert response.output


@pytest.mark.parametrize("model", OPENAI_MODELS)
async def test_sampling_params_with_no_reasoning(router: Router, model: SupportedModelOpenAI) -> None:
    """temperature, top_p, and top_logprobs are only valid with effort 'none' for GPT-5 family models."""
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="Write a creative one-sentence story.")),
    ]

    response = await router.create(
        model=model,
        input=messages,
        temperature=0.9,
        top_p=0.9,
        top_logprobs=5,
        reasoning={"effort": "none"},
        include=["message.output_text.logprobs"],
    )

    assert response is not None
    assert response.output


@pytest.mark.parametrize("model", OPENAI_MODELS)
async def test_preserve_reasoning(router: Router, model: SupportedModelOpenAI) -> None:
    """Test the preserve reasoning feature where the model can persist reasoning across multiple turns with reasoning={"context": "all_turns"} set."""
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content=(
                    "A store discounts a $240 item by 15%, then applies 8% sales tax. "
                    "Calculate the final price and briefly explain your reasoning."
                ),
            )
        ),
    ]
    reasoning: Reasoning = {"context": "all_turns", "effort": "high"}

    response1 = await router.create(
        model=model,
        input=messages,
        reasoning=reasoning,
        include=["reasoning.encrypted_content"],
    )

    assert response1.output

    messages.extend(response1.output)
    messages.append(
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Now calculate the final price if the discount were 25% instead, using the same method.",
            )
        )
    )

    response2 = await router.create(
        model=model,
        input=messages,
        reasoning=reasoning,
        include=["reasoning.encrypted_content"],
    )

    assert response2.output


@pytest.mark.parametrize("model", OPENAI_MODELS)
async def test_pro_reasoning(router: Router, model: SupportedModelOpenAI) -> None:
    """Pro reasoning mode produces a response."""
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content=("Write a haiku about bears that rhymes."),
            )
        ),
    ]

    response = await router.create(
        model=model,
        input=messages,
        reasoning={"mode": "pro", "effort": "medium"},
    )

    assert response.output


# endregion
