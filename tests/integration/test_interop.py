"""Integration tests to verify interoperability between providers."""

import os

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam, WebSearchToolParam
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.tool_param import ImageGeneration
import pytest

from interop_router.router import Router
from interop_router.types import ChatMessage, SupportedModel

INTEROP_PARAMS = [
    pytest.param("gpt-5.2", "gemini-3-flash-preview", id="openai-to-gemini"),
    pytest.param("gemini-3-flash-preview", "gpt-5.2", id="gemini-to-openai"),
    pytest.param("gpt-5.2", "claude-haiku-4-5-20251001", id="openai-to-anthropic"),
    pytest.param("claude-haiku-4-5-20251001", "gpt-5.2", id="anthropic-to-openai"),
    pytest.param("gemini-3-flash-preview", "claude-haiku-4-5-20251001", id="gemini-to-anthropic"),
    pytest.param("claude-haiku-4-5-20251001", "gemini-3-flash-preview", id="anthropic-to-gemini"),
]

FUNCTION_TOOLS: list[FunctionToolParam] = [
    FunctionToolParam(
        type="function",
        name="get_weather",
        description="Get the current weather for a given location.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and country"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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
                "ticker": {"type": "string", "description": "The stock ticker symbol"},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
        strict=True,
    ),
]


def filter_orphaned_tool_calls(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Remove function calls without matching outputs and vice versa."""
    call_ids = {msg.message.get("call_id") for msg in messages if msg.message.get("type") == "function_call"}
    output_ids = {msg.message.get("call_id") for msg in messages if msg.message.get("type") == "function_call_output"}
    matched_ids = call_ids & output_ids

    return [
        msg
        for msg in messages
        if msg.message.get("type") not in ("function_call", "function_call_output")
        or msg.message.get("call_id") in matched_ids
    ]


@pytest.fixture
def router() -> Router:
    router = Router()
    router.register("openai", AsyncOpenAI())
    router.register("gemini", genai.Client(api_key=os.getenv("GEMINI_API_KEY")))
    router.register("anthropic", AsyncAnthropic())
    return router


@pytest.mark.parametrize(("first_model", "second_model"), INTEROP_PARAMS)
async def test_basic(router: Router, first_model: SupportedModel, second_model: SupportedModel):
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="Hello! What is 2+2?")),
    ]

    response1 = await router.create(
        input=messages,
        model=first_model,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    assert response1 is not None

    messages.extend(response1.output)
    messages.append(ChatMessage(message=EasyInputMessageParam(role="user", content="And what is 3+3?")))

    response2 = await router.create(input=messages, model=second_model)
    assert response2 is not None


@pytest.mark.parametrize(("first_model", "second_model"), INTEROP_PARAMS)
async def test_web_search(router: Router, first_model: SupportedModel, second_model: SupportedModel):
    system_prompt = """You are a thoughtful assistant. \
You may be provided web search results that were generated with another web search tool. \
You must always reason about the correct tool definition to use, not necessarily one that was used before"""
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="system",
                content=system_prompt,
            )
        ),
        ChatMessage(message=EasyInputMessageParam(role="user", content="Can you look up one latest article about AI?")),
    ]

    response1 = await router.create(
        input=messages,
        model=first_model,
        include=["web_search_call.results", "web_search_call.action.sources"],
        tools=[WebSearchToolParam(type="web_search")],
    )
    assert response1 is not None

    messages.extend(response1.output)
    messages.append(
        ChatMessage(
            message=EasyInputMessageParam(role="user", content="Can you summarize that article in one sentence?")
        )
    )
    response2 = await router.create(
        input=messages,
        model=second_model,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["web_search_call.results", "web_search_call.action.sources", "reasoning.encrypted_content"],
        tools=[WebSearchToolParam(type="web_search")],
    )
    assert response2 is not None


@pytest.mark.parametrize(("first_model", "second_model"), INTEROP_PARAMS)
async def test_web_search_weather(router: Router, first_model: SupportedModel, second_model: SupportedModel):
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="What is the weather in New York, New York?")),
    ]

    response1 = await router.create(
        input=messages,
        model=first_model,
        include=["web_search_call.results", "web_search_call.action.sources"],
        tools=[WebSearchToolParam(type="web_search")],
    )
    assert response1 is not None

    messages.extend(response1.output)
    messages.append(ChatMessage(message=EasyInputMessageParam(role="user", content="Thank you!")))
    response2 = await router.create(input=messages, model=second_model)
    assert response2 is not None


@pytest.mark.parametrize(("first_model", "second_model"), INTEROP_PARAMS)
async def test_web_search_roundtrip(router: Router, first_model: SupportedModel, second_model: SupportedModel):
    system_prompt = """You are a thoughtful assistant. \
You may be provided web search results that were generated with another web search tool. \
You must always reason about the correct tool definition to use, not necessarily one that was used before"""
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="system", content=system_prompt)),
        ChatMessage(message=EasyInputMessageParam(role="user", content="Can you look up one latest article about AI?")),
    ]

    response1 = await router.create(
        input=messages,
        model=first_model,
        include=["web_search_call.results", "web_search_call.action.sources"],
        tools=[WebSearchToolParam(type="web_search")],
    )
    assert response1 is not None

    messages.extend(response1.output)
    messages.append(
        ChatMessage(
            message=EasyInputMessageParam(role="user", content="Can you summarize that article in one sentence?")
        )
    )
    response2 = await router.create(
        input=messages,
        model=second_model,
        include=["web_search_call.results", "web_search_call.action.sources"],
        tools=[WebSearchToolParam(type="web_search")],
    )
    assert response2 is not None

    messages.extend(response2.output)
    messages = filter_orphaned_tool_calls(messages)
    messages.append(ChatMessage(message=EasyInputMessageParam(role="user", content="Thank you!")))
    response3 = await router.create(input=messages, model=first_model)
    assert response3 is not None


@pytest.mark.parametrize(("first_model", "second_model"), INTEROP_PARAMS)
async def test_function_call_handoff(router: Router, first_model: SupportedModel, second_model: SupportedModel):
    """This tests the scenario where:
    1. Provider A makes a function call
    2. User provides function output
    3. Provider B interprets the result (not Provider A)

    This is distinct from test_function_calling_roundtrip where the same provider
    that made the call also interprets the result.
    """
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="What's the weather in Tokyo? Use celsius.")),
    ]

    response1 = await router.create(input=messages, model=first_model, tools=FUNCTION_TOOLS)
    assert response1 is not None

    messages.extend(response1.output)
    for msg in response1.output:
        if msg.message.get("type") == "function_call":
            messages.append(
                ChatMessage(
                    message=FunctionCallOutput(
                        call_id=msg.message.get("call_id", ""),
                        output='{"temperature": 22, "unit": "celsius", "conditions": "sunny"}',
                        type="function_call_output",
                    )
                )
            )

    # Second provider interprets the function output (not the same provider that made the call)
    response2 = await router.create(input=messages, model=second_model)
    assert response2 is not None


@pytest.mark.parametrize(("first_model", "second_model"), INTEROP_PARAMS)
async def test_function_calling_roundtrip(router: Router, first_model: SupportedModel, second_model: SupportedModel):
    messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="What's the weather in Tokyo? Use celsius.")),
    ]

    response1 = await router.create(input=messages, model=first_model, tools=FUNCTION_TOOLS)
    assert response1 is not None

    messages.extend(response1.output)
    for msg in response1.output:
        if msg.message.get("type") == "function_call":
            messages.append(
                ChatMessage(
                    message=FunctionCallOutput(
                        call_id=msg.message.get("call_id", ""),
                        output='{"temperature": 22, "unit": "celsius", "conditions": "sunny"}',
                        type="function_call_output",
                    )
                )
            )

    response2 = await router.create(input=messages, model=first_model)
    assert response2 is not None
    messages.extend(response2.output)

    messages.append(
        ChatMessage(message=EasyInputMessageParam(role="user", content="What should I wear for that weather?"))
    )
    response3 = await router.create(input=messages, model=second_model)
    assert response3 is not None


@pytest.mark.parametrize(("first_model", "second_model"), INTEROP_PARAMS)
async def test_function_calling_parallel_roundtrip(
    router: Router, first_model: SupportedModel, second_model: SupportedModel
):
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="I need the weather in NYC (fahrenheit) and Apple's stock price. Call both tools.",
            )
        ),
    ]

    response1 = await router.create(
        input=messages,
        model=first_model,
        tools=FUNCTION_TOOLS,
        tool_choice="required",
        instructions="You are a helpful assistant who calls tools.",
    )
    assert response1 is not None

    messages.extend(response1.output)
    for msg in response1.output:
        if msg.message.get("type") == "function_call":
            name = msg.message.get("name", "")
            call_id = msg.message.get("call_id", "")
            if name == "get_weather":
                messages.append(
                    ChatMessage(
                        message=FunctionCallOutput(
                            call_id=call_id,
                            output='{"temperature": 45, "unit": "fahrenheit", "conditions": "cloudy"}',
                            type="function_call_output",
                        )
                    )
                )
            elif name == "get_stock_price":
                messages.append(
                    ChatMessage(
                        message=FunctionCallOutput(
                            call_id=call_id,
                            output='{"ticker": "AAPL", "price": 178.50, "currency": "USD"}',
                            type="function_call_output",
                        )
                    )
                )

    response2 = await router.create(input=messages, model=first_model)
    assert response2 is not None
    messages.extend(response2.output)

    messages.append(
        ChatMessage(
            message=EasyInputMessageParam(role="user", content="Should I buy AAPL stock today given the weather?")
        )
    )
    response3 = await router.create(input=messages, model=second_model)
    assert response3 is not None


IMAGE_GEN_INTEROP_PARAMS = [
    pytest.param(
        ("gemini-3-flash-preview", "gemini-3-pro-image-preview"),
        ("gpt-5.2", "gpt-image-1.5"),
        id="gemini-image-to-openai",
    ),
    pytest.param(
        ("gpt-5.2", "gpt-image-1.5"),
        ("gemini-3-flash-preview", "gemini-3-pro-image-preview"),
        id="openai-to-gemini-image",
    ),
]


@pytest.mark.parametrize(("first_model_pair", "second_model_pair"), IMAGE_GEN_INTEROP_PARAMS)
async def test_image_gen_with_web_search_and_refinement(
    router: Router,
    first_model_pair: tuple[SupportedModel, str],
    second_model_pair: tuple[SupportedModel, str],
):
    first_model, first_image_tool_model = first_model_pair
    second_model, second_image_tool_model = second_model_pair
    first_image_tool = ImageGeneration(type="image_generation", model=first_image_tool_model)
    second_image_tool = ImageGeneration(type="image_generation", model=second_image_tool_model)
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content=(
                    "Visualize the current weather forecast for the next 3 days in San Francisco as a clean, modern weather chart."
                ),
            ),
            provider_kwargs={"gemini": {"image_config": {"aspect_ratio": "16:9"}}},
        ),
    ]
    response1 = await router.create(
        input=messages,
        model=first_model,
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[first_image_tool, WebSearchToolParam(type="web_search")],
        include=["web_search_call.results", "web_search_call.action.sources", "reasoning.encrypted_content"],
    )
    assert response1 is not None

    messages.extend(response1.output)
    messages.append(
        ChatMessage(message=EasyInputMessageParam(role="user", content="Thanks! What day looks best for a picnic?"))
    )
    response2 = await router.create(
        input=messages,
        model=second_model,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    assert response2 is not None

    messages.extend(response2.output)
    messages.append(
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Can you update the weather chart to highlight the best picnic day with a star icon?",
            ),
            provider_kwargs={"gemini": {"image_config": {"aspect_ratio": "16:9"}}},
        )
    )
    response3 = await router.create(
        input=messages,
        model=second_model,
        tools=[second_image_tool],
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    assert response3 is not None
