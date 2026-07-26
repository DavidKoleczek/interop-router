"""Streaming counterparts to tests in test_provider.py.

Each test mirrors its non-streaming sibling but invokes the router with
stream=True and asserts only that at least one event was yielded.
"""

import base64
import os
from pathlib import Path
from typing import Any, cast

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputTextParam,
    WebSearchToolParam,
)
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response_input_item_param import FunctionCallOutput, Message
import pytest

from interop_router.router import Router
from interop_router.types import ChatMessage, ProviderName, SupportedModel

READ_IMAGE_TOOL = FunctionToolParam(
    type="function",
    name="read_image",
    description="Read an image file and return its contents.",
    parameters=cast(
        dict[str, object],
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the image file."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    strict=True,
)

FUNCTION_TOOLS: list[FunctionToolParam] = [
    FunctionToolParam(
        type="function",
        name="get_weather",
        description="Get the current weather for a given location.",
        parameters=cast(
            dict[str, object],
            {
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
        ),
        strict=True,
    ),
    FunctionToolParam(
        type="function",
        name="get_stock_price",
        description="Get the current stock price for a given ticker symbol.",
        parameters=cast(
            dict[str, object],
            {
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
        ),
        strict=True,
    ),
]

PROVIDER_MODEL_PARAMS = [
    pytest.param("openai", "gpt-5.6-terra"),
    pytest.param("anthropic", "claude-sonnet-5"),
    pytest.param("gemini", "gemini-3.6-flash"),
]


@pytest.fixture
def router() -> Router:
    return Router()


def get_client(provider: ProviderName) -> AsyncOpenAI | genai.Client | AsyncAnthropic:
    if provider == "openai":
        return AsyncOpenAI()
    if provider == "gemini":
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    if provider == "anthropic":
        return AsyncAnthropic()
    raise ValueError(f"Unknown provider: {provider}")


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_basic(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    message = ChatMessage(message=EasyInputMessageParam(role="user", content="Hello!"))
    stream = await router.create(input=[message], model=model, stream=True)

    events = []
    async for event in stream:
        events.append(event)
    assert len(events) > 0


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_basic_chat_history(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(role="system", content="You are a helpful assistant who replies in one word.")
        ),
        ChatMessage(message=EasyInputMessageParam(role="user", content="Hello!")),
        ChatMessage(message=EasyInputMessageParam(role="assistant", content="Hiya!")),
        ChatMessage(message=EasyInputMessageParam(role="user", content="What can you help me with?")),
    ]

    stream = await router.create(
        model=model,
        input=messages,
        max_output_tokens=64_000,
        stream=True,
    )
    events = [event async for event in stream]
    assert len(events) > 0


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_reasoning(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(message=EasyInputMessageParam(role="system", content="You are a thoughtful assistant.")),
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Can you please think deeply about the meaning of life? Come up with a nuanced response in one sentence.",
            )
        ),
    ]

    stream = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        max_output_tokens=64_000,
        truncation="auto",
        stream=True,
    )
    events = [event async for event in stream]
    assert len(events) > 0


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_image_understanding(router: Router, provider: ProviderName, model: SupportedModel):
    image_path = Path(__file__).parents[1] / "integration" / "data" / "landscape.png"
    image_bytes = image_path.read_bytes()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{base64_image}"

    client = get_client(provider)
    router.register(provider, client)

    message = ChatMessage(
        message=Message(
            role="user",
            content=cast(
                list[Any],
                [
                    ResponseInputTextParam(type="input_text", text="What is in this image?"),
                    ResponseInputImageParam(type="input_image", image_url=data_url, detail="auto"),
                ],
            ),
        )
    )
    stream = await router.create(input=[message], model=model, stream=True)
    events = [event async for event in stream]
    assert len(events) > 0


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_image_in_tool_result(router: Router, provider: ProviderName, model: SupportedModel):
    """Verify that streaming works when a tool result contains an image.

    The setup call is intentionally non-streaming to obtain the function_call the
    model wants to make; the streamed call is the one that consumes the image in
    the function_call_output, which is the scenario under test.
    """
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Use the read_image tool to read the image at '/tmp/landscape.png', then describe what you see.",
            )
        ),
    ]

    response = await router.create(
        model=model,
        input=messages,
        tools=[READ_IMAGE_TOOL],
        tool_choice="auto",
    )
    assert response is not None

    messages.extend(response.output)
    for chat_message in response.output:
        msg = chat_message.message
        if msg.get("type") == "function_call" and msg.get("name") == "read_image":
            call_id = cast(str, msg.get("call_id", ""))

            image_path = Path(__file__).parents[1] / "integration" / "data" / "landscape.png"
            image_bytes = image_path.read_bytes()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{base64_image}"

            messages.append(
                ChatMessage(
                    message=FunctionCallOutput(
                        call_id=call_id,
                        output=cast(
                            list[Any],
                            [ResponseInputImageContentParam(type="input_image", image_url=data_url, detail="auto")],
                        ),
                        type="function_call_output",
                    )
                )
            )

            stream = await router.create(
                model=model,
                input=messages,
                tools=[READ_IMAGE_TOOL],
                tool_choice="auto",
                stream=True,
            )
            events = [event async for event in stream]
            assert len(events) > 0
            break


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_function_calling_basic(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="What's the weather like in Tokyo, Japan right now? Please use celsius.",
            )
        ),
    ]

    stream = await router.create(
        model=model,
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        stream=True,
    )
    events = [event async for event in stream]
    assert len(events) > 0


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_function_calling_parallel(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="I need two things at the same time: 1) the weather in New York in fahrenheit, and 2) Apple's current stock price. Please call both tools at the same time.",
            )
        ),
    ]

    stream = await router.create(
        model=model,
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="required",
        instructions="You are a helpful tool calling assistant who calls tools in parallel.",
        stream=True,
    )
    events = [event async for event in stream]
    assert len(events) > 0


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_function_calling_parallel_reasoning(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="I need two things at the same time: 1) the weather in New York in fahrenheit, and 2) Apple's current stock price. Please call both tools at the same time.",
            )
        ),
    ]

    stream = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        tools=FUNCTION_TOOLS,
        instructions="You are a helpful tool calling assistant who calls tools in parallel.",
        stream=True,
    )
    events = [event async for event in stream]
    assert len(events) > 0


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_web_search(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Can you find a list of 5 recent articles on AI advancements from 2025? Just list the links without a description.",
            )
        )
    ]
    stream = await router.create(
        input=messages,
        model=model,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["web_search_call.results", "web_search_call.action.sources", "reasoning.encrypted_content"],
        tools=[WebSearchToolParam(type="web_search")],
        stream=True,
    )
    events = [event async for event in stream]
    assert len(events) > 0


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_web_fetch(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Go to https://example.com and tell me what the page title is.",
            )
        )
    ]
    stream = await router.create(
        input=messages,
        model=model,
        tools=[WebSearchToolParam(type="web_search")],
        stream=True,
    )
    events = [event async for event in stream]
    assert len(events) > 0
