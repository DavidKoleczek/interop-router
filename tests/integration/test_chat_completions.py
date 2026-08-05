"""Integration tests for the chat_completions provider.

Local OpenAI-compatible servers are gated on CHAT_COMPLETIONS_BASE_URL and CHAT_COMPLETIONS_MODEL.
OpenAI Chat Completions cloud tests use OPENAI_API_KEY via AsyncOpenAI().
"""

import base64
import os
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam, ResponseInputImageParam, ResponseInputTextParam
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response_input_item_param import FunctionCallOutput, Message
import pytest

from interop_router.router import Router
from interop_router.types import ChatMessage, ContextLimitExceededError, RouterResponse

OPENAI_CHAT_COMPLETIONS_MODEL = "chat_completions/gpt-5.6-luna"

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


@pytest.fixture
def router() -> Router:
    return Router()


def _require_local_env() -> tuple[str, str]:
    base_url = os.getenv("CHAT_COMPLETIONS_BASE_URL")
    model = os.getenv("CHAT_COMPLETIONS_MODEL")
    if not base_url or not model:
        pytest.skip("Set CHAT_COMPLETIONS_BASE_URL and CHAT_COMPLETIONS_MODEL to run local chat_completions tests")
    return base_url, model


def _local_client(base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url)


def _openai_cc_client() -> AsyncOpenAI:
    return AsyncOpenAI()


# region: Local Chat Completions


async def test_local_basic(router: Router) -> None:
    base_url, model = _require_local_env()
    router.register("chat_completions", _local_client(base_url))

    message = ChatMessage(message=EasyInputMessageParam(role="user", content="Hello!"))
    response = await router.create(input=[message], model=f"chat_completions/{model}")
    assert response is not None


async def test_local_basic_chat_history(router: Router) -> None:
    base_url, model = _require_local_env()
    router.register("chat_completions", _local_client(base_url))

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(role="system", content="You are a helpful assistant who replies in one word.")
        ),
        ChatMessage(message=EasyInputMessageParam(role="user", content="Hello!")),
        ChatMessage(message=EasyInputMessageParam(role="assistant", content="Hiya!")),
        ChatMessage(message=EasyInputMessageParam(role="user", content="What can you help me with?")),
    ]

    response = await router.create(
        model=f"chat_completions/{model}",
        input=messages,
        max_output_tokens=1_000,
    )
    assert response is not None


async def test_local_function_calling_basic(router: Router) -> None:
    base_url, model = _require_local_env()
    router.register("chat_completions", _local_client(base_url))

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="What's the weather like in Tokyo, Japan right now? Please use celsius.",
            )
        ),
    ]

    response = await router.create(
        model=f"chat_completions/{model}",
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        reasoning={"effort": "medium"},
    )
    assert response is not None

    messages.extend(response.output)
    function_call_outputs: list[ChatMessage] = []
    for chat_message in response.output:
        msg = chat_message.message
        if msg.get("type") == "function_call" and msg.get("name") == "get_weather":
            call_id = cast(str, msg.get("call_id", ""))
            function_call_outputs.append(
                ChatMessage(
                    message=FunctionCallOutput(
                        call_id=call_id,
                        output='{"temperature": 22, "unit": "celsius", "conditions": "sunny"}',
                        type="function_call_output",
                    )
                )
            )

    messages.extend(function_call_outputs)
    response2 = await router.create(
        model=f"chat_completions/{model}",
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        reasoning={"effort": "medium"},
    )
    assert response2 is not None


async def test_local_function_calling_parallel(router: Router) -> None:
    base_url, model = _require_local_env()
    router.register("chat_completions", _local_client(base_url))

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content=(
                    "I need two things at the same time: 1) the weather in New York in fahrenheit, "
                    "and 2) Apple's current stock price. Please call both tools at the same time."
                ),
            )
        ),
    ]

    response = await router.create(
        model=f"chat_completions/{model}",
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="required",
        instructions="You are a helpful tool calling assistant who calls tools in parallel.",
        reasoning={"effort": "medium"},
    )
    assert response is not None

    messages.extend(response.output)
    function_call_outputs: list[ChatMessage] = []
    for chat_message in response.output:
        msg = chat_message.message
        if msg.get("type") == "function_call":
            name = msg.get("name", "")
            call_id = cast(str, msg.get("call_id", ""))
            if name == "get_weather":
                function_call_outputs.append(
                    ChatMessage(
                        message=FunctionCallOutput(
                            call_id=call_id,
                            output='{"temperature": 45, "unit": "fahrenheit", "conditions": "cloudy"}',
                            type="function_call_output",
                        )
                    )
                )
            elif name == "get_stock_price":
                function_call_outputs.append(
                    ChatMessage(
                        message=FunctionCallOutput(
                            call_id=call_id,
                            output='{"ticker": "AAPL", "price": 178.50, "currency": "USD"}',
                            type="function_call_output",
                        )
                    )
                )

    messages.extend(function_call_outputs)
    response2 = await router.create(
        model=f"chat_completions/{model}",
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        reasoning={"effort": "medium"},
    )
    assert response2 is not None


async def test_local_image_understanding(router: Router) -> None:
    image_path = Path(__file__).parents[0] / "data" / "landscape.png"
    image_bytes = image_path.read_bytes()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{base64_image}"

    base_url, model = _require_local_env()
    router.register("chat_completions", _local_client(base_url))

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
    response = await router.create(
        input=[message],
        model=f"chat_completions/{model}",
        reasoning={"effort": "medium"},
    )
    assert response is not None


async def test_local_context_limit_exceeded(router: Router) -> None:
    """Test that context limit errors are raised when input exceeds limits."""

    def generate_large_content(target_tokens: int) -> str:
        """Generate content large enough to exceed typical context limits."""
        base_text = "This is a test message with content to fill the context window. "
        chars_needed = target_tokens * 4
        repetitions = chars_needed // len(base_text) + 1
        return base_text * repetitions

    base_url, model = _require_local_env()
    router.register("chat_completions", _local_client(base_url))

    large_content = generate_large_content(target_tokens=2_000_000)
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(role="user", content=large_content),
        )
    ]

    with pytest.raises(ContextLimitExceededError):
        await router.create(input=messages, model=f"chat_completions/{model}")


# endregion

# region: Local Chat Completions Streaming


async def test_local_stream(router: Router) -> None:
    base_url, model = _require_local_env()
    router.register("chat_completions", _local_client(base_url))

    message = ChatMessage(message=EasyInputMessageParam(role="user", content="Hello!"))
    stream = await router.create(input=[message], model=f"chat_completions/{model}", stream=True)

    events = []
    async for event in stream:
        events.append(event)
    assert len(events) > 0


async def test_local_stream_function_calling(router: Router) -> None:
    base_url, model = _require_local_env()
    router.register("chat_completions", _local_client(base_url))

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="What's the weather like in Tokyo, Japan right now? Please use celsius.",
            )
        ),
    ]

    stream = await router.create(
        model=f"chat_completions/{model}",
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        reasoning={"effort": "medium"},
        stream=True,
    )
    events = [event async for event in stream]
    assert events
    response = events[-1]
    assert isinstance(response, RouterResponse)

    messages.extend(response.output)
    function_call_outputs: list[ChatMessage] = []
    for chat_message in response.output:
        msg = chat_message.message
        if msg.get("type") == "function_call" and msg.get("name") == "get_weather":
            call_id = cast(str, msg.get("call_id", ""))
            function_call_outputs.append(
                ChatMessage(
                    message=FunctionCallOutput(
                        call_id=call_id,
                        output='{"temperature": 22, "unit": "celsius", "conditions": "sunny"}',
                        type="function_call_output",
                    )
                )
            )

    messages.extend(function_call_outputs)
    followup_stream = await router.create(
        model=f"chat_completions/{model}",
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
        reasoning={"effort": "medium"},
        stream=True,
    )
    followup_events = [event async for event in followup_stream]
    assert followup_events
    assert isinstance(followup_events[-1], RouterResponse)


# endregion

# region: OpenAI Chat Completions


async def test_openai_basic(router: Router) -> None:
    router.register("chat_completions", _openai_cc_client())

    message = ChatMessage(message=EasyInputMessageParam(role="user", content="Hello!"))
    response = await router.create(input=[message], model=OPENAI_CHAT_COMPLETIONS_MODEL)
    assert response is not None


async def test_openai_reasoning_and_tool_call(router: Router) -> None:
    """Tool calling on OpenAI Chat Completions.

    gpt-5.6 rejects non null reasoning_effort with tools.
    Reasoning text is never returned on Chat Completions.
    """
    router.register("chat_completions", _openai_cc_client())

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="system",
                content="You are a helpful assistant. Think carefully especially when calling tools.",
            )
        ),
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="What's the weather like in Tokyo, Japan right now? Please use celsius.",
            )
        ),
    ]

    response = await router.create(
        model=OPENAI_CHAT_COMPLETIONS_MODEL,
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="required",
        reasoning={"effort": "none"},
    )
    assert response is not None
    assert any(
        msg.message.get("type") == "function_call" and msg.message.get("name") == "get_weather"
        for msg in response.output
    )


async def test_openai_stream(router: Router) -> None:
    router.register("chat_completions", _openai_cc_client())

    message = ChatMessage(message=EasyInputMessageParam(role="user", content="Hello!"))
    stream = await router.create(input=[message], model=OPENAI_CHAT_COMPLETIONS_MODEL, stream=True)

    events = []
    async for event in stream:
        events.append(event)
    assert len(events) > 0


# endregion
