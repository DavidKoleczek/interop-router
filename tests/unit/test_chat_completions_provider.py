"""Unit tests for ChatCompletionsProvider message conversion.

Fixtures are taken from real OpenAI Responses dumps
(see tmp/dump_openai_responses_conversation.py): parallel function tools,
image + web_search with reasoning, and assistant preamble before tool calls.
"""

import base64
from pathlib import Path
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCallParam,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputTextParam,
    ResponseOutputMessageParam,
    ResponseOutputTextParam,
    ResponseReasoningItemParam,
    ToolParam,
    WebSearchToolParam,
)
from openai.types.responses.file_search_tool_param import FileSearchToolParam
from openai.types.responses.response_input_item_param import FunctionCallOutput, Message, ResponseInputItemParam
from openai.types.responses.response_reasoning_item_param import Summary

from interop_router.chat_completions_provider import ChatCompletionsProvider
from interop_router.types import ChatMessage

# region: Final followup input

INPUT_MESSAGES_1: list[dict[str, Any]] = [
    {
        "role": "developer",
        "content": "You are a helpful assistant who calls tools when needed.",
    },
    {
        "role": "user",
        "content": "I need the weather in NYC (fahrenheit) and Apple's stock price.",
    },
    {
        "arguments": '{"location":"New York City","unit":"fahrenheit"}',
        "call_id": "call_VcLuEjxesXgjVzRtWyrYCEDd",
        "name": "get_weather",
        "type": "function_call",
    },
    {
        "arguments": '{"ticker":"AAPL"}',
        "call_id": "call_z9a3KEMi4e7i5qJscp8jHyG4",
        "name": "get_stock_price",
        "type": "function_call",
    },
    {
        "type": "function_call_output",
        "call_id": "call_VcLuEjxesXgjVzRtWyrYCEDd",
        "output": '{"temperature": 45, "unit": "fahrenheit", "conditions": "cloudy"}',
    },
    {
        "type": "function_call_output",
        "call_id": "call_z9a3KEMi4e7i5qJscp8jHyG4",
        "output": '{"ticker": "AAPL", "price": 178.50, "currency": "USD"}',
    },
    {
        "content": [
            {
                "annotations": [],
                "text": "- **NYC weather:** 45°F, cloudy  \n- **Apple (AAPL):** $178.50 per share",
                "type": "output_text",
                "logprobs": [],
            }
        ],
        "role": "assistant",
        "phase": "final_answer",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "Summarize that in one sentence.",
            }
        ],
    },
]


def test_convert_final_followup_input() -> None:
    """Full roundtrip input: tools, string tool outputs, assistant output_text, user list content."""
    weather_call_id = "call_VcLuEjxesXgjVzRtWyrYCEDd"
    stock_call_id = "call_z9a3KEMi4e7i5qJscp8jHyG4"

    input_messages = [ChatMessage(message=cast(ResponseInputItemParam, item)) for item in INPUT_MESSAGES_1]

    expected: list[ChatCompletionMessageParam] = [
        {
            "role": "developer",
            "content": "You are a helpful assistant who calls tools when needed.",
        },
        {
            "role": "user",
            "content": "I need the weather in NYC (fahrenheit) and Apple's stock price.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": weather_call_id,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"New York City","unit":"fahrenheit"}',
                    },
                },
                {
                    "id": stock_call_id,
                    "type": "function",
                    "function": {
                        "name": "get_stock_price",
                        "arguments": '{"ticker":"AAPL"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": weather_call_id,
            "content": '{"temperature": 45, "unit": "fahrenheit", "conditions": "cloudy"}',
        },
        {
            "role": "tool",
            "tool_call_id": stock_call_id,
            "content": '{"ticker": "AAPL", "price": 178.50, "currency": "USD"}',
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "- **NYC weather:** 45°F, cloudy  \n- **Apple (AAPL):** $178.50 per share",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Summarize that in one sentence."}],
        },
    ]

    assert ChatCompletionsProvider._convert_input_messages(input_messages) == expected


# endregion

# region: Image web search reasoning history

WEB_SEARCH_CALL_ID_1 = "ws_0d3e04ec423e8711016a6f5a6d730881979fe509fc2280e9e4"
WEB_SEARCH_CALL_ID_2 = "ws_0d3e04ec423e8711016a6f5a762c5081979dadf4324317282b"
WEB_SEARCH_ARGS_1 = '{"type": "search", "query": "sunset photography water tips exposure focus tripod"}'
WEB_SEARCH_ARGS_2 = """{"type": "search", "query": "photography polarizing filter water reflections landscape tips"}"""
WEB_SEARCH_OUTPUT_1 = """\
The complete result of this tool call was removed. Please call a web search tool if you need the data \
again. Importantly, the web_search tool definition may have changed, including the name of the tool \
itself. Be sure to use the latest definition.

Sources found:
- https://a.example/tips
- https://b.example/water
- https://c.example/omitted"""
WEB_SEARCH_OUTPUT_2 = """\
The complete result of this tool call was removed. Please call a web search tool if you need the data \
again. Importantly, the web_search tool definition may have changed, including the name of the tool \
itself. Be sure to use the latest definition.

Sources found:
- https://a.example/sunset
- https://b.example/filters
- https://c.example/omitted"""
ASSISTANT_SCENE_TEXT = "River sunset tips ([a.com](https://a.com), [b.com](https://b.com))."
ASSISTANT_SUMMARY_TEXT = "Use a tripod; compose around reflections."

INPUT_MESSAGES_2: list[ResponseInputItemParam] = [
    EasyInputMessageParam(
        role="developer",
        content="You are a helpful assistant. Use web search when you need current information.",
    ),
    EasyInputMessageParam(
        role="user",
        content=[
            ResponseInputTextParam(
                type="input_text",
                text="""\
Look at this photo. Briefly describe the scene, then use web search to find \
practical tips for photographing sunsets over water like this.""",
            ),
            ResponseInputImageParam(
                type="input_image",
                detail="auto",
                image_url="data:image/png;base64,SHORT_TEST_IMAGE",
            ),
        ],
    ),
    ResponseReasoningItemParam(
        id="rs_1",
        type="reasoning",
        encrypted_content="encrypted_reasoning_placeholder",
        summary=[
            Summary(
                type="summary_text",
                text="""**Searching for photography tips**""",
            )
        ],
    ),
    ResponseFunctionToolCallParam(
        type="function_call",
        call_id=WEB_SEARCH_CALL_ID_1,
        name="web_search",
        arguments=WEB_SEARCH_ARGS_1,
    ),
    FunctionCallOutput(
        type="function_call_output",
        call_id=WEB_SEARCH_CALL_ID_1,
        output=WEB_SEARCH_OUTPUT_1,
    ),
    ResponseReasoningItemParam(
        id="rs_2",
        type="reasoning",
        encrypted_content="encrypted_reasoning_placeholder",
        summary=[
            Summary(
                type="summary_text",
                text="""**Gathering photography tips**""",
            ),
            Summary(
                type="summary_text",
                text="""**Considering techniques**""",
            ),
        ],
    ),
    ResponseFunctionToolCallParam(
        type="function_call",
        call_id=WEB_SEARCH_CALL_ID_2,
        name="web_search",
        arguments=WEB_SEARCH_ARGS_2,
    ),
    FunctionCallOutput(
        type="function_call_output",
        call_id=WEB_SEARCH_CALL_ID_2,
        output=WEB_SEARCH_OUTPUT_2,
    ),
    ResponseReasoningItemParam(
        id="rs_3",
        type="reasoning",
        encrypted_content="encrypted_reasoning_placeholder",
        summary=[
            Summary(
                type="summary_text",
                text="""## Clarifying exposure""",
            )
        ],
    ),
    ResponseOutputMessageParam(
        id="msg_1",
        role="assistant",
        type="message",
        status="completed",
        phase="final_answer",
        content=[
            ResponseOutputTextParam(
                type="output_text",
                text=ASSISTANT_SCENE_TEXT,
                annotations=[],
            )
        ],
    ),
    Message(
        role="system",
        content=[
            ResponseInputTextParam(
                type="input_text",
                text="From now on, keep answers concise and prefer bullet points over prose.",
            )
        ],
    ),
    EasyInputMessageParam(
        role="user",
        content="Summarize the photography tips in two short bullets.",
    ),
    ResponseOutputMessageParam(
        id="msg_2",
        role="assistant",
        type="message",
        status="completed",
        phase="final_answer",
        content=[
            ResponseOutputTextParam(
                type="output_text",
                text=ASSISTANT_SUMMARY_TEXT,
                annotations=[],
            )
        ],
    ),
]


def test_convert_image_web_search_reasoning_history() -> None:
    """Image + interleaved web_search calls, reasoning skipped, mid-conversation system."""
    input_messages = [ChatMessage(message=item) for item in INPUT_MESSAGES_2]

    expected: list[ChatCompletionMessageParam] = [
        ChatCompletionDeveloperMessageParam(
            role="developer",
            content="You are a helpful assistant. Use web search when you need current information.",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=[
                ChatCompletionContentPartTextParam(
                    type="text",
                    text="""\
Look at this photo. Briefly describe the scene, then use web search to find \
practical tips for photographing sunsets over water like this.""",
                ),
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": "data:image/png;base64,SHORT_TEST_IMAGE", "detail": "auto"},
                ),
            ],
        ),
        ChatCompletionAssistantMessageParam(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageFunctionToolCallParam(
                    id=WEB_SEARCH_CALL_ID_1,
                    type="function",
                    function={"name": "web_search", "arguments": WEB_SEARCH_ARGS_1},
                )
            ],
        ),
        ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=WEB_SEARCH_CALL_ID_1,
            content=WEB_SEARCH_OUTPUT_1,
        ),
        ChatCompletionAssistantMessageParam(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageFunctionToolCallParam(
                    id=WEB_SEARCH_CALL_ID_2,
                    type="function",
                    function={"name": "web_search", "arguments": WEB_SEARCH_ARGS_2},
                )
            ],
        ),
        ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=WEB_SEARCH_CALL_ID_2,
            content=WEB_SEARCH_OUTPUT_2,
        ),
        ChatCompletionAssistantMessageParam(
            role="assistant",
            content=[ChatCompletionContentPartTextParam(type="text", text=ASSISTANT_SCENE_TEXT)],
        ),
        ChatCompletionSystemMessageParam(
            role="system",
            content=[
                ChatCompletionContentPartTextParam(
                    type="text",
                    text="From now on, keep answers concise and prefer bullet points over prose.",
                )
            ],
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content="Summarize the photography tips in two short bullets.",
        ),
        ChatCompletionAssistantMessageParam(
            role="assistant",
            content=[ChatCompletionContentPartTextParam(type="text", text=ASSISTANT_SUMMARY_TEXT)],
        ),
    ]

    assert ChatCompletionsProvider._convert_input_messages(input_messages) == expected


# endregion

# region: Assistant preamble then parallel tools

WEATHER_CALL_ID = "call_O2O3sQ5qFHJzQOCeDGhsHHDX"
STOCK_CALL_ID = "call_8YRKKXqYJQncJMomwkepLQ55"
WEATHER_ARGS = '{"location":"Tokyo","unit":"celsius"}'
STOCK_ARGS = '{"ticker":"AAPL"}'
WEATHER_OUTPUT = '{"temperature": 22, "unit": "celsius", "conditions": "sunny"}'
STOCK_OUTPUT = '{"ticker": "AAPL", "price": 178.50, "currency": "USD"}'
ASSISTANT_PREAMBLE_TEXT = (
    "I'll look up Tokyo's current temperature in Celsius and Apple's current stock price in parallel."
)

INPUT_MESSAGES_3: list[ResponseInputItemParam] = [
    EasyInputMessageParam(
        role="system",
        content=("Before every tool call, first explain what you plan on executing, then call the tool."),
    ),
    EasyInputMessageParam(
        role="user",
        content="What's the weather in Tokyo in celsius, and Apple's stock price? Look both up with the tools.",
    ),
    ResponseReasoningItemParam(
        id="rs_1",
        type="reasoning",
        encrypted_content="encrypted_reasoning_placeholder",
        summary=[Summary(type="summary_text", text="**Planning tool calls**")],
    ),
    ResponseOutputMessageParam(
        id="msg_1",
        role="assistant",
        type="message",
        status="completed",
        phase="commentary",
        content=[
            ResponseOutputTextParam(
                type="output_text",
                text=ASSISTANT_PREAMBLE_TEXT,
                annotations=[],
            )
        ],
    ),
    ResponseFunctionToolCallParam(
        type="function_call",
        call_id=WEATHER_CALL_ID,
        name="get_weather",
        arguments=WEATHER_ARGS,
    ),
    ResponseFunctionToolCallParam(
        type="function_call",
        call_id=STOCK_CALL_ID,
        name="get_stock_price",
        arguments=STOCK_ARGS,
    ),
    FunctionCallOutput(
        type="function_call_output",
        call_id=WEATHER_CALL_ID,
        output=WEATHER_OUTPUT,
    ),
    FunctionCallOutput(
        type="function_call_output",
        call_id=STOCK_CALL_ID,
        output=STOCK_OUTPUT,
    ),
]


def test_convert_assistant_preamble_then_parallel_tools() -> None:
    """Assistant commentary then consecutive function_calls attach to that assistant message."""
    input_messages = [ChatMessage(message=item) for item in INPUT_MESSAGES_3]

    expected: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=("Before every tool call, first explain what you plan on executing, then call the tool."),
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content="What's the weather in Tokyo in celsius, and Apple's stock price? Look both up with the tools.",
        ),
        ChatCompletionAssistantMessageParam(
            role="assistant",
            content=[ChatCompletionContentPartTextParam(type="text", text=ASSISTANT_PREAMBLE_TEXT)],
            tool_calls=[
                ChatCompletionMessageFunctionToolCallParam(
                    id=WEATHER_CALL_ID,
                    type="function",
                    function={"name": "get_weather", "arguments": WEATHER_ARGS},
                ),
                ChatCompletionMessageFunctionToolCallParam(
                    id=STOCK_CALL_ID,
                    type="function",
                    function={"name": "get_stock_price", "arguments": STOCK_ARGS},
                ),
            ],
        ),
        ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=WEATHER_CALL_ID,
            content=WEATHER_OUTPUT,
        ),
        ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=STOCK_CALL_ID,
            content=STOCK_OUTPUT,
        ),
    ]

    assert ChatCompletionsProvider._convert_input_messages(input_messages) == expected


# endregion

# region: Image in function call output

READ_IMAGE_CALL_ID = "call_read_image_1"
READ_IMAGE_ARGS = '{"path": "/tmp/landscape.png"}'
TOOL_RESULT_IMAGE_URL = "data:image/png;base64,SHORT_TEST_IMAGE"

INPUT_MESSAGES_IMAGE_TOOL_RESULT: list[ResponseInputItemParam] = [
    EasyInputMessageParam(
        role="user",
        content="Use the read_image tool to read '/tmp/landscape.png'.",
    ),
    ResponseFunctionToolCallParam(
        type="function_call",
        call_id=READ_IMAGE_CALL_ID,
        name="read_image",
        arguments=READ_IMAGE_ARGS,
    ),
    FunctionCallOutput(
        type="function_call_output",
        call_id=READ_IMAGE_CALL_ID,
        output=cast(
            list[Any],
            [ResponseInputImageContentParam(type="input_image", image_url=TOOL_RESULT_IMAGE_URL, detail="auto")],
        ),
    ),
]


def test_convert_image_in_function_call_output() -> None:
    """Images in function_call_output map to a follow-up user message (tool messages are text-only)."""
    input_messages = [ChatMessage(message=item) for item in INPUT_MESSAGES_IMAGE_TOOL_RESULT]

    expected: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(
            role="user",
            content="Use the read_image tool to read '/tmp/landscape.png'.",
        ),
        ChatCompletionAssistantMessageParam(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageFunctionToolCallParam(
                    id=READ_IMAGE_CALL_ID,
                    type="function",
                    function={"name": "read_image", "arguments": READ_IMAGE_ARGS},
                ),
            ],
        ),
        ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=READ_IMAGE_CALL_ID,
            content="",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=[
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url={"url": TOOL_RESULT_IMAGE_URL, "detail": "auto"},
                ),
            ],
        ),
    ]

    assert ChatCompletionsProvider._convert_input_messages(input_messages) == expected


# endregion

# region: Instructions


def test_convert_instructions_appends_to_existing_developer() -> None:
    """instructions appends to the first developer/system message when one exists."""
    input_messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="developer",
                content="You are a helpful assistant.",
            )
        ),
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Hello",
            )
        ),
    ]
    instructions = "Always answer in one sentence."

    expected: list[ChatCompletionMessageParam] = [
        ChatCompletionDeveloperMessageParam(
            role="developer",
            content="You are a helpful assistant.\nAlways answer in one sentence.",
        ),
        ChatCompletionUserMessageParam(role="user", content="Hello"),
    ]

    assert ChatCompletionsProvider._convert_input_messages(input_messages, instructions) == expected


def test_convert_instructions_creates_system_when_missing() -> None:
    """instructions inserts a system message when no system/developer message exists."""
    input_messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Hello",
            )
        ),
    ]
    instructions = "Always answer in one sentence."

    expected: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content="Always answer in one sentence."),
        ChatCompletionUserMessageParam(role="user", content="Hello"),
    ]

    assert ChatCompletionsProvider._convert_input_messages(input_messages, instructions) == expected


# endregion

# region: Convert tools


def test_convert_tools_skips_builtins() -> None:
    """Built-in / hosted Responses tools are dropped; only function tools remain."""
    tools: list[ToolParam] = [
        WebSearchToolParam(type="web_search"),
        FileSearchToolParam(type="file_search", vector_store_ids=["vs_123"]),
        FunctionToolParam(
            type="function",
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            strict=True,
        ),
        WebSearchToolParam(type="web_search_2025_08_26"),
    ]

    expected: list[ChatCompletionToolUnionParam] = [
        ChatCompletionFunctionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "strict": True,
            },
        ),
    ]

    assert ChatCompletionsProvider._convert_tools(tools) == expected


def test_convert_tools_nests_function_fields() -> None:
    """Flat Responses function tools nest under function; optional fields are preserved."""
    tools: list[ToolParam] = [
        FunctionToolParam(
            type="function",
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
            strict=True,
        ),
        FunctionToolParam(
            type="function",
            name="get_stock_price",
            parameters={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
            strict=False,
        ),
    ]

    expected: list[ChatCompletionToolUnionParam] = [
        ChatCompletionFunctionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
                "strict": True,
            },
        ),
        ChatCompletionFunctionToolParam(
            type="function",
            function={
                "name": "get_stock_price",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
                "strict": False,
            },
        ),
    ]

    assert ChatCompletionsProvider._convert_tools(tools) == expected


def test_convert_tools_drops_responses_only_fields() -> None:
    """Responses-only function fields are dropped; None parameters are omitted."""
    tools: list[ToolParam] = [
        FunctionToolParam(
            type="function",
            name="search_docs",
            description="Search internal docs",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            strict=True,
            allowed_callers=["direct", "programmatic"],
            defer_loading=True,
            output_schema={
                "type": "object",
                "properties": {"results": {"type": "array", "items": {"type": "string"}}},
                "required": ["results"],
            },
        ),
        FunctionToolParam(
            type="function",
            name="ping",
            parameters=None,
            strict=None,
        ),
        WebSearchToolParam(type="web_search"),
        FunctionToolParam(
            type="function",
            name="echo",
            description="Echo a message",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            strict=False,
            defer_loading=False,
        ),
    ]

    expected: list[ChatCompletionToolUnionParam] = [
        ChatCompletionFunctionToolParam(
            type="function",
            function={
                "name": "search_docs",
                "description": "Search internal docs",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ),
        ChatCompletionFunctionToolParam(
            type="function",
            function={
                "name": "ping",
                "strict": None,
            },
        ),
        ChatCompletionFunctionToolParam(
            type="function",
            function={
                "name": "echo",
                "description": "Echo a message",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                "strict": False,
            },
        ),
    ]

    assert ChatCompletionsProvider._convert_tools(tools) == expected


# endregion

# region: Token counting

TOKEN_COUNT_TOOLS: list[ToolParam] = [
    FunctionToolParam(
        type="function",
        name="get_weather",
        description="Get weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
        strict=True,
    ),
    FunctionToolParam(
        type="function",
        name="get_stock_price",
        parameters={
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
        strict=False,
    ),
    WebSearchToolParam(type="web_search"),
]


async def test_count_tokens_tool_call_history() -> None:
    """Counts a multi-turn tool-call conversation with o200k_base."""
    input_messages = [ChatMessage(message=cast(ResponseInputItemParam, item)) for item in INPUT_MESSAGES_1]

    token_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="o200k_base",
    )

    assert token_count


async def test_count_tokens_includes_tools() -> None:
    """Function tools add tokens; hosted tools are ignored by conversion."""
    input_messages = [ChatMessage(message=cast(ResponseInputItemParam, item)) for item in INPUT_MESSAGES_1]

    without_tools = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="o200k_base",
    )
    with_tools = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="o200k_base",
        tools=TOKEN_COUNT_TOOLS,
    )

    assert without_tools
    assert with_tools
    assert with_tools > without_tools


async def test_count_tokens_image_and_reasoning_history() -> None:
    """Image history is counted; reasoning items are dropped before counting."""
    input_messages = [ChatMessage(message=item) for item in INPUT_MESSAGES_2]

    token_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="o200k_base",
    )

    # Placeholder base64 cannot be decoded, so the image uses the low-detail fixed cost.
    assert token_count


async def test_count_tokens_assistant_preamble_with_tools() -> None:
    """Preamble + parallel tool calls, with tool definitions included."""
    input_messages = [ChatMessage(message=item) for item in INPUT_MESSAGES_3]

    token_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="o200k_base",
        tools=TOKEN_COUNT_TOOLS,
    )

    assert token_count


async def test_count_tokens_instructions_increase_count() -> None:
    """instructions are merged into messages and increase the token count."""
    input_messages = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="Hello")),
    ]

    without_instructions = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="o200k_base",
    )
    with_instructions = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="o200k_base",
        instructions="Always answer in one sentence.",
    )

    assert without_instructions
    assert with_instructions
    assert with_instructions > without_instructions


async def test_count_tokens_encoding_from_model_name() -> None:
    """OpenAI model ids resolve via tiktoken; unknown ids fall back to o200k_base."""
    input_messages = [ChatMessage(message=cast(ResponseInputItemParam, item)) for item in INPUT_MESSAGES_1]

    gpt4o_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="gpt-4o",
    )
    cl100k_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="cl100k_base",
    )
    fallback_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=input_messages,
        model="meta-llama/Llama-3",
    )

    assert gpt4o_count
    assert cl100k_count
    assert fallback_count


async def test_count_tokens_real_landscape_image() -> None:
    """Loads landscape.png and applies the OpenAI high-detail tile formula."""
    image_path = Path(__file__).parents[1] / "integration" / "data" / "landscape.png"
    data_url = f"data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode('utf-8')}"

    text_only = [
        ChatMessage(message=EasyInputMessageParam(role="user", content="Describe this photo.")),
    ]
    with_image = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(type="input_text", text="Describe this photo."),
                    ResponseInputImageParam(type="input_image", detail="auto", image_url=data_url),
                ],
            )
        ),
    ]
    with_image_low = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(type="input_text", text="Describe this photo."),
                    ResponseInputImageParam(type="input_image", detail="low", image_url=data_url),
                ],
            )
        ),
    ]

    text_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=text_only,
        model="o200k_base",
    )
    high_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=with_image,
        model="o200k_base",
    )
    low_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=with_image_low,
        model="o200k_base",
    )

    assert text_count
    assert high_count
    assert low_count


async def test_count_tokens_image_in_function_call_output() -> None:
    """Images in function_call_output are included in the local token count via user-message mapping."""
    image_path = Path(__file__).parents[1] / "integration" / "data" / "landscape.png"
    data_url = f"data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode('utf-8')}"

    read_image_tool = FunctionToolParam(
        type="function",
        name="read_image",
        description="Read an image file and return its contents.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the image file."}},
            "required": ["path"],
            "additionalProperties": False,
        },
        strict=True,
    )

    base_messages = [
        ChatMessage(message=EasyInputMessageParam(role="system", content="Describe images briefly.")),
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Use the read_image tool to read '/tmp/landscape.png'.",
            )
        ),
        ChatMessage(
            message=ResponseFunctionToolCallParam(
                type="function_call",
                call_id=READ_IMAGE_CALL_ID,
                name="read_image",
                arguments=READ_IMAGE_ARGS,
            )
        ),
    ]
    without_image = [
        *base_messages,
        ChatMessage(
            message=FunctionCallOutput(
                type="function_call_output",
                call_id=READ_IMAGE_CALL_ID,
                output="",
            )
        ),
    ]
    with_image = [
        *base_messages,
        ChatMessage(
            message=FunctionCallOutput(
                type="function_call_output",
                call_id=READ_IMAGE_CALL_ID,
                output=cast(
                    list[Any],
                    [ResponseInputImageContentParam(type="input_image", image_url=data_url, detail="auto")],
                ),
            )
        ),
    ]

    without_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=without_image,
        model="o200k_base",
        tools=[read_image_tool],
    )
    with_count = await ChatCompletionsProvider.count_tokens(
        client=cast(Any, None),
        input=with_image,
        model="o200k_base",
        tools=[read_image_tool],
    )

    # High-detail tile cost for landscape.png (1536x1024) is 1105.
    assert with_count - without_count >= 1105, (
        f"Expected image in tool result to add at least 1105 tokens, "
        f"got delta {with_count - without_count} (with={with_count}, without={without_count})"
    )


# endregion
