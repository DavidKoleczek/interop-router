"""Tests for individual providers, verifying each provider's functionality independently (no interop)."""

import base64
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputImageParam,
    ResponseInputTextParam,
    WebSearchToolParam,
)
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response_input_item_param import FunctionCallOutput, Message
from openai.types.responses.tool_param import ImageGeneration
import pytest

from interop_router.router import Router
from interop_router.types import ChatMessage, ContextLimitExceededError, ProviderName, SupportedModel

PROVIDER_MODEL_PARAMS = [
    pytest.param("openai", "gpt-5.2"),
    pytest.param("gemini", "gemini-3-flash-preview"),
    pytest.param("anthropic", "claude-haiku-4-5-20251001"),
]


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
    response = await router.create(input=[message], model=model)
    assert response is not None


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

    response = await router.create(
        model=model,
        input=messages,
        max_output_tokens=64_000,
    )
    assert response is not None


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

    response = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        max_output_tokens=64_000,
        truncation="auto",
    )

    assert response is not None

    messages.extend(response.output)
    messages.append(
        ChatMessage(message=EasyInputMessageParam(role="user", content="Wow that is really insightful, thank you"))
    )

    response2 = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        max_output_tokens=64_000,
        truncation="auto",
    )

    assert response2 is not None


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_xhigh_reasoning(router: Router, provider: ProviderName, model: SupportedModel):
    """xhigh reasoning in OpenAI models should corresponding to the highest level of reasoning in the other models:
    - Claude -> thinking_budget_tokens = 32_000
    - Gemini -> thinkingLevel = high
    """
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

    response = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "xhigh", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        max_output_tokens=64_000,
        truncation="auto",
    )

    assert response is not None


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
            content=[
                ResponseInputTextParam(type="input_text", text="What is in this image?"),
                ResponseInputImageParam(type="input_image", image_url=data_url, detail="auto"),
            ],
        )
    )
    response = await router.create(input=[message], model=model)
    assert response is not None


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

    response = await router.create(
        model=model,
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
    )
    assert response is not None

    # Check if get_weather was called and send function output
    messages.extend(response.output)
    for chat_message in response.output:
        msg = chat_message.message
        if msg.get("type") == "function_call" and msg.get("name") == "get_weather":
            call_id = msg.get("call_id", "")
            messages.append(
                ChatMessage(
                    message=FunctionCallOutput(
                        call_id=call_id,
                        output='{"temperature": 22, "unit": "celsius", "conditions": "sunny"}',
                        type="function_call_output",
                    )
                )
            )

            response2 = await router.create(model=model, input=messages, tools=FUNCTION_TOOLS, tool_choice="auto")
            assert response2 is not None
            break


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

    response = await router.create(
        model=model,
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="required",
        instructions="You are a helpful tool calling assistant who calls tools in parallel.",
    )
    assert response is not None

    # Append all function call outputs
    messages.extend(response.output)
    for chat_message in response.output:
        msg = chat_message.message
        if msg.get("type") == "function_call":
            name = msg.get("name", "")
            call_id = msg.get("call_id", "")
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

    response2 = await router.create(
        model=model,
        input=messages,
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
    )
    assert response2 is not None


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

    response = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        tools=FUNCTION_TOOLS,
        instructions="You are a helpful tool calling assistant who calls tools in parallel.",
    )
    assert response is not None

    # Append all function call outputs
    messages.extend(response.output)
    for chat_message in response.output:
        msg = chat_message.message
        if msg.get("type") == "function_call":
            name = msg.get("name", "")
            call_id = msg.get("call_id", "")
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

    response2 = await router.create(
        model=model,
        input=messages,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        tools=FUNCTION_TOOLS,
        tool_choice="auto",
    )
    assert response2 is not None


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
    response = await router.create(
        input=messages,
        model=model,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["web_search_call.results", "web_search_call.action.sources", "reasoning.encrypted_content"],
        tools=[WebSearchToolParam(type="web_search")],
    )
    assert response is not None
    messages.extend(response.output)

    message_2 = ChatMessage(
        message=EasyInputMessageParam(role="user", content="Can pick any of the linked articles and then summarize it?")
    )
    messages.append(message_2)
    response2 = await router.create(
        input=messages,
        model=model,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["web_search_call.results", "web_search_call.action.sources", "reasoning.encrypted_content"],
        tools=[WebSearchToolParam(type="web_search")],
    )
    assert response2 is not None


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_image_gen_basic(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    message = ChatMessage(
        message=EasyInputMessageParam(
            role="user", content="Create a picture of a nano banana dish in a fancy restaurant with a interop theme?"
        ),
        provider_kwargs={"gemini": {"image_config": {"aspect_ratio": "4:3", "image_size": "1K"}}},
    )

    if provider == "openai":
        image_tool_definition = ImageGeneration(
            type="image_generation",
            model="gpt-image-1.5",
            quality="low",
            size="1024x1024",
        )
    elif provider == "gemini":
        image_tool_definition = ImageGeneration(
            type="image_generation",
            model="gemini-3-pro-image-preview",
        )
    else:
        image_tool_definition = FunctionToolParam(
            type="function",
            name="generate_image",
            description="Calls an image generation model using the conversation context as input",
            parameters={},
            strict=True,
        )

    response = await router.create(
        input=[message],
        model=model,
        tools=[image_tool_definition],
    )
    assert response is not None


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_image_gen_two(router: Router, provider: ProviderName, model: SupportedModel):
    client = get_client(provider)
    router.register(provider, client)

    messages = [
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Create a picture of a nano banana dish in a fancy restaurant with a interop theme?",
            ),
            provider_kwargs={"gemini": {"image_config": {"aspect_ratio": "4:3", "image_size": "1K"}}},
        )
    ]

    if provider == "openai":
        image_tool_definition = ImageGeneration(
            type="image_generation",
            model="gpt-image-1.5",
            quality="low",
            size="1024x1024",
        )
    elif provider == "gemini":
        image_tool_definition = ImageGeneration(
            type="image_generation",
            model="gemini-3-pro-image-preview",
        )
    else:
        image_tool_definition = FunctionToolParam(
            type="function",
            name="generate_image",
            description="Calls an image generation model using the conversation context as input",
            parameters={},
            strict=True,
        )

    response = await router.create(
        input=messages,
        model=model,
        reasoning={"effort": "medium", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        tools=[image_tool_definition],
    )
    assert response is not None

    messages.extend(response.output)

    # For providers without native image generation, provide function output
    if provider not in ("openai", "gemini"):
        for chat_message in response.output:
            msg = chat_message.message
            if msg.get("type") == "function_call" and msg.get("name") == "generate_image":
                call_id = msg.get("call_id", "")
                messages.append(
                    ChatMessage(
                        message=FunctionCallOutput(
                            call_id=call_id,
                            output="Image generation is not implemented yet, let the user know there was an error",
                            type="function_call_output",
                        )
                    )
                )

    messages.append(
        ChatMessage(
            message=EasyInputMessageParam(
                role="user",
                content="Can you make the banana golden and add some sparkles around it?",
            ),
            provider_kwargs={"gemini": {"image_config": {"aspect_ratio": "4:3", "image_size": "1K"}}},
        )
    )

    response2 = await router.create(
        input=messages,
        model=model,
        tools=[image_tool_definition],
    )
    assert response2 is not None


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_context_limit_exceeded(
    router: Router,
    provider: ProviderName,
    model: SupportedModel,
) -> None:
    """Test that context limit errors are raised when input exceeds limits."""

    def generate_large_content(target_tokens: int) -> str:
        """Generate content large enough to exceed typical context limits."""
        base_text = "This is a test message with content to fill the context window. "
        chars_needed = target_tokens * 4
        repetitions = chars_needed // len(base_text) + 1
        return base_text * repetitions

    client = get_client(provider)
    router.register(provider, client)

    large_content = generate_large_content(target_tokens=2_000_000)
    messages = [
        ChatMessage(
            message=EasyInputMessageParam(role="user", content=large_content),
        )
    ]

    with pytest.raises(ContextLimitExceededError):
        await router.create(input=messages, model=model)


@pytest.mark.parametrize(("provider", "model"), PROVIDER_MODEL_PARAMS)
async def test_background_mode(
    router: Router,
    provider: ProviderName,
    model: SupportedModel,
) -> None:
    client = get_client(provider)
    router.register(provider, client)

    message = ChatMessage(
        message=EasyInputMessageParam(role="user", content="What is 2 + 2? Reply with just the number.")
    )
    response = await router.create(input=[message], model=model, background=True)
    assert response is not None


async def test_azure_openai_client():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    azure_token = os.getenv("AZURE_OPENAI_AD_TOKEN")

    client = AsyncOpenAI(
        base_url=f"{azure_endpoint}/openai/v1/",
        api_key=azure_token,
    )

    router = Router()
    router.register("openai", client)

    message = ChatMessage(message=EasyInputMessageParam(role="user", content="Hello!"))
    response = await router.create(input=[message], model="gpt-5-mini")
    assert response is not None
