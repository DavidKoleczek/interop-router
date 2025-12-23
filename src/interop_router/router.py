from collections.abc import Iterable
from typing import Any, Literal, cast, get_args

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import ResponseIncludable, ResponseTextConfigParam, response_create_params
from openai.types.responses.tool_param import ToolParam
from openai.types.shared_params.reasoning import Reasoning

from interop_router.anthropic_provider import AnthropicProvider
from interop_router.gemini_provider import GeminiProvider
from interop_router.openai_provider import OpenAIProvider
from interop_router.types import (
    ChatMessage,
    ProviderName,
    RouterResponse,
    SupportedModel,
    SupportedModelAnthropic,
    SupportedModelGemini,
    SupportedModelOpenAI,
)


class Router:
    """Router that dispatches API calls to the appropriate provider based on model type."""

    def __init__(self) -> None:
        self._clients: dict[ProviderName, Any] = {}

    def register(self, provider_name: ProviderName, client: AsyncOpenAI | genai.Client | AsyncAnthropic) -> None:
        self._clients[provider_name] = client

    def _get_provider_for_model(self, model: SupportedModel) -> ProviderName:
        if model in get_args(SupportedModelOpenAI):
            return "openai"
        if model in get_args(SupportedModelGemini):
            return "gemini"
        if model in get_args(SupportedModelAnthropic):
            return "anthropic"
        raise ValueError(f"Unknown model: {model}")

    async def create(
        self,
        *,
        input: list[ChatMessage],
        model: SupportedModel,
        include: list[ResponseIncludable] | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        parallel_tool_calls: bool | None = None,
        reasoning: Reasoning | None = None,
        temperature: float | None = None,
        text: ResponseTextConfigParam | None = None,
        tool_choice: response_create_params.ToolChoice | None = None,
        tools: Iterable[ToolParam] | None = None,
        truncation: Literal["auto", "disabled"] | None = None,
    ) -> RouterResponse:
        """Create a response using the appropriate provider for the given model.

        Args:
            input: List of chat messages.
            model: The model to use for generation.
            include: Optional list of response includables.
            instructions: Optional system instructions.
            max_output_tokens: Optional maximum output tokens.
            parallel_tool_calls: Optional flag for parallel tool calls.
            reasoning: Optional reasoning configuration.
            temperature: Optional temperature setting.
            text: Optional text configuration.
            tool_choice: Optional tool choice configuration.
            tools: Optional list of tools.
            truncation: Optional truncation setting.

        Returns:
            The response from the provider.

        Raises:
            ValueError: If no client is registered for the required provider.
        """
        if model in get_args(SupportedModelOpenAI):
            client = self._clients.get("openai")
            if client is None:
                raise ValueError("No client registered for provider: openai")
            return await OpenAIProvider.create(
                client=client,
                input=input,
                model=cast(SupportedModelOpenAI, model),
                include=include,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                parallel_tool_calls=parallel_tool_calls,
                reasoning=reasoning,
                temperature=temperature,
                text=text,
                tool_choice=tool_choice,
                tools=tools,
                truncation=truncation,
            )

        if model in get_args(SupportedModelGemini):
            client = self._clients.get("gemini")
            if client is None:
                raise ValueError("No client registered for provider: gemini")
            return await GeminiProvider.create(
                client=client,
                input=input,
                model=cast(SupportedModelGemini, model),
                include=include,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                parallel_tool_calls=parallel_tool_calls,
                reasoning=reasoning,
                temperature=temperature,
                text=text,
                tool_choice=tool_choice,
                tools=tools,
                truncation=truncation,
            )

        if model in get_args(SupportedModelAnthropic):
            client = self._clients.get("anthropic")
            if client is None:
                raise ValueError("No client registered for provider: anthropic")
            return await AnthropicProvider.create(
                client=client,
                input=input,
                model=cast(SupportedModelAnthropic, model),
                include=include,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                parallel_tool_calls=parallel_tool_calls,
                reasoning=reasoning,
                temperature=temperature,
                text=text,
                tool_choice=tool_choice,
                tools=tools,
                truncation=truncation,
            )

        raise ValueError(f"Unknown model: {model}")
