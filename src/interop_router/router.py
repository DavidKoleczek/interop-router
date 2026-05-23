from collections.abc import Iterable
from typing import Any, Literal, cast, get_args, overload

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
    RouterStream,
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

    # Non-streaming overload: `stream` omitted or False narrows the return to RouterResponse.
    @overload
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
        top_logprobs: int | None = None,
        top_p: float | None = None,
        truncation: Literal["auto", "disabled"] | None = None,
        background: bool | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        stream: Literal[False] | None = None,
    ) -> RouterResponse: ...

    # Streaming overload: `stream=True` is required (no default) and narrows the return to RouterStream.
    @overload
    async def create(
        self,
        *,
        stream: Literal[True],
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
        top_logprobs: int | None = None,
        top_p: float | None = None,
        truncation: Literal["auto", "disabled"] | None = None,
        background: bool | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> RouterStream: ...

    # Runtime-bool fallback: required when callers pass a `bool` variable that the
    # type checker cannot narrow to `Literal[True]` or `Literal[False]`.
    @overload
    async def create(
        self,
        *,
        stream: bool,
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
        top_logprobs: int | None = None,
        top_p: float | None = None,
        truncation: Literal["auto", "disabled"] | None = None,
        background: bool | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> RouterResponse | RouterStream: ...

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
        top_logprobs: int | None = None,
        top_p: float | None = None,
        truncation: Literal["auto", "disabled"] | None = None,
        background: bool | None = None,
        provider_kwargs: dict[str, Any] | None = None,
        stream: bool | None = None,
    ) -> RouterResponse | RouterStream:
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
            top_logprobs: Optional number of most likely tokens to return at each
              position (0-20). Only supported with reasoning effort "none".
            top_p: Optional nucleus sampling parameter. Only supported with
              reasoning effort "none".
            truncation: Optional truncation setting.
            background: Whether to run the model response in the background.
              Mutually exclusive with stream=True.
            provider_kwargs: Optional provider-specific keyword arguments.
              See the PROVIDER_GUIDE.md for details.
            stream: If set to true, the model response data will be streamed as events are generated.
              The last event will always be RouterResponse

        Returns:
            A RouterResponse when stream is False or omitted, otherwise a RouterStream.

        Raises:
            ValueError: If no client is registered for the required provider, if the
                model is unknown, or if stream=True is combined with background=True.
        """
        if stream and background:
            raise ValueError("stream=True is not compatible with background=True")

        if stream:
            return await self._dispatch_stream(
                input=input,
                model=model,
                include=include,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                parallel_tool_calls=parallel_tool_calls,
                reasoning=reasoning,
                temperature=temperature,
                text=text,
                tool_choice=tool_choice,
                tools=tools,
                top_logprobs=top_logprobs,
                top_p=top_p,
                truncation=truncation,
            )

        return await self._dispatch_create(
            input=input,
            model=model,
            include=include,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            parallel_tool_calls=parallel_tool_calls,
            reasoning=reasoning,
            temperature=temperature,
            text=text,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            truncation=truncation,
            background=background,
            provider_kwargs=provider_kwargs,
        )

    async def _dispatch_create(
        self,
        *,
        input: list[ChatMessage],
        model: SupportedModel,
        include: list[ResponseIncludable] | None,
        instructions: str | None,
        max_output_tokens: int | None,
        parallel_tool_calls: bool | None,
        reasoning: Reasoning | None,
        temperature: float | None,
        text: ResponseTextConfigParam | None,
        tool_choice: response_create_params.ToolChoice | None,
        tools: Iterable[ToolParam] | None,
        top_logprobs: int | None,
        top_p: float | None,
        truncation: Literal["auto", "disabled"] | None,
        background: bool | None,
        provider_kwargs: dict[str, Any] | None,
    ) -> RouterResponse:
        provider = self._get_provider_for_model(model)
        client = self._clients.get(provider)
        if client is None:
            raise ValueError(f"No client registered for provider: {provider}")

        if provider == "openai":
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
                top_logprobs=top_logprobs,
                top_p=top_p,
                truncation=truncation,
                background=background,
            )

        if provider == "gemini":
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
                top_logprobs=top_logprobs,
                top_p=top_p,
                truncation=truncation,
            )

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
            top_logprobs=top_logprobs,
            top_p=top_p,
            truncation=truncation,
            provider_kwargs=provider_kwargs,
        )

    async def _dispatch_stream(
        self,
        *,
        input: list[ChatMessage],
        model: SupportedModel,
        include: list[ResponseIncludable] | None,
        instructions: str | None,
        max_output_tokens: int | None,
        parallel_tool_calls: bool | None,
        reasoning: Reasoning | None,
        temperature: float | None,
        text: ResponseTextConfigParam | None,
        tool_choice: response_create_params.ToolChoice | None,
        tools: Iterable[ToolParam] | None,
        top_logprobs: int | None,
        top_p: float | None,
        truncation: Literal["auto", "disabled"] | None,
    ) -> RouterStream:
        provider = self._get_provider_for_model(model)
        client = self._clients.get(provider)
        if client is None:
            raise ValueError(f"No client registered for provider: {provider}")

        if provider == "openai":
            return await OpenAIProvider.create_stream(
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
                top_logprobs=top_logprobs,
                top_p=top_p,
                truncation=truncation,
            )

        if provider == "gemini":
            return await GeminiProvider.create_stream(
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
                top_logprobs=top_logprobs,
                top_p=top_p,
                truncation=truncation,
            )

        return await AnthropicProvider.create_stream(
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
            top_logprobs=top_logprobs,
            top_p=top_p,
            truncation=truncation,
        )

    async def count_tokens(
        self,
        *,
        input: list[ChatMessage],
        model: SupportedModel,
        instructions: str | None = None,
        reasoning: Reasoning | None = None,
        tools: Iterable[ToolParam] | None = None,
    ) -> int:
        """Count input tokens for the given messages and configuration.

        Uses the provider's native token counting endpoint.

        Args:
            input: List of chat messages.
            model: The model to use for token counting.
            instructions: Optional system instructions.
            reasoning: Optional reasoning configuration.
            tools: Optional list of tools.

        Returns:
            Token count estimate for the input.

        Raises:
            ValueError: If no client is registered for the required provider or if the
                provider does not support token counting.
        """
        if model in get_args(SupportedModelOpenAI):
            client = self._clients.get("openai")
            if client is None:
                raise ValueError("No client registered for provider: openai")
            return await OpenAIProvider.count_tokens(
                client=client,
                input=input,
                model=cast(SupportedModelOpenAI, model),
                instructions=instructions,
                reasoning=reasoning,
                tools=tools,
            )

        if model in get_args(SupportedModelGemini):
            client = self._clients.get("gemini")
            if client is None:
                raise ValueError("No client registered for provider: gemini")
            return await GeminiProvider.count_tokens(
                client=client,
                input=input,
                model=cast(SupportedModelGemini, model),
                instructions=instructions,
                reasoning=reasoning,
                tools=tools,
            )

        if model in get_args(SupportedModelAnthropic):
            client = self._clients.get("anthropic")
            if client is None:
                raise ValueError("No client registered for provider: anthropic")
            return await AnthropicProvider.count_tokens(
                client=client,
                input=input,
                model=cast(SupportedModelAnthropic, model),
                instructions=instructions,
                reasoning=reasoning,
                tools=tools,
            )

        raise ValueError(f"Unknown model: {model}")
