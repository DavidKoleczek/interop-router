from collections.abc import Iterable
from typing import Any, Literal, NamedTuple, get_args, overload

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import ResponseIncludable, ResponseTextConfigParam, response_create_params
from openai.types.responses.tool_param import ToolParam
from openai.types.shared_params.reasoning import Reasoning

from interop_router.anthropic_provider import AnthropicProvider
from interop_router.chat_completions_provider import ChatCompletionsProvider
from interop_router.gemini_provider import GeminiProvider
from interop_router.openai_provider import OpenAIProvider
from interop_router.types import (
    ChatMessage,
    ModelRef,
    ProviderName,
    RouterResponse,
    RouterStream,
    SupportedModelAnthropic,
    SupportedModelGemini,
    SupportedModelOpenAI,
)

_PROVIDER_NAMES: frozenset[str] = frozenset(get_args(ProviderName))
_PROVIDER_BY_PREFIX: dict[str, ProviderName] = {
    "openai": "openai",
    "gemini": "gemini",
    "anthropic": "anthropic",
    "chat_completions": "chat_completions",
}
_OPENAI_MODELS: frozenset[str] = frozenset(get_args(SupportedModelOpenAI))
_GEMINI_MODELS: frozenset[str] = frozenset(get_args(SupportedModelGemini))
_ANTHROPIC_MODELS: frozenset[str] = frozenset(get_args(SupportedModelAnthropic))


def resolve_model(model: ModelRef) -> tuple[ProviderName, str]:
    """Resolve a model reference to a provider and API model id.

    Prefixed forms use only the first ``/`` for disambiguation.
    When the segment before it is a known provider name, that selects the provider and the remainder
    (which may itself contain ``/``) is the API model id.
    Otherwise the full string is looked up in the supported-model catalogs.
    """
    if "/" in model:
        prefix, api_model = model.split("/", 1)
        provider = _PROVIDER_BY_PREFIX.get(prefix)
        if provider is not None:
            if not api_model:
                raise ValueError(f"Empty model id in model reference: {model!r}")
            return provider, api_model

    if model in _OPENAI_MODELS:
        return "openai", model
    if model in _GEMINI_MODELS:
        return "gemini", model
    if model in _ANTHROPIC_MODELS:
        return "anthropic", model

    raise ValueError(
        f"Unknown model: {model!r}. Use a supported bare model id, or a "
        f"'provider/model' reference (providers: {', '.join(sorted(_PROVIDER_NAMES))})."
    )


class _Registration(NamedTuple):
    """A registered client and the provider adapter that handles its requests."""

    provider: ProviderName
    client: Any


class Router:
    """Router that dispatches API calls to the appropriate provider based on model type."""

    def __init__(self) -> None:
        self._registrations: dict[str, _Registration] = {}

    def register(
        self,
        provider_name: ProviderName,
        client: AsyncOpenAI | genai.Client | AsyncAnthropic,
        name: str | None = None,
    ) -> None:
        """Register a client for a provider.

        Args:
            provider_name: Provider whose adapter handles requests for this client.
            client: Provider SDK client instance.
            name: Optional registration name, used as the prefix in ``name/model`` references.
            Defaults to ``provider_name``.

        Raises:
            ValueError: If ``name`` is empty, contains ``/``, or is the name of a different provider.
        """
        registration_name = provider_name if name is None else name
        if not registration_name:
            raise ValueError("Registration name must be a non-empty string.")
        if "/" in registration_name:
            raise ValueError(f"Registration name must not contain '/': {registration_name!r}")
        if registration_name in _PROVIDER_NAMES and registration_name != provider_name:
            raise ValueError(f"Registration name {registration_name!r} is reserved for the provider of the same name.")

        self._registrations[registration_name] = _Registration(provider=provider_name, client=client)

    def _resolve(self, model: ModelRef) -> tuple[ProviderName, str, Any]:
        """Resolve a model reference to a provider, API model id, and registered client.

        Registration names take precedence over the provider catalogs so that a named client can be addressed directly. Anything else falls back to `resolve_model`.
        """
        if "/" in model:
            prefix, api_model = model.split("/", 1)
            registration = self._registrations.get(prefix)
            if registration is not None:
                if not api_model:
                    raise ValueError(f"Empty model id in model reference: {model!r}")
                return registration.provider, api_model, registration.client

        provider, api_model = resolve_model(model)
        return provider, api_model, self._client_for(provider)

    def _client_for(self, provider: ProviderName) -> Any:
        """Return the client registered for a provider when it is unambiguous.

        Raises:
            ValueError: If no client is registered for the provider, or if several are registered under different names.
        """
        default = self._registrations.get(provider)
        if default is not None:
            return default.client

        matches = {name: entry for name, entry in self._registrations.items() if entry.provider == provider}
        if not matches:
            raise ValueError(f"No client registered for provider: {provider}")
        if len(matches) > 1:
            names = ", ".join(sorted(matches))
            raise ValueError(
                f"Multiple clients are registered for provider {provider!r}: {names}. "
                f"Use a 'name/model' reference to select one."
            )
        return next(iter(matches.values())).client

    # Non-streaming overload: `stream` omitted or False narrows the return to RouterResponse.
    @overload
    async def create(
        self,
        *,
        input: list[ChatMessage],
        model: ModelRef,
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
        model: ModelRef,
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
        model: ModelRef,
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
        model: ModelRef,
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
            model: Bare catalog model id, or ``provider/model`` reference. When clients are registered under custom names, ``name/model`` selects one of them.
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
            ValueError: If the model is unknown, if no client is registered for the required provider,
            if several clients are registered for it and the reference does not name one, or if stream=True is combined with background=True.
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
        model: ModelRef,
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
        provider, api_model, client = self._resolve(model)

        if provider == "openai":
            return await OpenAIProvider.create(
                client=client,
                input=input,
                model=api_model,
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
                model=api_model,
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

        if provider == "chat_completions":
            return await ChatCompletionsProvider.create(
                client=client,
                input=input,
                model=api_model,
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

        return await AnthropicProvider.create(
            client=client,
            input=input,
            model=api_model,
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
        model: ModelRef,
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
        provider, api_model, client = self._resolve(model)

        if provider == "openai":
            return await OpenAIProvider.create_stream(
                client=client,
                input=input,
                model=api_model,
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
                model=api_model,
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

        if provider == "chat_completions":
            return await ChatCompletionsProvider.create_stream(
                client=client,
                input=input,
                model=api_model,
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
            model=api_model,
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
        model: ModelRef,
        instructions: str | None = None,
        reasoning: Reasoning | None = None,
        tools: Iterable[ToolParam] | None = None,
    ) -> int:
        """Count input tokens for the given messages and configuration.

        Uses the provider's native token counting endpoint.

        Args:
            input: List of chat messages.
            model: Bare catalog model id, or ``provider/model`` reference. When clients are registered under custom names, ``name/model`` selects one of them.
            instructions: Optional system instructions.
            reasoning: Optional reasoning configuration.
            tools: Optional list of tools.

        Returns:
            Token count estimate for the input.

        Raises:
            ValueError: If the model is unknown, if no client is registered for the required provider,
            or if several clients are registered for it and the reference does not name one.
        """
        provider, api_model, client = self._resolve(model)

        if provider == "openai":
            return await OpenAIProvider.count_tokens(
                client=client,
                input=input,
                model=api_model,
                instructions=instructions,
                reasoning=reasoning,
                tools=tools,
            )

        if provider == "gemini":
            return await GeminiProvider.count_tokens(
                client=client,
                input=input,
                model=api_model,
                instructions=instructions,
                reasoning=reasoning,
                tools=tools,
            )

        if provider == "chat_completions":
            return await ChatCompletionsProvider.count_tokens(
                client=client,
                input=input,
                model=api_model,
                instructions=instructions,
                reasoning=reasoning,
                tools=tools,
            )

        return await AnthropicProvider.count_tokens(
            client=client,
            input=input,
            model=api_model,
            instructions=instructions,
            reasoning=reasoning,
            tools=tools,
        )
