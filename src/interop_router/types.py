from collections.abc import Iterable
from dataclasses import dataclass, field
import datetime
import json
from typing import Any, Literal, Protocol, TypeAlias
import uuid

from openai.types.responses import (
    ResponseError,
    ResponseIncludable,
    ResponseInputItemParam,
    ResponseTextConfigParam,
    ResponseUsage,
    ToolParam,
    response_create_params,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.shared_params.reasoning import Reasoning

ProviderName: TypeAlias = Literal["openai", "gemini", "anthropic"]
CreatedBy: TypeAlias = Literal["user"] | ProviderName

SupportedModelOpenAI: TypeAlias = Literal[
    "gpt-5.2-codex",
    "gpt-5.2",
    "gpt-5.2-2025-12-11",
    "gpt-5.2-chat-latest",
    "gpt-5.2-pro",
    "gpt-5.2-pro-2025-12-11",
    "gpt-5-codex",
    "gpt-5-pro",
    "gpt-5-pro-2025-10-06",
    "gpt-5.1-codex-max",
    "gpt-5.1",
    "gpt-5.1-2025-11-13",
    "gpt-5.1-codex",
    "gpt-5.1-chat-latest",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5-chat-latest",
]

SupportedModelGemini: TypeAlias = Literal[
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]

SupportedModelAnthropic: TypeAlias = Literal[
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
]

SupportedModel: TypeAlias = SupportedModelOpenAI | SupportedModelGemini | SupportedModelAnthropic


@dataclass
class ChatMessage:
    """
    This needs to be a dataclass because of the serialization issues with Pydantic and OpenAI types.
    """

    message: ResponseInputItemParam
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.UTC))
    created_by: CreatedBy = "user"
    interop: dict[ProviderName, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_kwargs: dict[ProviderName, dict[str, Any]] = field(default_factory=dict)
    original_response: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate that timestamp is timezone-aware.

        Raises:
            ValueError: If timestamp does not have timezone information.
        """
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware. Use datetime.now(UTC) or ensure your datetime has timezone info."
            )

    def model_dump_json(self) -> str:
        """Serialize ChatMessage to JSON string.

        Returns:
            JSON string representation of the ChatMessage.
        """
        return json.dumps(
            obj={
                "id": self.id,
                "timestamp": self.timestamp.isoformat(),
                "message": dict(self.message),
                "created_by": self.created_by,
                "interop": self.interop,
                "metadata": self.metadata,
                "provider_kwargs": self.provider_kwargs,
                "original_response": self.original_response,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ChatMessage":
        """Deserialize ChatMessage from JSON string.

        Args:
            json_str: JSON string containing serialized ChatMessage data.

        Returns:
            ChatMessage instance.

        Raises:
            ValueError: If timestamp string cannot be parsed or is missing timezone.
            json.JSONDecodeError: If json_str is not valid JSON.
            KeyError: If required fields are missing from JSON.
        """
        data = json.loads(json_str)
        timestamp_str = data["timestamp"]
        timestamp = datetime.datetime.fromisoformat(timestamp_str)

        return cls(
            id=data["id"],
            timestamp=timestamp,
            message=data["message"],
            created_by=data.get("created_by", "user"),
            interop=data.get("interop", {}),
            metadata=data.get("metadata", {}),
            provider_kwargs=data.get("provider_kwargs", {}),
            original_response=data.get("original_response"),
        )


@dataclass
class RouterResponse:
    """Response from a router provider call."""

    output: list[ChatMessage]
    error: ResponseError | None = None
    incomplete_details: IncompleteDetails | None = None
    usage: ResponseUsage | None = None


class InteropRouterError(Exception):
    def __init__(self, message: str, provider: ProviderName, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.cause = cause


class ContextLimitExceededError(InteropRouterError):
    """Raised when the input context exceeds the provider's limit."""


class ResponsesAPIProtocol(Protocol):
    """
    Protocol defining the interface for provider-agnostic Responses API.
    Uses OpenAI's Responses API types as the common denominator.
    """

    async def create(
        self,
        *,
        client: Any,
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
    ) -> RouterResponse: ...
