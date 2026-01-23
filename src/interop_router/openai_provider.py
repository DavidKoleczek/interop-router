from collections.abc import Iterable
import json
from typing import Any, Literal, cast
import uuid

import openai
from openai import AsyncOpenAI
from openai._types import omit
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCallParam,
    ResponseFunctionWebSearch,
    ResponseIncludable,
    ResponseInputImageContentParam,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseTextConfigParam,
    response_create_params,
)
from openai.types.responses.response_function_web_search import ActionFind, ActionOpenPage, ActionSearch
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.tool_param import ToolParam
from openai.types.shared_params.reasoning import Reasoning

from interop_router.types import (
    ChatMessage,
    ContextLimitExceededError,
    ProviderName,
    RouterResponse,
    SupportedModelOpenAI,
)


class OpenAIProvider:
    PROVIDER_NAME = "openai"

    @staticmethod
    async def create(
        *,
        client: AsyncOpenAI,
        input: list[ChatMessage],
        model: SupportedModelOpenAI,
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
        input_messages = OpenAIProvider._prepare_input_messages(input)
        try:
            response_stream = await client.responses.create(
                model=model,
                input=input_messages,
                include=include if include is not None else omit,
                instructions=instructions if instructions is not None else omit,
                max_output_tokens=max_output_tokens if max_output_tokens is not None else omit,
                parallel_tool_calls=parallel_tool_calls if parallel_tool_calls is not None else omit,
                reasoning=reasoning if reasoning is not None else omit,
                temperature=temperature if temperature is not None else omit,
                text=text if text is not None else omit,
                tool_choice=tool_choice if tool_choice is not None else omit,
                tools=tools if tools is not None else omit,
                truncation=truncation if truncation is not None else omit,
                store=False,
                stream=True,
            )
            response: Response | None = None
            async for event in response_stream:
                if isinstance(event, ResponseCompletedEvent):
                    response = event.response
                    break
            if response is None:
                raise RuntimeError("Response stream ended without a completed response event.")
        except openai.BadRequestError as e:
            if e.code == "context_length_exceeded":
                raise ContextLimitExceededError(str(e), provider="openai", cause=e) from e
            raise

        converted_output = OpenAIProvider._convert_to_chat_messages(response.output)
        if converted_output:
            converted_output[-1].original_response = response.model_dump(mode="json")
        return RouterResponse(
            output=converted_output,
            error=response.error,
            incomplete_details=response.incomplete_details,
            usage=response.usage,
        )

    @staticmethod
    def _prepare_input_messages(messages: list[ChatMessage]) -> list[ResponseInputItemParam]:
        """Filters out reasoning from other providers, expands Gemini web search
        metadata into function call pairs, and removes id/status fields.
        """
        expanded: list[ResponseInputItemParam] = []
        for msg in messages:
            is_foreign_reasoning = (
                msg.message.get("type") == "reasoning"
                and msg.created_by != "user"
                and msg.created_by != OpenAIProvider.PROVIDER_NAME
            )
            if is_foreign_reasoning:
                continue
            expanded.extend(OpenAIProvider._expand_gemini_web_search(msg))

        sanitized = []
        for item in expanded:
            sanitized.append(cast(ResponseInputItemParam, {k: v for k, v in item.items() if k not in ("id", "status")}))
        return sanitized

    @staticmethod
    def _expand_gemini_web_search(chat_message: ChatMessage) -> list[ResponseInputItemParam]:
        """Expand a Gemini message with web search metadata into OpenAI function call format.

        If the message has Gemini grounding_metadata, returns:
        - A function_call for "web_search" with args from web_search_queries
        - A function_call_output with grounding_chunks as the result
        - The original message

        Otherwise returns just the original message.
        """
        grounding_metadata: dict[str, Any] | None = chat_message.interop.get("gemini", {}).get("grounding_metadata")
        if grounding_metadata is None:
            return [chat_message.message]

        call_id = str(uuid.uuid4())

        web_search_queries = grounding_metadata.get("web_search_queries", [])
        arguments = json.dumps({"queries": web_search_queries}) if web_search_queries else "{}"

        function_call: ResponseFunctionToolCallParam = {
            "type": "function_call",
            "call_id": call_id,
            "name": "web_search",
            "arguments": arguments,
        }

        grounding_chunks = grounding_metadata.get("grounding_chunks", [])
        function_call_output: FunctionCallOutput = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(grounding_chunks),
        }

        return [function_call, function_call_output, chat_message.message]

    @staticmethod
    def _convert_to_chat_messages(output_items: list[ResponseOutputItem]) -> list[ChatMessage]:
        chat_messages: list[ChatMessage] = []
        for output_item in output_items:
            match output_item.type:
                case "reasoning" | "function_call" | "message":
                    input_param = cast(
                        ResponseInputItemParam, output_item.model_dump(exclude={"status", "id"}, exclude_unset=True)
                    )
                    chat_messages.append(ChatMessage(message=input_param, created_by="openai"))
                case "image_generation_call":
                    chat_messages.extend(OpenAIProvider._convert_image_generation_call(output_item))
                case "web_search_call":
                    output_item = cast(ResponseFunctionWebSearch, output_item)
                    chat_messages.extend(OpenAIProvider._convert_web_search_call(output_item))
                case _:
                    raise ValueError(f"Currently unsupported output item type: {output_item.type}")
        return chat_messages

    @staticmethod
    def _convert_image_generation_call(output_item: ResponseOutputItem) -> list[ChatMessage]:
        """Converts an ImageGenerationCall output item into ResponseFunctionToolCallParam and FunctionCallOutput messages.
        This needs special handling because when store=False, OpenAI does not let you pass the ImageGenerationCall back
        """
        call_id = output_item.id or str(uuid.uuid4())
        function_call: ResponseFunctionToolCallParam = {
            "type": "function_call",
            "call_id": call_id,
            "name": "image_generation",
            "arguments": "{}",
        }
        base64_image = getattr(output_item, "result", None)
        image_url = f"data:image/png;base64,{base64_image}" if base64_image else None
        function_call_output: FunctionCallOutput = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": [ResponseInputImageContentParam(type="input_image", detail="auto", image_url=image_url)],
        }

        extra_fields = output_item.model_dump(
            exclude={"id", "type", "result", "status"},
            exclude_unset=True,
        )
        interop: dict[ProviderName, Any] = {"openai": {"image_generation": extra_fields}} if extra_fields else {}

        return [
            ChatMessage(message=function_call, created_by="openai", interop=interop),
            ChatMessage(message=function_call_output, created_by="openai"),
        ]

    @staticmethod
    def _convert_web_search_call(output_item: ResponseFunctionWebSearch) -> list[ChatMessage]:
        """Converts a built-in web search call into a ResponseFunctionToolCallParam and FunctionCallOutput.
        This is because the built-in web search call data is not persisted, so the model cannot see what it
        searched for or what results were found.
        """
        call_id = output_item.id
        action = output_item.action
        base_message = """The complete result of this tool call was removed. Please call a web search tool if you need the data again. \
Importantly, the web_search tool definition may have changed, including the name of the tool itself. Be sure to use the latest definition."""

        if isinstance(action, ActionSearch):
            arguments = {"type": "search", "query": action.query}
            if action.sources:
                sources_list = [source.url for source in action.sources]
                output_content = base_message + "\n\nSources found:\n" + "\n".join(f"- {url}" for url in sources_list)
            else:
                output_content = base_message
        elif isinstance(action, ActionOpenPage):
            arguments = {"type": "open_page", "url": action.url}
            output_content = base_message
        elif isinstance(action, ActionFind):
            arguments = {"type": "find", "pattern": action.pattern, "url": action.url}
            output_content = base_message
        else:
            arguments = {}
            output_content = base_message

        function_call: ResponseFunctionToolCallParam = {
            "type": "function_call",
            "call_id": call_id,
            "name": "web_search",
            "arguments": json.dumps(arguments),
        }
        function_call_output: FunctionCallOutput = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_content,
        }
        return [
            ChatMessage(message=function_call, created_by="openai"),
            ChatMessage(message=function_call_output, created_by="openai"),
        ]
