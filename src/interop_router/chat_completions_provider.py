from collections.abc import AsyncIterator, Iterable
import json
import time
from typing import Any, ClassVar, Literal, cast
import uuid

import openai
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartRefusalParam,
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
from openai.types.chat.chat_completion_allowed_tool_choice_param import ChatCompletionAllowedToolChoiceParam
from openai.types.chat.chat_completion_named_tool_choice_param import ChatCompletionNamedToolChoiceParam
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    ResponseError,
    ResponseFunctionToolCallParam,
    ResponseIncludable,
    ResponseOutputTextParam,
    ResponseStreamEvent,
    ResponseTextConfigParam,
    ResponseUsage,
    ToolParam,
    response_create_params,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_function_call_arguments_delta_event import ResponseFunctionCallArgumentsDeltaEvent
from openai.types.responses.response_output_message_param import ResponseOutputMessageParam
from openai.types.responses.response_output_refusal_param import ResponseOutputRefusalParam
from openai.types.responses.response_output_text_param import Logprob, LogprobTopLogprob
from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam, Summary
from openai.types.responses.response_reasoning_summary_text_delta_event import ResponseReasoningSummaryTextDeltaEvent
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.shared_params.reasoning import Reasoning

from interop_router.chat_completions_count_tokens import count_tokens as count_chat_completion_tokens
from interop_router.types import ChatMessage, ContextLimitExceededError, RouterResponse, RouterStream


class ChatCompletionsProvider:
    PROVIDER_NAME: ClassVar[Literal["chat_completions"]] = "chat_completions"

    @staticmethod
    async def create(
        *,
        client: AsyncOpenAI,
        input: list[ChatMessage],
        model: str,
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
    ) -> RouterResponse:

        messages = ChatCompletionsProvider._convert_input_messages(input, instructions)
        config = ChatCompletionsProvider._create_config(
            max_output_tokens=max_output_tokens,
            parallel_tool_calls=parallel_tool_calls,
            reasoning=reasoning,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            provider_kwargs=provider_kwargs,
        )

        start_time = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **config,
            )
        except openai.APIError as e:
            if ChatCompletionsProvider._is_context_limit_error(e):
                raise ContextLimitExceededError(str(e), provider="chat_completions", cause=e) from e
            raise
        duration_seconds = time.perf_counter() - start_time

        interop_response = ChatCompletionsProvider._convert_response(response)
        interop_response.duration_seconds = duration_seconds
        return interop_response

    @staticmethod
    async def create_stream(
        *,
        client: AsyncOpenAI,
        input: list[ChatMessage],
        model: str,
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
        provider_kwargs: dict[str, Any] | None = None,
    ) -> RouterStream:
        messages = ChatCompletionsProvider._convert_input_messages(input, instructions)
        config = ChatCompletionsProvider._create_config(
            max_output_tokens=max_output_tokens,
            parallel_tool_calls=parallel_tool_calls,
            reasoning=reasoning,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            provider_kwargs=provider_kwargs,
        )

        start_time = time.perf_counter()
        try:
            sdk_stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **config,
            )
        except openai.APIError as e:
            if ChatCompletionsProvider._is_context_limit_error(e):
                raise ContextLimitExceededError(str(e), provider="chat_completions", cause=e) from e
            raise

        async def _stream() -> AsyncIterator[ResponseStreamEvent | RouterResponse]:
            completion_id = ""
            response_model = model
            content_parts: list[str] = []
            refusal_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: dict[int, dict[str, Any]] = {}
            finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None
            usage: CompletionUsage | None = None
            duration_seconds: float | None = None

            try:
                sequence_number = 0
                async for chunk in sdk_stream:
                    if chunk.id:
                        completion_id = chunk.id
                    if chunk.model:
                        response_model = chunk.model
                    if chunk.usage is not None:
                        usage = chunk.usage

                    events = ChatCompletionsProvider._convert_stream_chunk(chunk, sequence_number)
                    for event in events:
                        yield event
                    sequence_number += len(events)

                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason

                    delta = choice.delta
                    if delta.content:
                        content_parts.append(delta.content)
                    if delta.refusal:
                        refusal_parts.append(delta.refusal)

                    reasoning_delta = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                    if isinstance(reasoning_delta, str) and reasoning_delta:
                        reasoning_parts.append(reasoning_delta)

                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            entry = tool_calls.setdefault(
                                tool_call_delta.index,
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tool_call_delta.id:
                                entry["id"] = tool_call_delta.id
                            if tool_call_delta.type:
                                entry["type"] = tool_call_delta.type
                            if tool_call_delta.function is not None:
                                if tool_call_delta.function.name:
                                    entry["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    entry["function"]["arguments"] += tool_call_delta.function.arguments
            except openai.APIError as e:
                if ChatCompletionsProvider._is_context_limit_error(e):
                    raise ContextLimitExceededError(str(e), provider="chat_completions", cause=e) from e
                raise
            finally:
                await sdk_stream.close()

            duration_seconds = time.perf_counter() - start_time
            final_completion = ChatCompletionsProvider._build_completion_from_stream(
                completion_id=completion_id or f"chatcmpl-{uuid.uuid4()}",
                model=response_model,
                content="".join(content_parts),
                refusal="".join(refusal_parts),
                reasoning="".join(reasoning_parts),
                tool_calls=[tool_calls[index] for index in sorted(tool_calls)],
                finish_reason=finish_reason or "stop",
                usage=usage,
            )
            interop_response = ChatCompletionsProvider._convert_response(final_completion)
            interop_response.duration_seconds = duration_seconds
            yield interop_response

        return _stream()

    @staticmethod
    async def count_tokens(
        *,
        client: AsyncOpenAI,
        input: list[ChatMessage],
        model: str,
        instructions: str | None = None,
        reasoning: Reasoning | None = None,
        tools: Iterable[ToolParam] | None = None,
    ) -> int:
        messages = ChatCompletionsProvider._convert_input_messages(input, instructions)
        chat_tools = None
        if tools is not None:
            converted_tools = ChatCompletionsProvider._convert_tools(tools)
            chat_tools = converted_tools or None
        # `model` is treated as a tiktoken encoding name or OpenAI model id; see chat_completions_count_tokens.count_tokens.
        return count_chat_completion_tokens(messages, encoding=model, tools=chat_tools)

    @staticmethod
    def _is_context_limit_error(error: openai.APIError) -> bool:
        if error.code == "context_length_exceeded":
            return True
        error_text = str(error).lower().replace("_", " ")
        has_limit_indicator = any(indicator in error_text for indicator in ("exceed", "limit", "maximum"))
        return has_limit_indicator and ("context length" in error_text or "input token" in error_text)

    @staticmethod
    def _convert_input_messages(
        input: list[ChatMessage],
        instructions: str | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """Converts Responses API messages to chat completion messages.

        Conversion:
        - `instructions` -> append to the first system/developer message if present
        otherwise insert a new system message with that content

        - role = user, content = str -> ChatCompletionUserMessageParam
        - role = user, content = list -> ChatCompletionUserMessageParam
          - content item type = input_text -> ChatCompletionContentPartTextParam {type="text", text=<same>}
          - content item type = input_image -> ChatCompletionContentPartImageParam, coerce original to auto

        - role=system, content=str -> ChatCompletionSystemMessageParam
        - role=system, content=list -> ChatCompletionSystemMessageParam
          - content item type = input_text -> {type="text", text=<same>}
          - skip input_image / input_file
        - developer is the same, maps to ChatCompletionDeveloperMessageParam

        - role=assistant, content=str -> ChatCompletionAssistantMessageParam {role="assistant", content=<same>}
        - role=assistant, content=list -> ChatCompletionAssistantMessageParam
          - content item type = output_text -> {type="text", text=<same>}
          - content item type = input_text -> {type="text", text=<same>}
          - content item type = refusal -> {type="refusal", refusal=<same>}
          - skip input_image / input_file

        - type=function_call -> append ChatCompletionMessageFunctionToolCallParam to ChatCompletionAssistantMessageParam.tool_calls
        (typed as Iterable[ChatCompletionMessageToolCallUnionParam]) {id=call_id, type="function", function={name, arguments}}
        Aggregate consecutive function_call items into tool_calls on one ChatCompletionAssistantMessageParam.
        Prefer attaching to the preceding assistant message when present; otherwise create assistant with content=None.

        - type=function_call_output -> ChatCompletionToolMessageParam {role="tool", tool_call_id=call_id, content=...}
          - output=str -> content=<same>
          - output=list -> convert input_text parts to text on the tool message (text-only in Chat Completions);
            map input_image parts to a follow-up user message with image_url parts (coerce original to auto);
            skip input_file

        - type=reasoning should be dropped (for now assume chat completions models don't persist reasoning across turns)
        """
        messages = [x.message for x in input]
        chat_completion_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            # EasyInputMessageParam / Message may omit type; role identifies message items.
            if message.get("role") is not None:
                role = message.get("role")
                content = message.get("content", "")
                if isinstance(content, str):
                    if role == "user":
                        chat_completion_messages.append(ChatCompletionUserMessageParam(role="user", content=content))
                    elif role == "system":
                        chat_completion_messages.append(
                            ChatCompletionSystemMessageParam(role="system", content=content)
                        )
                    elif role == "developer":
                        chat_completion_messages.append(
                            ChatCompletionDeveloperMessageParam(role="developer", content=content)
                        )
                    elif role == "assistant":
                        chat_completion_messages.append(
                            ChatCompletionAssistantMessageParam(role="assistant", content=content)
                        )
                elif isinstance(content, list):
                    if role == "user":
                        content_parts = []
                        for content_item in content:
                            if content_item.get("type") == "input_text":
                                content_parts.append(
                                    ChatCompletionContentPartTextParam(
                                        type="text",
                                        text=content_item.get("text", ""),
                                    )
                                )
                            elif content_item.get("type") == "input_image":
                                image_url = content_item.get("image_url")
                                if not image_url:
                                    continue
                                detail = content_item.get("detail", "auto")
                                if detail == "original":
                                    detail = "auto"
                                content_parts.append(
                                    ChatCompletionContentPartImageParam(
                                        type="image_url",
                                        image_url={"url": image_url, "detail": detail},
                                    )
                                )
                        if content_parts:
                            chat_completion_messages.append(
                                ChatCompletionUserMessageParam(role="user", content=content_parts)
                            )
                    elif role == "system":
                        text_parts: list[ChatCompletionContentPartTextParam] = []
                        for content_item in content:
                            if content_item.get("type") == "input_text":
                                text_parts.append(
                                    ChatCompletionContentPartTextParam(
                                        type="text",
                                        text=content_item.get("text", ""),
                                    )
                                )
                        if text_parts:
                            chat_completion_messages.append(
                                ChatCompletionSystemMessageParam(role="system", content=text_parts)
                            )
                    elif role == "developer":
                        text_parts = []
                        for content_item in content:
                            if content_item.get("type") == "input_text":
                                text_parts.append(
                                    ChatCompletionContentPartTextParam(
                                        type="text",
                                        text=content_item.get("text", ""),
                                    )
                                )
                        if text_parts:
                            chat_completion_messages.append(
                                ChatCompletionDeveloperMessageParam(role="developer", content=text_parts)
                            )
                    elif role == "assistant":
                        assistant_parts = []
                        for content_item in content:
                            content_type = content_item.get("type")
                            if content_type in ("output_text", "input_text"):
                                assistant_parts.append(
                                    ChatCompletionContentPartTextParam(
                                        type="text",
                                        text=content_item.get("text", ""),
                                    )
                                )
                            elif content_type == "refusal":
                                assistant_parts.append(
                                    ChatCompletionContentPartRefusalParam(
                                        type="refusal",
                                        refusal=content_item.get("refusal", ""),
                                    )
                                )
                        if assistant_parts:
                            chat_completion_messages.append(
                                ChatCompletionAssistantMessageParam(
                                    role="assistant",
                                    content=assistant_parts,
                                )
                            )
            elif message.get("type") == "reasoning":
                continue
            elif message.get("type") == "function_call":
                call_id = message.get("call_id") or ""
                name = message.get("name") or ""
                raw_arguments = message.get("arguments")
                arguments = raw_arguments if isinstance(raw_arguments, str) else ""
                tool_call = ChatCompletionMessageFunctionToolCallParam(
                    id=call_id,
                    type="function",
                    function={
                        "name": name,
                        "arguments": arguments,
                    },
                )
                if chat_completion_messages and chat_completion_messages[-1].get("role") == "assistant":
                    last_message = cast(ChatCompletionAssistantMessageParam, chat_completion_messages[-1])
                    existing_tool_calls = last_message.get("tool_calls")
                    if existing_tool_calls is None:
                        last_message["tool_calls"] = [tool_call]
                    else:
                        last_message["tool_calls"] = [*existing_tool_calls, tool_call]
                else:
                    chat_completion_messages.append(
                        ChatCompletionAssistantMessageParam(
                            role="assistant",
                            content=None,
                            tool_calls=[tool_call],
                        )
                    )
            elif message.get("type") == "function_call_output":
                call_id = message.get("call_id") or ""
                output = message.get("output", "")
                if isinstance(output, str):
                    chat_completion_messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=call_id,
                            content=output,
                        )
                    )
                elif isinstance(output, list):
                    output_parts: list[ChatCompletionContentPartTextParam] = []
                    image_parts: list[ChatCompletionContentPartImageParam] = []
                    for output_item in output:
                        if output_item.get("type") == "input_text":
                            output_parts.append(
                                ChatCompletionContentPartTextParam(
                                    type="text",
                                    text=output_item.get("text", ""),
                                )
                            )
                        elif output_item.get("type") == "input_image":
                            image_url = output_item.get("image_url")
                            if not image_url:
                                continue
                            raw_detail = output_item.get("detail", "auto")
                            detail: Literal["auto", "low", "high"] = (
                                raw_detail if raw_detail in ("auto", "low", "high") else "auto"
                            )
                            image_parts.append(
                                ChatCompletionContentPartImageParam(
                                    type="image_url",
                                    image_url={"url": image_url, "detail": detail},
                                )
                            )
                    chat_completion_messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=call_id,
                            content=output_parts if output_parts else "",
                        )
                    )
                    if image_parts:
                        chat_completion_messages.append(
                            ChatCompletionUserMessageParam(role="user", content=image_parts)
                        )

        if instructions:
            for message in chat_completion_messages:
                if message.get("role") in ("system", "developer"):
                    instruction_message = cast(
                        ChatCompletionSystemMessageParam | ChatCompletionDeveloperMessageParam,
                        message,
                    )
                    content = instruction_message.get("content", "")
                    if isinstance(content, str):
                        instruction_message["content"] = f"{content}\n{instructions}" if content else instructions
                    else:
                        instruction_message["content"] = [
                            *content,
                            ChatCompletionContentPartTextParam(type="text", text=instructions),
                        ]
                    break
            else:
                chat_completion_messages.insert(
                    0,
                    ChatCompletionSystemMessageParam(role="system", content=instructions),
                )

        return chat_completion_messages

    @staticmethod
    def _create_config(
        *,
        max_output_tokens: int | None = None,
        parallel_tool_calls: bool | None = None,
        reasoning: Reasoning | None = None,
        temperature: float | None = None,
        tool_choice: response_create_params.ToolChoice | None = None,
        tools: Iterable[ToolParam] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Builds kwargs for ``client.chat.completions.create``.

        Ignored Responses-only params (handled by not accepting them here):
        ``include``, ``text``, ``truncation``, ``background``.
        """
        config: dict[str, Any] = {}

        if max_output_tokens is not None:
            config["max_tokens"] = max_output_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if parallel_tool_calls is not None:
            config["parallel_tool_calls"] = parallel_tool_calls

        if top_logprobs is not None:
            # Chat Completions requires logprobs=True whenever top_logprobs is set.
            config["top_logprobs"] = top_logprobs
            config["logprobs"] = True

        if reasoning is not None:
            # Chat Completions only exposes effort as reasoning_effort; summary / context / mode have no equivalents.
            effort = reasoning.get("effort")
            if effort is not None:
                config["reasoning_effort"] = effort

        chat_tool_choice = ChatCompletionsProvider._convert_tool_choice(tool_choice)
        if chat_tool_choice is not None:
            config["tool_choice"] = chat_tool_choice

        if tools is not None:
            chat_tools = ChatCompletionsProvider._convert_tools(tools)
            if chat_tools:
                config["tools"] = chat_tools

        if provider_kwargs:
            config["extra_body"] = provider_kwargs

        return config

    @staticmethod
    def _convert_tool_choice(
        tool_choice: response_create_params.ToolChoice | None,
    ) -> ChatCompletionToolChoiceOptionParam | None:
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            if tool_choice in ("none", "auto", "required"):
                return tool_choice
            return None

        if not isinstance(tool_choice, dict):
            return None

        choice_type = tool_choice.get("type")
        if choice_type == "function":
            # Responses keeps name at the top level while Chat Completions nests it.
            name = tool_choice.get("name") or ""
            return ChatCompletionNamedToolChoiceParam(
                type="function",
                function={"name": name},
            )
        if choice_type == "allowed_tools":
            # Same nesting difference: mode/tools live under allowed_tools, and each function ref is {type, function: {name}} instead of {type, name}.
            mode = tool_choice.get("mode")
            if mode not in ("auto", "required"):
                return None
            converted_tools: list[dict[str, object]] = []
            for tool in tool_choice.get("tools") or []:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    continue
                name = tool.get("name")
                if not isinstance(name, str):
                    continue
                converted_tools.append({"type": "function", "function": {"name": name}})
            return ChatCompletionAllowedToolChoiceParam(
                type="allowed_tools",
                allowed_tools={"mode": mode, "tools": converted_tools},
            )

        # Hosted / Responses-only choices (mcp, custom, shell, etc.) have no useful Chat Completions mapping.
        return None

    @staticmethod
    def _convert_tools(
        tools: Iterable[ToolParam],
    ) -> list[ChatCompletionToolUnionParam]:
        """Converts Responses API tools to chat completion tools.

        Only `type="function"` tools are mapped. Built-in / hosted tools are skipped.
        Flat Responses function fields are nested under `function`.
        Responses-only fields (`allowed_callers`, `defer_loading`, `output_schema`) are dropped.
        """
        chat_tools: list[ChatCompletionToolUnionParam] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            # Cast needed because ToolParam is a union of many TypedDicts, and ty
            # can't narrow it to FunctionToolParam even when type is "function".
            name = cast(str, tool.get("name") or "")
            function: FunctionDefinition = {"name": name}
            description = tool.get("description")
            if isinstance(description, str):
                function["description"] = description
            parameters = tool.get("parameters")
            if isinstance(parameters, dict):
                function["parameters"] = cast(dict[str, object], parameters)
            if "strict" in tool:
                function["strict"] = cast(bool | None, tool.get("strict"))
            chat_tools.append(
                ChatCompletionFunctionToolParam(
                    type="function",
                    function=function,
                )
            )
        return chat_tools

    @staticmethod
    def _convert_error(raw: object) -> ResponseError | None:
        """Converts an error payload from a 200-with-error body to a ResponseError.

        ResponseError's code is a closed literal, so raw HTTP-style codes map onto it (429 becomes rate_limit_exceeded, everything else server_error)
        and are folded into the message along with any metadata so no information is lost.
        """
        if not raw:
            return None
        if not isinstance(raw, dict):
            return ResponseError(code="server_error", message=str(raw))
        raw_code = raw.get("code")
        raw_message = raw.get("message")
        message = raw_message if isinstance(raw_message, str) and raw_message else "Unknown provider error"
        if raw_code is not None:
            message = f"[{raw_code}] {message}"
        metadata = raw.get("metadata")
        if isinstance(metadata, dict) and metadata:
            message = f"{message} (metadata: {json.dumps(metadata, default=str)})"
        code: Literal["server_error", "rate_limit_exceeded"] = (
            "rate_limit_exceeded" if str(raw_code) == "429" else "server_error"
        )
        return ResponseError(code=code, message=message)

    @staticmethod
    def _convert_response(completion: ChatCompletion) -> RouterResponse:
        output: list[ChatMessage] = []
        incomplete_details: IncompleteDetails | None = None
        item_status: Literal["completed", "incomplete"] = "completed"

        error = ChatCompletionsProvider._convert_error(getattr(completion, "error", None))

        if not completion.choices:
            usage = ChatCompletionsProvider._convert_usage(completion.usage) if completion.usage else None
            return RouterResponse(output=output, error=error, usage=usage)

        # We don't have an n parameter, so we always take the first choice.
        choice = completion.choices[0]
        message = choice.message

        if choice.finish_reason == "length":
            item_status = "incomplete"
            incomplete_details = IncompleteDetails(reason="max_output_tokens")
        elif choice.finish_reason == "content_filter":
            item_status = "incomplete"
            incomplete_details = IncompleteDetails(reason="content_filter")

        reasoning_text = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        if isinstance(reasoning_text, str) and reasoning_text:
            reasoning_item: ResponseReasoningItemParam = {
                "id": str(uuid.uuid4()),
                "type": "reasoning",
                "summary": [Summary(text=reasoning_text, type="summary_text")],
                "status": item_status,
            }
            output.append(ChatMessage(message=reasoning_item, created_by=ChatCompletionsProvider.PROVIDER_NAME))

        content_parts: list[ResponseOutputTextParam | ResponseOutputRefusalParam] = []
        if isinstance(message.content, str) and message.content:
            text_part: ResponseOutputTextParam = {
                "type": "output_text",
                "text": message.content,
                "annotations": [],
            }
            if choice.logprobs is not None and choice.logprobs.content is not None:
                text_part["logprobs"] = [
                    Logprob(
                        token=token_logprob.token,
                        bytes=token_logprob.bytes or [],
                        logprob=token_logprob.logprob,
                        top_logprobs=[
                            LogprobTopLogprob(
                                token=top.token,
                                bytes=top.bytes or [],
                                logprob=top.logprob,
                            )
                            for top in token_logprob.top_logprobs
                        ],
                    )
                    for token_logprob in choice.logprobs.content
                ]
            content_parts.append(text_part)
        if isinstance(message.refusal, str) and message.refusal:
            content_parts.append(ResponseOutputRefusalParam(type="refusal", refusal=message.refusal))

        if content_parts:
            message_param = ResponseOutputMessageParam(
                id=str(uuid.uuid4()),
                type="message",
                role="assistant",
                status=item_status,
                content=content_parts,
            )
            output.append(ChatMessage(message=message_param, created_by=ChatCompletionsProvider.PROVIDER_NAME))

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type != "function":
                    continue
                function_call = ResponseFunctionToolCallParam(
                    type="function_call",
                    call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    id=str(uuid.uuid4()),
                    status=item_status,
                )
                output.append(ChatMessage(message=function_call, created_by=ChatCompletionsProvider.PROVIDER_NAME))

        usage = ChatCompletionsProvider._convert_usage(completion.usage) if completion.usage else None
        if output:
            output[-1].original_response = completion.model_dump(mode="json")
        return RouterResponse(
            output=output,
            error=error,
            incomplete_details=incomplete_details,
            usage=usage,
        )

    @staticmethod
    def _convert_usage(usage: CompletionUsage) -> ResponseUsage:
        cached_tokens = 0
        cache_write_tokens = 0
        if usage.prompt_tokens_details is not None:
            cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
            cache_write_tokens = usage.prompt_tokens_details.cache_write_tokens or 0

        reasoning_tokens = 0
        if usage.completion_tokens_details is not None:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0

        return ResponseUsage(
            input_tokens=usage.prompt_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
            output_tokens=usage.completion_tokens,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
            total_tokens=usage.total_tokens,
        )

    @staticmethod
    def _convert_stream_chunk(chunk: ChatCompletionChunk, sequence_number: int) -> list[ResponseStreamEvent]:
        """Maps Chat Completions stream deltas to Responses-style stream events.

        - reasoning / reasoning_content -> ResponseReasoningSummaryTextDeltaEvent
        - content -> ResponseTextDeltaEvent
        - tool_calls[].function.arguments -> ResponseFunctionCallArgumentsDeltaEvent
        """
        events: list[ResponseStreamEvent] = []
        if not chunk.choices:
            return events

        delta = chunk.choices[0].delta
        reasoning_delta = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if isinstance(reasoning_delta, str) and reasoning_delta:
            events.append(
                ResponseReasoningSummaryTextDeltaEvent(
                    delta=reasoning_delta,
                    item_id="",
                    output_index=0,
                    sequence_number=sequence_number,
                    summary_index=0,
                    type="response.reasoning_summary_text.delta",
                )
            )
            sequence_number += 1

        if delta.content:
            events.append(
                ResponseTextDeltaEvent(
                    content_index=0,
                    delta=delta.content,
                    item_id="",
                    logprobs=[],
                    output_index=0,
                    sequence_number=sequence_number,
                    type="response.output_text.delta",
                )
            )
            sequence_number += 1

        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                arguments_delta = tool_call_delta.function.arguments if tool_call_delta.function else None
                if not arguments_delta:
                    continue
                events.append(
                    ResponseFunctionCallArgumentsDeltaEvent(
                        delta=arguments_delta,
                        item_id=tool_call_delta.id or "",
                        output_index=tool_call_delta.index,
                        sequence_number=sequence_number,
                        type="response.function_call_arguments.delta",
                    )
                )
                sequence_number += 1

        return events

    @staticmethod
    def _build_completion_from_stream(
        *,
        completion_id: str,
        model: str,
        content: str,
        refusal: str,
        reasoning: str,
        tool_calls: list[dict[str, Any]],
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"],
        usage: CompletionUsage | None,
    ) -> ChatCompletion:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
            "refusal": refusal or None,
        }
        if reasoning:
            message["reasoning"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls

        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": message,
                    "logprobs": None,
                }
            ],
            "usage": usage.model_dump(mode="json") if usage is not None else None,
        }
        return ChatCompletion.model_validate(payload)
