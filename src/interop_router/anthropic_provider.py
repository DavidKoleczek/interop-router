from collections.abc import Iterable
import json
from typing import Any, Literal
import uuid

import anthropic
from anthropic import NOT_GIVEN, AsyncAnthropic
from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    Message,
    MessageParam,
    RedactedThinkingBlock,
    RedactedThinkingBlockParam,
    ServerToolUseBlock,
    ServerToolUseBlockParam,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ThinkingBlockParam,
    ThinkingConfigDisabledParam,
    ThinkingConfigEnabledParam,
    ToolChoiceAnyParam,
    ToolChoiceAutoParam,
    ToolChoiceNoneParam,
    ToolChoiceParam,
    ToolChoiceToolParam,
    ToolResultBlockParam,
    ToolUnionParam,
    ToolUseBlock,
    ToolUseBlockParam,
    URLImageSourceParam,
    Usage,
    WebSearchTool20250305Param,
    WebSearchToolResultBlock,
    WebSearchToolResultBlockParam,
    WebSearchToolResultError,
)
from anthropic.types import ToolParam as AnthropicToolParam
from anthropic.types.beta import BetaWebFetchTool20250910Param, BetaWebFetchToolResultBlockParam
from anthropic.types.web_search_tool_20250305_param import UserLocation as AnthropicUserLocation
from openai.types.responses import (
    ResponseFunctionToolCallParam,
    ResponseIncludable,
    ResponseOutputTextParam,
    ResponseTextConfigParam,
    ResponseUsage,
    ToolParam,
    response_create_params,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_output_message_param import ResponseOutputMessageParam
from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam, Summary
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from openai.types.shared_params.reasoning import Reasoning

from interop_router.types import (
    ChatMessage,
    ContextLimitExceededError,
    ProviderName,
    RouterResponse,
    SupportedModelAnthropic,
)


class AnthropicProvider:
    PROVIDER_NAME = "anthropic"
    REDACTED_THINKING_BLOCK_ID = "REDACTEDTHINKINGBLOCK"

    @staticmethod
    async def create(
        *,
        client: AsyncAnthropic,
        input: list[ChatMessage],
        model: SupportedModelAnthropic,
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
        preprocessed_input, system_instruction = AnthropicProvider._preprocess_input(input)
        anthropic_messages = AnthropicProvider._convert_input_messages(preprocessed_input)
        config, extra_headers = AnthropicProvider._create_config(
            max_output_tokens, temperature, reasoning, tool_choice, tools, system_instruction
        )
        try:
            async with client.messages.stream(
                messages=anthropic_messages,
                model=model,
                extra_headers=extra_headers if extra_headers else None,
                **config,
            ) as stream:
                async for _text in stream.text_stream:
                    pass
            response = await stream.get_final_message()
        except anthropic.BadRequestError as e:
            if "prompt is too long" in str(e):
                raise ContextLimitExceededError(str(e), provider="anthropic", cause=e) from e
            raise

        interop_response = AnthropicProvider._convert_response(response)
        return interop_response

    @staticmethod
    def _preprocess_input(input: list[ChatMessage]) -> tuple[list[ChatMessage], str]:
        preprocessed_messages = []
        instructions = ""
        for message in input:
            input_message = message.message
            if input_message.get("role", None) in ["system", "developer"]:
                content = input_message.get("content", [])
                if isinstance(content, str):
                    instructions += content + "\n"
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            instructions += item + "\n"
            elif (
                input_message.get("type", None) == "reasoning" and message.created_by != AnthropicProvider.PROVIDER_NAME
            ):
                continue
            else:
                preprocessed_messages.append(message)
        instructions = instructions.strip()
        return preprocessed_messages, instructions

    @staticmethod
    def _convert_input_messages(input: list[ChatMessage]) -> list[MessageParam]:
        message_params: list[MessageParam] = []
        for message in input:
            input_message = message.message

            if input_message.get("role") is not None:
                role = input_message.get("role")
                content = input_message.get("content", [])
                if isinstance(content, str):
                    if role == "user":
                        message_params.append(MessageParam(role="user", content=content))
                    elif role == "assistant":
                        message_params.append(MessageParam(role="assistant", content=content))
                elif isinstance(content, list):
                    content_items = []
                    for content_item in content:
                        if content_item.get("type") == "input_text":
                            text = content_item.get("text", "")
                            content_items.append(TextBlockParam(type="text", text=text))
                        elif content_item.get("type") == "input_image":
                            image_block = AnthropicProvider._convert_input_image(content_item)
                            if image_block:
                                content_items.append(image_block)
                        elif role == "assistant" and content_item.get("type") == "output_text":
                            text = content_item.get("text", "")
                            content_items.append(TextBlockParam(type="text", text=text))
                    if role == "user":
                        message_params.append(MessageParam(role="user", content=content_items))
                    elif role == "assistant":
                        message_params.append(MessageParam(role="assistant", content=content_items))

            elif input_message.get("type") == "reasoning":
                summaries = input_message.get("summary", [])
                if summaries:
                    summary = summaries[0]  # Claude should only have one summary
                    encrypted_content = input_message.get("encrypted_content", "") or ""
                    if summary == AnthropicProvider.REDACTED_THINKING_BLOCK_ID:
                        message_param = MessageParam(
                            role="assistant",
                            content=[
                                RedactedThinkingBlockParam(
                                    data=encrypted_content,
                                    type="redacted_thinking",
                                )
                            ],
                        )
                        message_params.append(message_param)
                    else:
                        thinking_text = summary.get("text", "")
                        message_param = MessageParam(
                            role="assistant",
                            content=[
                                ThinkingBlockParam(
                                    signature=encrypted_content,
                                    thinking=thinking_text,
                                    type="thinking",
                                )
                            ],
                        )
                        message_params.append(message_param)

            elif input_message.get("type") == "function_call":
                anthropic_interop = message.interop.get(AnthropicProvider.PROVIDER_NAME, {})
                call_id = input_message.get("call_id") or ""
                arguments_str = input_message.get("arguments") or "{}"
                arguments = json.loads(arguments_str)
                name = input_message.get("name") or ""

                if anthropic_interop.get("type") == "server_tool_use":
                    block: ServerToolUseBlockParam = ServerToolUseBlockParam(
                        id=call_id,
                        input=arguments,
                        # NOTE: Ignore this type because ServerToolUseBlockParam should allow for name = 'web_fetch' but it does not currently
                        name=name,  # type: ignore
                        type="server_tool_use",
                    )
                else:
                    block = ToolUseBlockParam(
                        id=call_id,
                        input=arguments,
                        name=name,
                        type="tool_use",
                    )
                message_params.append(MessageParam(role="assistant", content=[block]))

            elif input_message.get("type") == "function_call_output":
                anthropic_interop = message.interop.get(AnthropicProvider.PROVIDER_NAME, {})
                call_id = input_message.get("call_id") or ""

                if anthropic_interop.get("type") == "web_search_tool_result":
                    stored_block = anthropic_interop.get("web_search_tool_result_block_param", {})
                    result_block: WebSearchToolResultBlockParam = WebSearchToolResultBlockParam(
                        content=stored_block.get("content"),
                        tool_use_id=call_id,
                        type="web_search_tool_result",
                    )
                    message_params.append(MessageParam(role="assistant", content=[result_block]))
                elif anthropic_interop.get("type") == "web_fetch_tool_result":
                    stored_block = anthropic_interop.get("web_fetch_tool_result", {})
                    fetch_result_block = BetaWebFetchToolResultBlockParam(
                        content=stored_block.get("content"),
                        tool_use_id=call_id,
                        type="web_fetch_tool_result",
                    )
                    # NOTE: The BetaWebFetchToolResultBlockParam is not yet in the main MessageParam union
                    message_params.append(MessageParam(role="assistant", content=[fetch_result_block]))  # type: ignore
                else:
                    output = str(input_message.get("output") or "")
                    tool_result_block = ToolResultBlockParam(
                        tool_use_id=call_id,
                        content=output,
                        type="tool_result",
                    )
                    message_params.append(MessageParam(role="user", content=[tool_result_block]))

        return message_params

    @staticmethod
    def _convert_input_image(content_item: Any) -> ImageBlockParam | None:
        image_url = content_item.get("image_url", "") or ""
        if not image_url:
            return None

        if image_url.startswith("data:"):
            # Parse data URL format: "data:<media_type>;base64,<base64_data>"
            header, base64_data = image_url.split(",", 1)
            media_type = header.split(":")[1].split(";")[0]
            source = Base64ImageSourceParam(type="base64", media_type=media_type, data=base64_data)
            return ImageBlockParam(type="image", source=source)
        else:
            # Regular URL
            source = URLImageSourceParam(type="url", url=image_url)
            return ImageBlockParam(type="image", source=source)

    @staticmethod
    def _create_config(
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning: Reasoning | None = None,
        tool_choice: response_create_params.ToolChoice | None = None,
        tools: Iterable[ToolParam] | None = None,
        system_instruction: str = "",
    ) -> tuple[dict[str, Any], dict[str, str] | None]:
        max_output_tokens = max_output_tokens if max_output_tokens is not None else 64000

        thinking_config_param = ThinkingConfigDisabledParam(type="disabled")
        thinking_budget_tokens = 0
        if reasoning is not None:
            reasoning_effort = reasoning.get("effort", "none")
            thinking_budget_tokens = 1024
            match reasoning_effort:
                case "none":
                    thinking_budget_tokens = 1024  # The minimum if thinking is enabled is 1024
                case "low":
                    thinking_budget_tokens = 2_000
                case "medium":
                    thinking_budget_tokens = 8_000
                case "high":
                    thinking_budget_tokens = 16_000
                case "xhigh":
                    # 64k is the maximum, but this does not leave room for output tokens
                    thinking_budget_tokens = 32_000

            thinking_config_param = ThinkingConfigEnabledParam(budget_tokens=thinking_budget_tokens, type="enabled")

        anthropic_tools, has_web_fetch = AnthropicProvider._convert_tools(tools) if tools else ([], False)
        anthropic_tool_choice = AnthropicProvider._convert_tool_choice(tool_choice)

        extra_headers: dict[str, str] | None = None
        if has_web_fetch:
            extra_headers = {"anthropic-beta": "web-fetch-2025-09-10,interleaved-thinking-2025-05-14"}

        config = {
            "max_tokens": max_output_tokens,
            "temperature": temperature if temperature is not None else NOT_GIVEN,
            "system": system_instruction if system_instruction else NOT_GIVEN,
            "thinking": thinking_config_param,
            "tools": anthropic_tools,
            "tool_choice": anthropic_tool_choice if anthropic_tool_choice is not None else NOT_GIVEN,
            "timeout": 3600,
        }
        return config, extra_headers

    @staticmethod
    def _convert_tools(
        tools: Iterable[ToolParam],
    ) -> tuple[list[ToolUnionParam | BetaWebFetchTool20250910Param], bool]:
        anthropic_tools: list[ToolUnionParam | BetaWebFetchTool20250910Param] = []
        web_search_tool: ToolParam | None = None
        has_web_fetch = False

        for tool in tools:
            tool_type = tool.get("type")
            if tool_type == "function":
                name = tool.get("name") or ""
                description = tool.get("description") or ""
                parameters = tool.get("parameters") or {"type": "object", "properties": {}}
                anthropic_tools.append(
                    AnthropicToolParam(
                        name=name,
                        description=description,
                        input_schema=parameters,
                    )
                )
            elif tool_type in ("web_search", "web_search_2025_08_26"):
                web_search_tool = tool

        if web_search_tool:
            web_search_param: WebSearchTool20250305Param = {
                "type": "web_search_20250305",
                "name": "web_search",
            }
            filters = web_search_tool.get("filters")
            if filters and isinstance(filters, dict):
                allowed_domains = filters.get("allowed_domains")
                if allowed_domains:
                    web_search_param["allowed_domains"] = allowed_domains
            user_location = web_search_tool.get("user_location")
            if user_location and isinstance(user_location, dict):
                anthropic_user_location: AnthropicUserLocation = {"type": "approximate"}
                if user_location.get("city"):
                    anthropic_user_location["city"] = user_location.get("city")
                if user_location.get("country"):
                    anthropic_user_location["country"] = user_location.get("country")
                if user_location.get("region"):
                    anthropic_user_location["region"] = user_location.get("region")
                if user_location.get("timezone"):
                    anthropic_user_location["timezone"] = user_location.get("timezone")
                web_search_param["user_location"] = anthropic_user_location
            anthropic_tools.append(web_search_param)

            web_fetch_param: BetaWebFetchTool20250910Param = {
                "type": "web_fetch_20250910",
                "name": "web_fetch",
                "citations": {"enabled": False},
            }
            if filters and isinstance(filters, dict):
                allowed_domains = filters.get("allowed_domains")
                if allowed_domains:
                    web_fetch_param["allowed_domains"] = allowed_domains
            anthropic_tools.append(web_fetch_param)
            has_web_fetch = True

        return anthropic_tools, has_web_fetch

    @staticmethod
    def _convert_tool_choice(
        tool_choice: response_create_params.ToolChoice | None,
    ) -> ToolChoiceParam | None:
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            match tool_choice:
                case "auto":
                    return ToolChoiceAutoParam(type="auto")
                case "none":
                    return ToolChoiceNoneParam(type="none")
                case "required":
                    return ToolChoiceAnyParam(type="any")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function_name = tool_choice.get("name") or ""
            return ToolChoiceToolParam(type="tool", name=function_name)

        return None

    @staticmethod
    def _convert_response(response: Message) -> RouterResponse:
        output: list[ChatMessage] = []
        usage = AnthropicProvider._convert_usage(response.usage)
        for content_block in response.content:
            if isinstance(content_block, TextBlock):
                if content_block.type == "web_fetch_tool_result":
                    # Currently the web fetch tool result is returned as a text block and doesn't have a dedicated type
                    block_content = content_block.content
                    interop_data: dict[ProviderName, Any] = {
                        AnthropicProvider.PROVIDER_NAME: {
                            "type": "web_fetch_tool_result",
                            "web_fetch_tool_result": content_block.model_dump(),
                        }
                    }
                    generic_output = json.dumps(block_content)
                    function_call_output = FunctionCallOutput(
                        call_id=content_block.tool_use_id,
                        output=generic_output,
                        type="function_call_output",
                    )
                    output.append(
                        ChatMessage(
                            message=function_call_output,
                            created_by=AnthropicProvider.PROVIDER_NAME,
                            interop=interop_data,
                        )
                    )
                else:
                    content_item = ResponseOutputTextParam(text=content_block.text, type="output_text", annotations=[])
                    message_param = ResponseOutputMessageParam(
                        id=str(uuid.uuid4()),
                        content=[content_item],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                    output.append(ChatMessage(message=message_param, created_by=AnthropicProvider.PROVIDER_NAME))
            elif isinstance(content_block, ThinkingBlock):
                reasoning_item: ResponseReasoningItemParam = {
                    "id": str(uuid.uuid4()),
                    "type": "reasoning",
                    "summary": [Summary(text=content_block.thinking, type="summary_text")],
                    "status": "completed",
                    "encrypted_content": content_block.signature,
                }
                output.append(ChatMessage(message=reasoning_item, created_by=AnthropicProvider.PROVIDER_NAME))
            elif isinstance(content_block, RedactedThinkingBlock):
                reasoning_item: ResponseReasoningItemParam = {
                    "id": str(uuid.uuid4()),
                    "type": "reasoning",
                    "summary": [Summary(text=AnthropicProvider.REDACTED_THINKING_BLOCK_ID, type="summary_text")],
                    "status": "completed",
                    "encrypted_content": content_block.data,
                }
                output.append(ChatMessage(message=reasoning_item, created_by=AnthropicProvider.PROVIDER_NAME))
            elif isinstance(content_block, ToolUseBlock):
                tool_call = ResponseFunctionToolCallParam(
                    call_id=content_block.id,
                    arguments=json.dumps(content_block.input),
                    name=content_block.name,
                    type="function_call",
                    id=str(uuid.uuid4()),
                    status="completed",
                )
                output.append(ChatMessage(message=tool_call, created_by=AnthropicProvider.PROVIDER_NAME))
            elif isinstance(content_block, ServerToolUseBlock):
                tool_call = ResponseFunctionToolCallParam(
                    call_id=content_block.id,
                    arguments=json.dumps(content_block.input),
                    name=content_block.name,
                    type="function_call",
                    id=str(uuid.uuid4()),
                    status="completed",
                )
                interop_data: dict[ProviderName, Any] = {
                    AnthropicProvider.PROVIDER_NAME: {
                        "type": "server_tool_use",
                    }
                }
                output.append(
                    ChatMessage(message=tool_call, created_by=AnthropicProvider.PROVIDER_NAME, interop=interop_data)
                )
            elif isinstance(content_block, WebSearchToolResultBlock):
                generic_output = AnthropicProvider._convert_web_search_tool_result_block(content_block)
                function_call_output = FunctionCallOutput(
                    call_id=content_block.tool_use_id,
                    output=generic_output,
                    type="function_call_output",
                )
                interop_data: dict[ProviderName, Any] = {
                    AnthropicProvider.PROVIDER_NAME: {
                        "type": "web_search_tool_result",
                        "web_search_tool_result_block_param": content_block.model_dump(),
                    }
                }
                output.append(
                    ChatMessage(
                        message=function_call_output,
                        created_by=AnthropicProvider.PROVIDER_NAME,
                        interop=interop_data,
                    )
                )

        if output:
            output[-1].original_response = response.model_dump(mode="json")
        return RouterResponse(output=output, usage=usage)

    @staticmethod
    def _convert_web_search_tool_result_block(
        block: WebSearchToolResultBlock,
    ) -> str:
        if isinstance(block.content, WebSearchToolResultError):
            return ""

        results = [
            {
                "title": result.title,
                "url": result.url,
                "page_age": result.page_age,
            }
            for result in block.content
        ]
        return json.dumps(results)

    @staticmethod
    def _convert_usage(usage: Usage) -> ResponseUsage:
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        total_tokens = input_tokens + output_tokens
        cached_tokens = usage.cache_read_input_tokens or 0

        input_tokens_details = InputTokensDetails(cached_tokens=cached_tokens)
        output_tokens_details = OutputTokensDetails(reasoning_tokens=0)

        return ResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=input_tokens_details,
            output_tokens=output_tokens,
            output_tokens_details=output_tokens_details,
            total_tokens=total_tokens,
        )
