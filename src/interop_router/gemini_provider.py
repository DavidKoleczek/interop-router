import base64
from collections.abc import Iterable
import json
from typing import Any, Literal
import uuid

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from google.genai.types import Content, GenerateContentResponse, GenerateContentResponseUsageMetadata
from openai.types.responses import (
    ResponseFunctionToolCallParam,
    ResponseIncludable,
    ResponseInputImageContentParam,
    ResponseInputItemParam,
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

from interop_router.types import ChatMessage, ContextLimitExceededError, RouterResponse, SupportedModelGemini


class GeminiProvider:
    PROVIDER_NAME = "gemini"
    DUMMY_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"

    @staticmethod
    async def create(
        *,
        client: genai.Client,
        input: list[ChatMessage],
        model: SupportedModelGemini,
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
        preprocessed_input, system_instruction = GeminiProvider._preprocess_input(input)
        gemini_messages = GeminiProvider._convert_input_messages(preprocessed_input)

        # We can have kwargs specific to Gemini in the last message's provider_kwargs
        gemini_kwargs = input[-1].provider_kwargs.get("gemini", {}) if input else {}
        gemini_config, effective_model = GeminiProvider._create_config(
            model=model,
            system_instruction=system_instruction,
            include=include,
            max_output_tokens=max_output_tokens,
            reasoning=reasoning,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            gemini_kwargs=gemini_kwargs,
        )

        try:
            response = await client.aio.models.generate_content(
                model=effective_model,
                contents=gemini_messages,
                config=gemini_config,
            )
        except genai_errors.ClientError as e:
            if "input_token" in str(e).lower():
                raise ContextLimitExceededError(str(e), provider="gemini", cause=e) from e
            raise

        interop_response = GeminiProvider._convert_response(response)
        return interop_response

    @staticmethod
    def _preprocess_input(input: list[ChatMessage]) -> tuple[list[ChatMessage], str]:
        """Removes non-Gemini reasoning messages and
        concats system/developer messages to instructions (and removes them from the input)
        """
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
            elif input_message.get("type", None) == "reasoning" and message.created_by != GeminiProvider.PROVIDER_NAME:
                continue
            else:
                preprocessed_messages.append(message)
        instructions = instructions.strip()
        return preprocessed_messages, instructions

    # region: Input conversion

    @staticmethod
    def _convert_input_messages(input: list[ChatMessage]) -> list[Content]:
        gemini_content: list[Content] = []
        previous_was_function_call = False
        # ids to skip because they were processed already (for image_generation function calls)
        skip_messages: set[str] = set()
        for i in range(len(input)):
            if input[i].id in skip_messages:
                continue
            input_message = input[i].message

            if input_message.get("role") is not None:
                previous_was_function_call = False
                role = input_message.get("role", None)
                content = input_message.get("content", [])
                if isinstance(content, str) and role == "user":
                    gemini_content.append(
                        types.Content(
                            parts=[types.Part(text=content)],
                            role=role,
                        )
                    )
                elif isinstance(content, str) and role == "assistant":
                    thought_signature = GeminiProvider.DUMMY_THOUGHT_SIGNATURE
                    gemini_role = "model" if role == "assistant" else role
                    if i > 0:
                        thought_signature = GeminiProvider._get_thought_signature(input[i - 1].message)
                    gemini_content.append(
                        types.Content(
                            parts=[types.Part(text=content, thought_signature=thought_signature)],
                            role=gemini_role,
                        )
                    )
                elif isinstance(content, list):
                    for content_item in content:
                        if role == "user":
                            if content_item.get("type") == "input_text":
                                text = content_item.get("text", "")
                                gemini_content.append(
                                    types.Content(
                                        parts=[types.Part(text=text)],
                                        role=role,
                                    )
                                )
                            elif content_item.get("type") == "input_image":
                                image_url = content_item.get("image_url")
                                if image_url and image_url.startswith("data:"):
                                    header, base64_data = image_url.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    image_bytes = base64.b64decode(base64_data)
                                    gemini_content.append(
                                        types.Content(
                                            parts=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                                            role=role,
                                        )
                                    )
                        elif role == "assistant" and content_item.get("type") == "output_text":
                            text = content_item.get("text", "")
                            thought_signature = GeminiProvider.DUMMY_THOUGHT_SIGNATURE
                            if i > 0:
                                thought_signature = GeminiProvider._get_thought_signature(input[i - 1].message)
                            gemini_content.append(
                                types.Content(
                                    parts=[types.Part(text=text, thought_signature=thought_signature)],
                                    role="model",
                                )
                            )
            elif input_message.get("type") == "function_call":
                if input_message.get("name") == "image_generation":
                    call_id = input_message.get("call_id", "")
                    function_call_output = GeminiProvider._get_corresponding_function_call_output(call_id, input)
                    if function_call_output:
                        skip_messages.add(function_call_output.id)

                        thought_signature = GeminiProvider.DUMMY_THOUGHT_SIGNATURE
                        if i > 0:
                            thought_signature = GeminiProvider._get_thought_signature(input[i - 1].message)

                        # Get the image content from message.output (which is a list)'s image_urls
                        function_call_output_message = function_call_output.message
                        output_content = function_call_output_message.get("output", [])
                        parts: list[types.Part] = []
                        if isinstance(output_content, list):
                            for idx, item in enumerate(output_content):
                                if isinstance(item, dict) and item.get("type") == "input_image":
                                    # Convert each image_url to bytes that Gemini expects
                                    image_url = item.get("image_url", "")
                                    if image_url.startswith("data:"):
                                        header, base64_data = image_url.split(",", 1)
                                        mime_type = header.split(":")[1].split(";")[0]
                                        image_bytes = base64.b64decode(base64_data)
                                        # Only first part gets thought_signature
                                        part_thought_sig = thought_signature if idx == 0 else None
                                        # Create parts with inline data for each image
                                        parts.append(
                                            types.Part(
                                                inline_data=types.Blob(data=image_bytes, mime_type=mime_type),
                                                thought_signature=part_thought_sig,
                                            )
                                        )
                        if parts:
                            gemini_content.append(types.Content(parts=parts, role="model"))
                else:
                    thought_signature = GeminiProvider.DUMMY_THOUGHT_SIGNATURE
                    if not previous_was_function_call and i > 0:
                        thought_signature = GeminiProvider._get_thought_signature(input[i - 1].message)

                    fc_name = input_message.get("name", "")
                    try:
                        fc_arguments = json.loads(input_message.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        fc_arguments = {}
                    function_call = types.FunctionCall(name=fc_name, args=fc_arguments)

                    if not previous_was_function_call:
                        parts = [types.Part(function_call=function_call, thought_signature=thought_signature)]
                    else:
                        parts = [types.Part(function_call=function_call)]

                    gemini_content.append(types.Content(parts=parts, role="model"))
                previous_was_function_call = True
            elif input_message.get("type") == "function_call_output":
                previous_was_function_call = False
                fc_name = GeminiProvider._get_function_call_name_from_id(input, input_message.get("call_id", ""))

                # Gemini expects the function output to be a dict.
                raw_output = input_message.get("output", "{}")
                try:
                    if isinstance(raw_output, str):
                        parsed = json.loads(raw_output)
                        response_dict = parsed if isinstance(parsed, dict) else {"output": parsed}
                    else:
                        response_dict = {"output": raw_output}
                except json.JSONDecodeError:
                    response_dict = {"output": raw_output}

                gemini_content.append(
                    types.Content(
                        parts=[
                            types.Part.from_function_response(
                                name=fc_name,
                                response=response_dict,
                            )
                        ]
                    )
                )

        return gemini_content

    @staticmethod
    def _get_thought_signature(input_message: ResponseInputItemParam) -> bytes:
        """Extract thought_signature from a reasoning message, if it exists."""
        if input_message.get("type") != "reasoning":
            return GeminiProvider.DUMMY_THOUGHT_SIGNATURE

        encrypted_content = input_message.get("encrypted_content")
        thought_signature: bytes = GeminiProvider.DUMMY_THOUGHT_SIGNATURE
        if encrypted_content and isinstance(encrypted_content, str):
            thought_signature = base64.b64decode(encrypted_content)

        return thought_signature

    @staticmethod
    def _get_corresponding_function_call_output(call_id: str, input: list[ChatMessage]) -> ChatMessage | None:
        for message in input:
            input_message = message.message
            if input_message.get("type") == "function_call_output" and input_message.get("call_id") == call_id:
                return message
        return None

    @staticmethod
    def _get_function_call_name_from_id(input: list[ChatMessage], call_id: str) -> str:
        for message in input:
            input_message = message.message
            if input_message.get("type") == "function_call" and input_message.get("call_id") == call_id:
                return input_message.get("name", "")
        return ""

    # endregion

    # region: Gemini config conversion

    @staticmethod
    def _create_config(
        model: SupportedModelGemini,
        system_instruction: str,
        include: list[ResponseIncludable] | None = None,
        max_output_tokens: int | None = None,
        reasoning: Reasoning | None = None,
        temperature: float | None = None,
        tools: Iterable[ToolParam] | None = None,
        tool_choice: response_create_params.ToolChoice | None = None,
        gemini_kwargs: dict[str, Any] | None = None,
    ) -> tuple[types.GenerateContentConfig, str]:
        """Returns the Gemini Config and the model to use based on if image generation is configured."""

        image_config = gemini_kwargs.get("image_config") if gemini_kwargs else None
        gemini_image_config = types.ImageConfig(**image_config) if image_config else None
        image_gen_tool = next(
            (tool for tool in (tools or []) if tool.get("type") == "image_generation"),
            None,
        )
        effective_model = image_gen_tool.get("model") if image_gen_tool else model

        gemini_tools: types.ToolListUnion | None = None
        tool_config: types.ToolConfig | None = None
        if tools:
            gemini_tools = GeminiProvider._convert_tools(tools)
        if tool_choice:
            tool_config = GeminiProvider._convert_tool_choice(tool_choice)

        thinking_config: types.ThinkingConfig | None = None
        include_thoughts: bool | None = None
        thinking_level: types.ThinkingLevel | None = None
        if include and "reasoning.encrypted_content" in include:
            include_thoughts = True

        # Image gen models do not support thinking levels.
        if reasoning and not image_gen_tool:
            match reasoning.get("effort", None):
                case "minimal":
                    thinking_level = types.ThinkingLevel.LOW
                case "low":
                    thinking_level = types.ThinkingLevel.LOW
                case "medium":
                    thinking_level = types.ThinkingLevel.HIGH
                case "high":
                    thinking_level = types.ThinkingLevel.HIGH
                case "xhigh":
                    thinking_level = types.ThinkingLevel.HIGH
                case _:
                    thinking_level = None

        if include_thoughts is not None or thinking_level is not None:
            thinking_config = types.ThinkingConfig(
                include_thoughts=include_thoughts,
                thinking_level=thinking_level,
            )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            thinking_config=thinking_config,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            tools=gemini_tools,
            tool_config=tool_config,
            image_config=gemini_image_config,
            response_modalities=["TEXT", "IMAGE"] if image_gen_tool else None,
            http_options=types.HttpOptions(timeout=3_600_000),
        )
        return config, effective_model

    @staticmethod
    def _convert_tools(tools: Iterable[ToolParam]) -> types.ToolListUnion:
        function_declarations: list[types.FunctionDeclaration] = []
        has_web_search = False
        has_image_generation = False

        for tool in tools:
            tool_type = tool.get("type")
            if tool_type == "image_generation":
                has_image_generation = True
                continue
            if tool_type == "function":
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=tool.get("name"),
                        description=tool.get("description"),
                        parameters_json_schema=tool.get("parameters"),
                    )
                )
            elif tool_type in ("web_search", "web_search_2025_08_26"):
                has_web_search = True

        result: types.ToolListUnion = []
        if function_declarations:
            result.append(types.Tool(function_declarations=function_declarations))
        if has_web_search:
            result.append(types.Tool(google_search=types.GoogleSearch()))
            # Nano banana image gen does not support the UrlContext tool
            if not has_image_generation:
                result.append(types.Tool(url_context=types.UrlContext()))

        return result

    @staticmethod
    def _convert_tool_choice(tool_choice: response_create_params.ToolChoice | None) -> types.ToolConfig | None:
        """Convert OpenAI tool_choice to Gemini ToolConfig."""
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            mode_mapping: dict[str, types.FunctionCallingConfigMode] = {
                "auto": types.FunctionCallingConfigMode.AUTO,
                "none": types.FunctionCallingConfigMode.NONE,
                "required": types.FunctionCallingConfigMode.ANY,
            }
            mode = mode_mapping.get(tool_choice)
            if mode:
                return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function_name = tool_choice.get("name")
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[function_name] if function_name else None,
                )
            )

        return None

    # endregion

    # region: Output conversion

    @staticmethod
    def _convert_response(response: GenerateContentResponse) -> RouterResponse:
        output: list[ChatMessage] = []

        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    # Thought summaries will be in a part before where the thought signature is.
                    # It needs to be associated with the ResponseReasoningItemParam. There should only be one per candidate.
                    thought_summary: list[Summary] = []
                    for part in candidate.content.parts:
                        if part.thought and part.text:
                            thought_summary.append(Summary(text=part.text, type="summary_text"))
                            continue

                        if part.thought_signature:
                            encrypted_content = base64.b64encode(part.thought_signature).decode("utf-8")
                            reasoning_item: ResponseReasoningItemParam = {
                                "id": str(uuid.uuid4()),
                                "type": "reasoning",
                                "summary": thought_summary,
                                "status": "completed",
                                "encrypted_content": encrypted_content,
                            }
                            output.append(ChatMessage(message=reasoning_item, created_by=GeminiProvider.PROVIDER_NAME))

                        if part.text and not part.thought:
                            content_item = ResponseOutputTextParam(text=part.text, type="output_text", annotations=[])
                            message_param = ResponseOutputMessageParam(
                                id=str(uuid.uuid4()),
                                content=[content_item],
                                role="assistant",
                                status="completed",
                                type="message",
                            )
                            # Grounding metadata is available on the candidate and only exists if web search was used.
                            grounding_metadata = candidate.grounding_metadata
                            if grounding_metadata:
                                output.append(
                                    ChatMessage(
                                        message=message_param,
                                        created_by=GeminiProvider.PROVIDER_NAME,
                                        interop={
                                            GeminiProvider.PROVIDER_NAME: {
                                                "grounding_metadata": grounding_metadata.model_dump(exclude_none=True)
                                            }
                                        },
                                    )
                                )
                            else:
                                output.append(
                                    ChatMessage(message=message_param, created_by=GeminiProvider.PROVIDER_NAME)
                                )

                        if part.function_call:
                            tool_call = ResponseFunctionToolCallParam(
                                arguments=json.dumps(part.function_call.args or {}),
                                call_id=str(uuid.uuid4()),
                                name=part.function_call.name or "",
                                type="function_call",
                                id=str(uuid.uuid4()),
                                status="completed",
                            )
                            output.append(ChatMessage(message=tool_call, created_by=GeminiProvider.PROVIDER_NAME))

                        # The OpenAI types don't allow inline data (images) in thoughts, so we ignore them.
                        if part.inline_data and part.inline_data.data and not part.thought:
                            call_id = str(uuid.uuid4())
                            mime_type = part.inline_data.mime_type or "image/png"
                            base64_image = base64.b64encode(part.inline_data.data).decode("utf-8")
                            image_url = f"data:{mime_type};base64,{base64_image}"

                            function_call: ResponseFunctionToolCallParam = {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": "image_generation",
                                "arguments": "{}",
                            }
                            function_call_output: FunctionCallOutput = {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": [
                                    ResponseInputImageContentParam(
                                        type="input_image", detail="auto", image_url=image_url
                                    )
                                ],
                            }

                            output.append(ChatMessage(message=function_call, created_by=GeminiProvider.PROVIDER_NAME))
                            output.append(
                                ChatMessage(message=function_call_output, created_by=GeminiProvider.PROVIDER_NAME)
                            )

        usage = GeminiProvider._convert_usage(response.usage_metadata) if response.usage_metadata else None
        if output:
            output[-1].original_response = response.model_dump(mode="json")
        return RouterResponse(output=output, usage=usage)

    @staticmethod
    def _convert_usage(usage_metadata: GenerateContentResponseUsageMetadata) -> ResponseUsage:
        return ResponseUsage(
            input_tokens=usage_metadata.prompt_token_count or 0,
            output_tokens=usage_metadata.candidates_token_count or 0,
            total_tokens=usage_metadata.total_token_count or 0,
            input_tokens_details=InputTokensDetails(cached_tokens=-1),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=-1),
        )

    # endregion
