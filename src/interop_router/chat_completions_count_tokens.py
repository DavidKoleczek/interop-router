"""Approximate Chat Completions token counts via tiktoken.

Counts are estimates: framing and tool serialization follow OpenAI Chat Completions conventions, not arbitrary chat templates (e.g. Llama on vLLM).

`encoding` may be a tiktoken encoding name (`cl100k_base`, `o200k_base`, ...) or an OpenAI model id that tiktoken knows.
Unknown values fall back to `o200k_base`.
"""

import base64
from collections.abc import Iterable, Mapping, Sequence
from io import BytesIO
import math
import re
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolUnionParam,
)
import tiktoken

_DEFAULT_ENCODING = "o200k_base"
_TOKENS_PER_MESSAGE = 3
_TOKENS_PER_NAME = 1
_REPLY_PRIMING_TOKENS = 3
_TOOLS_OVERHEAD_TOKENS = 9
_SYSTEM_AND_TOOLS_OVERLAP_TOKENS = 4
_TOOL_CHOICE_NONE_TOKENS = 1
_TOOL_CHOICE_FUNCTION_OVERHEAD_TOKENS = 7


def count_tokens(
    messages: Sequence[ChatCompletionMessageParam],
    encoding: str,
    tools: Sequence[ChatCompletionToolUnionParam] | None = None,
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
) -> int:
    """Estimate prompt tokens for a Chat Completions request.

    Args:
        messages: Chat Completions messages (already converted from Responses shape).
        encoding: Tiktoken encoding name or OpenAI model id.
        tools: Optional Chat Completions tools.
        tool_choice: Optional tool choice (affects a small constant overhead).

    Returns:
        Estimated input token count.
    """
    enc = _get_encoding(encoding)
    function_tools = _function_tools(tools)

    total = 0
    remaining: Sequence[ChatCompletionMessageParam] = messages

    if function_tools:
        system_message, remaining = _split_leading_system_message(messages)
        total += _count_tokens_for_system_and_tools(
            enc,
            system_message=system_message,
            tools=function_tools,
            tool_choice=tool_choice,
        )

    for message in remaining:
        total += _count_tokens_for_message(enc, message)

    total += _REPLY_PRIMING_TOKENS
    return total


def _get_encoding(encoding: str) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(encoding)
    except ValueError:
        pass
    try:
        return tiktoken.encoding_for_model(encoding)
    except KeyError:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)


def _function_tools(
    tools: Sequence[ChatCompletionToolUnionParam] | None,
) -> list[dict[str, Any]]:
    if not tools:
        return []
    function_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function_tools.append(cast(dict[str, Any], tool))
    return function_tools


def _split_leading_system_message(
    messages: Sequence[ChatCompletionMessageParam],
) -> tuple[ChatCompletionMessageParam | None, Sequence[ChatCompletionMessageParam]]:
    if not messages:
        return None, messages
    first = messages[0]
    if first.get("role") in ("system", "developer"):
        return first, messages[1:]
    return None, messages


def _count_tokens_for_message(enc: tiktoken.Encoding, message: ChatCompletionMessageParam) -> int:
    num_tokens = _TOKENS_PER_MESSAGE
    for key, value in cast(Mapping[str, Any], message).items():
        if value is None:
            continue
        if key == "tool_calls" and isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            num_tokens += _count_tokens_for_tool_calls(enc, value)
            continue
        if isinstance(value, list):
            num_tokens += _count_tokens_for_content_parts(enc, value)
            continue
        if isinstance(value, str):
            num_tokens += len(enc.encode(value))
            if key == "name":
                num_tokens += _TOKENS_PER_NAME
            continue
        # Ignore non-string metadata (e.g. audio objects) rather than failing the count.
    return num_tokens


def _count_tokens_for_content_parts(enc: tiktoken.Encoding, parts: list[Any]) -> int:
    num_tokens = 0
    for item in parts:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            if isinstance(text, str):
                num_tokens += len(enc.encode(text))
        elif item_type == "refusal":
            refusal = item.get("refusal", "")
            if isinstance(refusal, str):
                num_tokens += len(enc.encode(refusal))
        elif item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, Mapping):
                url = image_url.get("url", "")
                detail = image_url.get("detail", "auto")
                if isinstance(url, str):
                    num_tokens += _count_tokens_for_image(url, detail if isinstance(detail, str) else "auto")
    return num_tokens


def _count_tokens_for_tool_calls(enc: tiktoken.Encoding, tool_calls: Iterable[Any]) -> int:
    num_tokens = 0
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        for key in ("id", "type"):
            value = tool_call.get(key)
            if isinstance(value, str):
                num_tokens += len(enc.encode(value))
        function = tool_call.get("function")
        if isinstance(function, Mapping):
            for key in ("name", "arguments"):
                value = function.get(key)
                if isinstance(value, str):
                    num_tokens += len(enc.encode(value))
    return num_tokens


def _count_tokens_for_system_and_tools(
    enc: tiktoken.Encoding,
    system_message: ChatCompletionMessageParam | None,
    tools: Sequence[Mapping[str, Any]],
    tool_choice: ChatCompletionToolChoiceOptionParam | None,
) -> int:
    """Count system message + tools together.

    OpenAI serializes tools in a form where the combined count is lower when a system message is also present (empirically -4).
    """
    tokens = 0
    if system_message is not None:
        tokens += _count_tokens_for_message(enc, system_message)
    if tools:
        tokens += len(enc.encode(_format_function_definitions(tools)))
        tokens += _TOOLS_OVERHEAD_TOKENS
    if tools and system_message is not None:
        tokens -= _SYSTEM_AND_TOOLS_OVERLAP_TOKENS
    if tool_choice == "none":
        tokens += _TOOL_CHOICE_NONE_TOKENS
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            tokens += _TOOL_CHOICE_FUNCTION_OVERHEAD_TOKENS
            tokens += len(enc.encode(name))
    return tokens


def _format_function_definitions(tools: Sequence[Mapping[str, Any]]) -> str:
    lines = ["namespace functions {", ""]
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        description = function.get("description")
        if isinstance(description, str) and description:
            lines.append(f"// {description}")
        function_name = function.get("name")
        if not isinstance(function_name, str):
            continue
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if isinstance(properties, Mapping) and properties:
            lines.append(f"type {function_name} = (_: {{")
            lines.append(_format_object_parameters(cast(Mapping[str, Any], parameters), 0))
            lines.append("}) => any;")
        else:
            lines.append(f"type {function_name} = () => any;")
        lines.append("")
    lines.append("} // namespace functions")
    return "\n".join(lines)


def _format_object_parameters(parameters: Mapping[str, Any], indent: int) -> str:
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        return ""
    required_params = parameters.get("required", [])
    if not isinstance(required_params, list):
        required_params = []
    lines: list[str] = []
    for key, props in properties.items():
        if not isinstance(props, Mapping):
            continue
        description = props.get("description")
        if isinstance(description, str) and description:
            lines.append(f"// {description}")
        optional = "" if key in required_params else "?"
        lines.append(f"{key}{optional}: {_format_type(cast(Mapping[str, Any], props), indent)},")
    return "\n".join([" " * max(0, indent) + line for line in lines])


def _format_type(props: Mapping[str, Any], indent: int) -> str:
    prop_type = props.get("type")
    if prop_type == "string":
        enum = props.get("enum")
        if isinstance(enum, list):
            return " | ".join(f'"{item}"' for item in enum)
        return "string"
    if prop_type == "array":
        items = props.get("items")
        if isinstance(items, Mapping):
            return f"{_format_type(cast(Mapping[str, Any], items), indent)}[]"
        return "any[]"
    if prop_type == "object":
        return f"{{\n{_format_object_parameters(props, indent + 2)}\n}}"
    if prop_type in ("integer", "number"):
        enum = props.get("enum")
        if isinstance(enum, list):
            return " | ".join(f'"{item}"' for item in enum)
        return "number"
    if prop_type == "boolean":
        return "boolean"
    if prop_type == "null":
        return "null"
    return "any"


def _count_tokens_for_image(image_uri: str, detail: str = "auto") -> int:
    """OpenAI vision tile formula.

    Remote URLs cannot be sized without a fetch; those use the low-detail fixed cost.
    """
    cost_per_tile = 85
    low_detail_cost = cost_per_tile
    high_detail_cost_per_tile = cost_per_tile * 2

    if detail == "auto":
        detail = "high"

    if detail == "low":
        return low_detail_cost

    if detail != "high":
        return low_detail_cost

    dims = _get_image_dims(image_uri)
    if dims is None:
        return low_detail_cost

    width, height = dims
    if max(width, height) > 2048:
        ratio = 2048 / max(width, height)
        width = int(width * ratio)
        height = int(height * ratio)
    if min(width, height) > 768:
        ratio = 768 / min(width, height)
        width = int(width * ratio)
        height = int(height * ratio)
    num_squares = math.ceil(width / 512) * math.ceil(height / 512)
    total_cost = num_squares * high_detail_cost_per_tile + cost_per_tile
    return math.ceil(total_cost)


def _get_image_dims(image_uri: str) -> tuple[int, int] | None:
    if not re.match(r"data:image/\w+;base64", image_uri):
        return None
    # Lazy import: text-only counting should not require Pillow at import time.
    from PIL import Image

    image_data = re.sub(r"data:image/\w+;base64,", "", image_uri)
    try:
        with Image.open(BytesIO(base64.b64decode(image_data, validate=False))) as image:
            return image.size
    except Exception:
        return None
