"""Integration test that validates all supported models with a simple request."""

import os
from typing import get_args

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
import pytest

from interop_router.router import Router
from interop_router.types import (
    ChatMessage,
    SupportedModel,
    SupportedModelAnthropic,
    SupportedModelGemini,
    SupportedModelOpenAI,
)

ALL_MODELS: list[SupportedModel] = [
    *get_args(SupportedModelOpenAI),
    *get_args(SupportedModelGemini),
    *get_args(SupportedModelAnthropic),
]


@pytest.fixture
def router() -> Router:
    router = Router()
    router.register("openai", AsyncOpenAI())
    router.register("gemini", genai.Client(api_key=os.getenv("GEMINI_API_KEY")))
    router.register("anthropic", AsyncAnthropic())
    return router


@pytest.mark.parametrize("model", ALL_MODELS)
async def test_model(router: Router, model: SupportedModel) -> None:
    """Test a single model with a simple request."""
    message = ChatMessage(message=EasyInputMessageParam(role="user", content="Say hello in one word."))
    response = await router.create(input=[message], model=model)
    assert response.output
