---
name: supported-model-update
description: Run this to update supported models
disable-model-invocation: true
---

Update `SupportedModelOpenAI`, `SupportedModelGemini`, and `SupportedModelAnthropic` in `src/interop_router/types.py` based on the installed package type definitions.

1. OpenAI instructions
   - Read the OpenAI models from two files to find any latest models
     - `ai_working/openai-python/src/openai/types/shared/chat_model.py` - the `ChatModel` Literal
     - `ai_working/openai-python/src/openai/types/shared_params/responses_model.py` - the additional models in the `ResponsesModel` Literal (excluding `str` and `ChatModel`)
   - Update the `SupportedModelOpenAI` type alias in `src/interop_router/types.py`, based on these rules.
     - Any newer models. For example, if gpt-5.1 is already in our supported models, but gpt-5.2 is in the OpenAI package, add gpt-5.2.
     - Do not add in older models.

2. Gemini instructions
   - Read the Gemini models from `ai_working/python-genai/google/genai/_interactions/types/model.py`. Use these for `SupportedModelGemini`.
   - Update the `SupportedModelGemini` type alias in `src/interop_router/types.py`, based on these rules.
     - Any newer models. For example, if gemini-2.5 is already in our supported models, but gemini-3 is in the Gemini package, add gemini-3.
     - Do not add in older models.
   - Do not add text to speech or audio specific models like `gemini-2.5-pro-preview-tts` or `gemini-2.5-flash-preview-native-audio-dialog`
   - Do not add the image models separately like `gemini-3-pro-image-preview` since they are included as part tool calling for the text model.

3. Anthropic instructions
   - Read the Anthropic models from `ai_working/anthropic-sdk-python/src/anthropic/types/model_param.py` - the `ModelParam` Union (excluding `str`)
   - Update the `SupportedModelAnthropic` type alias in `src/interop_router/types.py`, based on these rules.
     - Any newer models. For example, if claude-sonnet-4-5 is already in our supported models, but claude-sonnet-4-6 is in the Anthropic package, add claude-sonnet-4-6.
     - Do not add in older models.

NOTE: We might have to add some models manually. These should be denoted with a comment and not overwritten.

1. Run checks: `uv run ruff format && uv run ruff check --fix && uv run ty check`. Only fix issues related to the changes you just made.
