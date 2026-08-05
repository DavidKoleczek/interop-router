import pytest

from interop_router.router import resolve_model


def test_bare_openai_model() -> None:
    assert resolve_model("gpt-5.6-terra") == ("openai", "gpt-5.6-terra")


def test_bare_gemini_model() -> None:
    assert resolve_model("gemini-3.6-flash") == ("gemini", "gemini-3.6-flash")


def test_bare_anthropic_model() -> None:
    assert resolve_model("claude-sonnet-5") == ("anthropic", "claude-sonnet-5")


def test_prefixed_openai() -> None:
    assert resolve_model("openai/gpt-5.6-terra") == ("openai", "gpt-5.6-terra")


def test_prefixed_gemini() -> None:
    assert resolve_model("gemini/gemini-3.6-flash") == ("gemini", "gemini-3.6-flash")


def test_prefixed_anthropic() -> None:
    assert resolve_model("anthropic/claude-sonnet-5") == ("anthropic", "claude-sonnet-5")


def test_prefixed_chat_completions() -> None:
    assert resolve_model("chat_completions/qwen3.6-27B") == ("chat_completions", "qwen3.6-27B")


def test_api_model_id_may_contain_slashes() -> None:
    assert resolve_model("chat_completions/nvidia/Qwen3.6-27B-NVFP4") == (
        "chat_completions",
        "nvidia/Qwen3.6-27B-NVFP4",
    )


def test_prefix_allows_non_catalog_model() -> None:
    assert resolve_model("openai/my-finetune") == ("openai", "my-finetune")


def test_prefix_overrides_catalog_default() -> None:
    assert resolve_model("chat_completions/gpt-5.6-terra") == ("chat_completions", "gpt-5.6-terra")


def test_unknown_bare_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("qwen3.6-27B")


def test_unknown_prefix_treated_as_bare() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("foo/bar")


def test_empty_model_id_after_prefix() -> None:
    with pytest.raises(ValueError, match="Empty model id"):
        resolve_model("openai/")
