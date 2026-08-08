from openai import AsyncOpenAI
import pytest

from interop_router.router import Router, resolve_model


def _openai_client(tag: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="test", base_url=f"http://localhost:8000/{tag}/v1")


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


def test_registration_name_defaults_to_the_provider_name() -> None:
    router = Router()
    default = _openai_client("default")
    explicit = _openai_client("explicit")

    router.register("chat_completions", default)
    assert router._resolve("chat_completions/qwen3.6-27B") == ("chat_completions", "qwen3.6-27B", default)

    router.register("chat_completions", explicit, name="chat_completions")
    assert router._resolve("chat_completions/qwen3.6-27B")[2] is explicit


def test_named_registrations_route_independently() -> None:
    """Each name selects its own client, and the model id keeps every `/` after the first.

    "vllm/nvidia/Qwen3.6-27B-NVFP4" -> the vllm client, model id "nvidia/Qwen3.6-27B-NVFP4".
    """
    router = Router()
    vllm = _openai_client("vllm")
    openrouter = _openai_client("openrouter")
    router.register("chat_completions", vllm, name="vllm")
    router.register("chat_completions", openrouter, name="openrouter")

    assert router._resolve("vllm/nvidia/Qwen3.6-27B-NVFP4") == (
        "chat_completions",
        "nvidia/Qwen3.6-27B-NVFP4",
        vllm,
    )
    assert router._resolve("openrouter/moonshotai/kimi-k2") == (
        "chat_completions",
        "moonshotai/kimi-k2",
        openrouter,
    )


def test_provider_prefix_resolves_to_the_sole_named_registration() -> None:
    """A provider prefix reaches that provider's only client, whatever name it was registered under.

    "chat_completions/qwen3.6-27B" -> the vllm client, model id "qwen3.6-27B".
    """
    router = Router()
    vllm = _openai_client("vllm")
    router.register("chat_completions", vllm, name="vllm")

    assert router._resolve("chat_completions/qwen3.6-27B") == ("chat_completions", "qwen3.6-27B", vllm)


def test_provider_prefix_is_ambiguous_with_several_named_registrations() -> None:
    """A provider prefix cannot pick between two clients of that provider, so it raises instead.

    "chat_completions/qwen3.6-27B" -> ValueError naming "openrouter, vllm".
    """
    router = Router()
    router.register("chat_completions", _openai_client("vllm"), name="vllm")
    router.register("chat_completions", _openai_client("openrouter"), name="openrouter")

    with pytest.raises(ValueError, match=r"Multiple clients are registered.*openrouter, vllm"):
        router._resolve("chat_completions/qwen3.6-27B")


def test_registration_name_takes_precedence_over_provider_name() -> None:
    """A name is matched before the provider catalogs, so both clients stay addressable.

    "chat_completions/..." -> the default client, "vllm/..." -> the named client.
    """
    router = Router()
    default = _openai_client("default")
    vllm = _openai_client("vllm")
    router.register("chat_completions", default)
    router.register("chat_completions", vllm, name="vllm")

    assert router._resolve("chat_completions/qwen3.6-27B")[2] is default
    assert router._resolve("vllm/qwen3.6-27B")[2] is vllm


def test_bare_catalog_model_resolves_through_a_named_registration() -> None:
    """A catalog id needs no prefix, even when its provider's only client has a custom name.

    "gpt-5.6-terra" -> provider "openai" -> the azure client.
    """
    router = Router()
    azure = _openai_client("azure")
    router.register("openai", azure, name="azure")

    assert router._resolve("gpt-5.6-terra") == ("openai", "gpt-5.6-terra", azure)


def test_unknown_model_is_not_routed_to_a_chat_completions_client() -> None:
    """An id matching no catalog entry and no known prefix stays an error, never a fallback.

    "nvidia/Qwen3.6-27B-NVFP4" -> ValueError, since "nvidia" is neither a name nor a provider.
    """
    router = Router()
    router.register("chat_completions", _openai_client("vllm"), name="vllm")

    with pytest.raises(ValueError, match="Unknown model"):
        router._resolve("nvidia/Qwen3.6-27B-NVFP4")


def test_missing_registration_for_resolved_provider() -> None:
    router = Router()
    router.register("chat_completions", _openai_client("vllm"), name="vllm")

    with pytest.raises(ValueError, match="No client registered for provider: anthropic"):
        router._resolve("claude-sonnet-5")


def test_empty_model_id_after_registration_name() -> None:
    router = Router()
    router.register("chat_completions", _openai_client("vllm"), name="vllm")

    with pytest.raises(ValueError, match="Empty model id"):
        router._resolve("vllm/")


@pytest.mark.parametrize(
    ("name", "expected_message"),
    [("openai", "reserved"), ("vllm/v1", "must not contain"), ("", "non-empty")],
)
def test_invalid_registration_name(name: str, expected_message: str) -> None:
    router = Router()

    with pytest.raises(ValueError, match=expected_message):
        router.register("chat_completions", _openai_client("vllm"), name=name)
