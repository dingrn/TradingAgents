"""Generic OpenAI-compatible provider (vLLM / LM Studio / llama.cpp / relays).

Verifies the user-supplied base_url is required and honored, the key is optional
(keyless local default), Chat Completions (not the Responses API) is used, any
model name is accepted, and the env backend URL precedence (#978).
"""

import pytest

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client, resolve_llm_routing
from tradingagents.llm_clients.routing import (
    RoutingResolutionError,
    UnsupportedRoutingSelectorError,
)
from tradingagents.llm_clients.validators import validate_model

# Note: assert by class NAME, not isinstance — other tests reload the
# openai_client module, which would otherwise create a second class identity.


@pytest.mark.unit
def test_factory_routes_to_openai_client():
    client = create_llm_client(
        provider="openai_compatible", model="my-model", base_url="http://localhost:8000/v1"
    )
    assert type(client).__name__ == "OpenAIClient"


@pytest.mark.unit
def test_base_url_required(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires a base_url"):
        create_llm_client(provider="openai_compatible", model="m").get_llm()


@pytest.mark.unit
def test_keyless_local_uses_placeholder_and_chat_completions(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    llm = create_llm_client(
        provider="openai_compatible", model="qwen2.5", base_url="http://localhost:8000/v1"
    ).get_llm()
    assert type(llm).__name__ == "LocalCompatibleChatOpenAI"
    assert str(llm.openai_api_base) == "http://localhost:8000/v1"
    # keyless local servers: a placeholder key is sent
    key = llm.openai_api_key.get_secret_value() if hasattr(llm.openai_api_key, "get_secret_value") else llm.openai_api_key
    assert key == "EMPTY"
    # must use Chat Completions, not OpenAI's Responses API
    assert getattr(llm, "use_responses_api", False) in (False, None)


@pytest.mark.unit
def test_optional_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-relay-123")
    llm = create_llm_client(
        provider="openai_compatible", model="m", base_url="https://relay.example/v1"
    ).get_llm()
    key = llm.openai_api_key.get_secret_value() if hasattr(llm.openai_api_key, "get_secret_value") else llm.openai_api_key
    assert key == "sk-relay-123"


@pytest.mark.unit
def test_any_model_accepted_no_forced_key():
    assert validate_model("openai_compatible", "literally-anything") is True
    # The key env exists (read for keyed relays) but the provider is marked
    # key-optional, so the CLI never forces a prompt and keyless servers work.
    assert get_api_key_env("openai_compatible") == "OPENAI_COMPATIBLE_API_KEY"
    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS
    assert OPENAI_COMPATIBLE_PROVIDERS["openai_compatible"].key_optional is True


@pytest.mark.unit
def test_env_backend_url_precedence():
    # #978: explicit env URL wins over the menu/default regardless of provider source.
    from cli.utils import resolve_backend_url
    assert resolve_backend_url("openai", "https://api.openai.com/v1", env_url="http://proxy/v1") == "http://proxy/v1"
    assert resolve_backend_url("openai", "https://api.openai.com/v1", env_url=None) == "https://api.openai.com/v1"
    assert resolve_backend_url("deepseek", None, None) == "https://api.deepseek.com"


@pytest.mark.unit
def test_routing_resolution_uses_explicit_environment_without_mutating_process(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ambient.invalid/v1")
    environment = {
        "OLLAMA_BASE_URL": "http://selected.example/v1",
        "OPENAI_API_KEY": "must-not-leak",
    }

    resolved = resolve_llm_routing(
        "ollama", "qwen3:30b", environment=environment
    )

    assert resolved.endpoint.value == "http://selected.example/v1"
    assert resolved.endpoint.source == "environment"
    assert resolved.endpoint.environment_variable == "OLLAMA_BASE_URL"
    assert resolved.protocol.value == "chat_completions"
    assert resolved.credential.present is False
    assert resolved.routing_environment_variables == ("OLLAMA_BASE_URL",)
    assert "must-not-leak" not in repr(resolved)
    assert __import__("os").environ["OLLAMA_BASE_URL"] == "http://ambient.invalid/v1"


@pytest.mark.unit
def test_native_openai_resolves_sdk_environment_and_credential_source():
    environment = {
        "OPENAI_BASE_URL": "https://gateway.example/v1",
        "OPENAI_API_KEY": "secret-value",
    }

    resolved = resolve_llm_routing("openai", "gpt-5", environment=environment)

    assert resolved.endpoint.value == "https://gateway.example/v1"
    assert resolved.endpoint.source == "sdk_environment"
    assert resolved.protocol.value == "responses"
    assert resolved.credential.present is True
    assert resolved.credential.source == "environment"
    assert resolved.credential.environment_variable == "OPENAI_API_KEY"
    assert "secret-value" not in repr(resolved)


@pytest.mark.unit
def test_explicit_base_url_and_key_take_precedence_without_exposing_key():
    resolved = resolve_llm_routing(
        "openai",
        "gpt-5",
        "https://explicit.example/v1",
        environment={
            "OPENAI_BASE_URL": "https://ambient.example/v1",
            "OPENAI_API_KEY": "ambient-secret",
        },
        api_key="explicit-secret",
    )

    assert resolved.endpoint.value == "https://explicit.example/v1"
    assert resolved.endpoint.source == "explicit"
    assert resolved.protocol.value == "chat_completions"
    assert resolved.credential.source == "explicit"
    assert "explicit-secret" not in repr(resolved)
    assert "ambient-secret" not in repr(resolved)


@pytest.mark.unit
def test_unsupported_openai_sdk_selector_fails_explicitly():
    with pytest.raises(UnsupportedRoutingSelectorError, match="use_responses_api"):
        resolve_llm_routing(
            "openai",
            "gpt-5",
            environment={"OPENAI_API_KEY": "secret"},
            use_responses_api=False,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:password@gateway.example/v1",
        "https://gateway.example/v1?api-key=secret-value",
    ],
)
def test_credential_bearing_endpoint_is_not_returned(endpoint):
    with pytest.raises(RoutingResolutionError, match="credential-bearing") as exc_info:
        resolve_llm_routing(
            "openai", "gpt-5", endpoint, environment={"OPENAI_API_KEY": "secret"}
        )

    assert "password" not in str(exc_info.value)
    assert "secret-value" not in str(exc_info.value)


@pytest.mark.unit
def test_structured_output_suppresses_object_tool_choice(monkeypatch):
    # LM Studio / vLLM reject the object-form tool_choice langchain sends for
    # function-calling structured output (#1057). The generic provider binds the
    # schema as a tool but must not force tool_choice.
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

    class Schema(BaseModel):
        x: int

    captured = {}
    monkeypatch.setattr(
        ChatOpenAI,
        "with_structured_output",
        lambda self, schema, method=None, **kw: captured.update({"method": method, **kw}) or "BOUND",
    )
    llm = create_llm_client(
        provider="openai_compatible", model="local-llm-30b", base_url="http://localhost:1234/v1"
    ).get_llm()
    out = llm.with_structured_output(Schema)
    assert out == "BOUND"
    assert captured["method"] == "function_calling"
    assert captured["tool_choice"] is None  # not the object form
