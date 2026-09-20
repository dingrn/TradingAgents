"""The OpenAI-compatible provider registry is the single source of truth for the
family; this guards each provider's resolved config (base URL, subclass, auth,
Responses API) so a future edit can't silently break one.
"""
import os
import subprocess
import sys

import pytest

from tradingagents.llm_clients import (
    UnsupportedRoutingSelectorError,
    create_llm_client,
    resolve_llm_routing,
)
from tradingagents.llm_clients.openai_client import (
    OPENAI_COMPATIBLE_PROVIDERS,
    DeepSeekChatOpenAI,
    LocalCompatibleChatOpenAI,
    MinimaxChatOpenAI,
    NormalizedChatOpenAI,
    is_openai_compatible,
)

_AMBIENT_ROUTING_ENV_VARS = (
    "ANTHROPIC_API_URL",
    "ANTHROPIC_BASE_URL",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "LANGSMITH_GATEWAY",
    "LANGSMITH_GATEWAY_API_KEY",
    "LANGSMITH_API_KEY",
)


@pytest.fixture()
def clean_routing_env(monkeypatch):
    """Drop ambient endpoint/gateway selectors so precedence is exactly asserted."""
    for name in _AMBIENT_ROUTING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
def test_registry_membership():
    assert is_openai_compatible("openai")
    assert is_openai_compatible("openai_compatible")  # the generic endpoint
    # native (different API) clients are intentionally NOT in the registry
    assert not is_openai_compatible("anthropic")
    assert not is_openai_compatible("google")
    assert not is_openai_compatible("azure")


@pytest.mark.unit
@pytest.mark.parametrize("provider,base_url,chat_class,responses", [
    ("openai", None, NormalizedChatOpenAI, True),
    ("xai", "https://api.x.ai/v1", NormalizedChatOpenAI, False),
    ("deepseek", "https://api.deepseek.com", DeepSeekChatOpenAI, False),
    ("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", NormalizedChatOpenAI, False),
    ("qwen-cn", "https://dashscope.aliyuncs.com/compatible-mode/v1", NormalizedChatOpenAI, False),
    ("glm", "https://api.z.ai/api/paas/v4/", NormalizedChatOpenAI, False),
    ("glm-cn", "https://open.bigmodel.cn/api/paas/v4/", NormalizedChatOpenAI, False),
    ("minimax", "https://api.minimax.io/v1", MinimaxChatOpenAI, False),
    ("minimax-cn", "https://api.minimaxi.com/v1", MinimaxChatOpenAI, False),
    ("openrouter", "https://openrouter.ai/api/v1", NormalizedChatOpenAI, False),
    ("mistral", "https://api.mistral.ai/v1", NormalizedChatOpenAI, False),
    ("kimi", "https://api.moonshot.ai/v1", NormalizedChatOpenAI, False),
    ("groq", "https://api.groq.com/openai/v1", NormalizedChatOpenAI, False),
    ("nvidia", "https://integrate.api.nvidia.com/v1", NormalizedChatOpenAI, False),
    ("ollama", "http://localhost:11434/v1", LocalCompatibleChatOpenAI, False),
])
def test_registry_spec(provider, base_url, chat_class, responses):
    spec = OPENAI_COMPATIBLE_PROVIDERS[provider]
    assert spec.base_url == base_url
    assert spec.chat_class is chat_class
    assert spec.use_responses_api is responses


@pytest.mark.unit
def test_key_optionality():
    # Local/generic endpoints are key-optional; hosted APIs require a key.
    assert OPENAI_COMPATIBLE_PROVIDERS["ollama"].key_optional is True
    assert OPENAI_COMPATIBLE_PROVIDERS["openai_compatible"].key_optional is True
    assert OPENAI_COMPATIBLE_PROVIDERS["openai_compatible"].require_base_url is True
    assert OPENAI_COMPATIBLE_PROVIDERS["xai"].key_optional is False
    # OLLAMA_BASE_URL is the only base-URL env override.
    assert OPENAI_COMPATIBLE_PROVIDERS["ollama"].base_url_env == "OLLAMA_BASE_URL"


@pytest.mark.unit
def test_factory_routing_imports_remain_lazy():
    code = (
        "import sys; import tradingagents.llm_clients.factory; "
        "assert 'tradingagents.llm_clients.openai_client' not in sys.modules; "
        "assert 'tradingagents.llm_clients.azure_client' not in sys.modules"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


@pytest.mark.unit
def test_azure_routing_resolves_environment_and_model_default():
    environment = {
        "AZURE_OPENAI_ENDPOINT": "https://selected.openai.azure.com/",
        "OPENAI_API_VERSION": "2025-03-01-preview",
        "AZURE_OPENAI_API_KEY": "must-not-leak",
    }

    resolved = resolve_llm_routing("azure", "gpt-deployment", environment=environment)

    assert resolved.endpoint.value == "https://selected.openai.azure.com/"
    assert resolved.endpoint.environment_variable == "AZURE_OPENAI_ENDPOINT"
    assert resolved.deployment.value == "gpt-deployment"
    assert resolved.deployment.source == "model_default"
    assert resolved.api_version.value == "2025-03-01-preview"
    assert resolved.protocol.value == "azure_chat_completions"
    assert resolved.credential.present is True
    assert resolved.credential.environment_variable == "AZURE_OPENAI_API_KEY"
    assert "must-not-leak" not in repr(resolved)


@pytest.mark.unit
def test_azure_routing_deployment_override_and_client_share_resolution(monkeypatch):
    from tradingagents.llm_clients import azure_client

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://resource.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "selected-deployment")
    monkeypatch.setenv("OPENAI_API_VERSION", "2026-01-01-preview")
    monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
    captured = {}

    class StubAzureChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(azure_client, "NormalizedAzureChatOpenAI", StubAzureChat)
    resolved = azure_client.resolve_azure_routing(
        "fallback-model", environment=dict(os.environ)
    )
    azure_client.AzureOpenAIClient("fallback-model").get_llm()

    assert resolved.deployment.value == "selected-deployment"
    assert captured["azure_endpoint"] == resolved.endpoint.value
    assert captured["azure_deployment"] == resolved.deployment.value
    assert captured["api_version"] == resolved.api_version.value


@pytest.mark.unit
@pytest.mark.parametrize("explicit_key", [None, "explicit-secret"])
def test_azure_ad_token_preserves_existing_precedence(explicit_key):
    environment = {
        "AZURE_OPENAI_ENDPOINT": "https://resource.openai.azure.com/",
        "OPENAI_API_VERSION": "2026-01-01-preview",
        "AZURE_OPENAI_API_KEY": "environment-secret",
        "AZURE_OPENAI_AD_TOKEN": "token-secret",
    }
    kwargs = {"api_key": explicit_key} if explicit_key else {}

    resolved = resolve_llm_routing(
        "azure", "deployment", environment=environment, **kwargs
    )

    assert resolved.credential.kind == "azure_ad_token"
    assert resolved.credential.environment_variable == "AZURE_OPENAI_AD_TOKEN"
    assert "environment-secret" not in repr(resolved)
    assert "token-secret" not in repr(resolved)
    assert "explicit-secret" not in repr(resolved)


@pytest.mark.unit
@pytest.mark.parametrize(
    "environment,missing",
    [
        ({"OPENAI_API_VERSION": "2025-01-01"}, "AZURE_OPENAI_ENDPOINT"),
        ({"AZURE_OPENAI_ENDPOINT": "https://example.invalid"}, "OPENAI_API_VERSION"),
    ],
)
def test_azure_routing_fails_when_required_sdk_selector_is_unresolved(
    environment, missing
):
    with pytest.raises(ValueError, match=missing):
        resolve_llm_routing("azure", "deployment", environment=environment)


@pytest.mark.unit
def test_unimplemented_provider_routing_is_not_guessed():
    with pytest.raises(ValueError, match="not implemented"):
        resolve_llm_routing("cohere", "command-r")


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_url,environment,endpoint,source,variable",
    [
        (
            "https://explicit.example/",
            {"ANTHROPIC_API_URL": "https://api-url.example/"},
            "https://explicit.example/",
            "explicit",
            None,
        ),
        (
            None,
            {
                "ANTHROPIC_API_URL": "https://api-url.example/",
                "ANTHROPIC_BASE_URL": "https://base.example/",
            },
            "https://api-url.example/",
            "environment",
            "ANTHROPIC_API_URL",
        ),
        (
            None,
            {"ANTHROPIC_BASE_URL": "https://base.example/"},
            "https://base.example/",
            "environment",
            "ANTHROPIC_BASE_URL",
        ),
        (
            None,
            {"LANGSMITH_GATEWAY": "https://gw.example/"},
            "https://gw.example/anthropic",
            "gateway",
            "LANGSMITH_GATEWAY",
        ),
        (None, {}, "https://api.anthropic.com", "sdk_default", None),
    ],
)
def test_anthropic_endpoint_precedence(base_url, environment, endpoint, source, variable):
    resolved = resolve_llm_routing(
        "anthropic", "claude-sonnet-5", base_url, environment=environment
    )

    assert resolved.endpoint.value == endpoint
    assert resolved.endpoint.source == source
    assert resolved.endpoint.environment_variable == variable
    assert resolved.protocol.value == "anthropic_messages"
    assert "ANTHROPIC_PROXY" in resolved.routing_environment_variables


@pytest.mark.unit
@pytest.mark.parametrize(
    "environment,expected",
    [
        (
            {
                "LANGSMITH_GATEWAY": "true",
                "LANGSMITH_GATEWAY_API_KEY": "gateway-secret",
                "ANTHROPIC_API_KEY": "provider-secret",
            },
            "LANGSMITH_GATEWAY_API_KEY",
        ),
        (
            {
                "LANGSMITH_GATEWAY": "true",
                "LANGSMITH_GATEWAY_API_KEY": "gateway-secret",
                "ANTHROPIC_API_KEY": "provider-secret",
                "ANTHROPIC_API_URL": "https://direct.example/",
            },
            "ANTHROPIC_API_KEY",
        ),
    ],
)
def test_anthropic_gateway_key_follows_the_endpoint_it_belongs_to(environment, expected):
    # The gateway key is only sent when the endpoint itself came from the gateway.
    resolved = resolve_llm_routing("anthropic", "claude-sonnet-5", environment=environment)

    assert resolved.credential.present is True
    assert resolved.credential.environment_variable == expected
    assert "gateway-secret" not in repr(resolved)
    assert "provider-secret" not in repr(resolved)


@pytest.mark.unit
def test_anthropic_resolution_matches_the_installed_sdk(monkeypatch, clean_routing_env):
    # Fails loudly if langchain-anthropic changes the rule this mirrors.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://relay.example/")

    resolved = resolve_llm_routing(
        "anthropic", "claude-sonnet-5", environment=dict(os.environ)
    )
    llm = create_llm_client("anthropic", "claude-sonnet-5").get_llm()

    assert llm.anthropic_api_url == resolved.endpoint.value == "https://relay.example/"


@pytest.mark.unit
def test_anthropic_selectors_the_client_never_forwards_are_not_guessed():
    with pytest.raises(UnsupportedRoutingSelectorError, match="anthropic_proxy"):
        resolve_llm_routing(
            "anthropic",
            "claude-sonnet-5",
            environment={},
            anthropic_proxy="http://proxy.example",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_url,environment,endpoint,source",
    [
        (None, {}, "https://generativelanguage.googleapis.com/", "sdk_default"),
        ("https://explicit.example/", {}, "https://explicit.example/", "explicit"),
        (
            None,
            {"LANGSMITH_GATEWAY": "true"},
            "https://gateway.smith.langchain.com/gemini",
            "gateway",
        ),
    ],
)
def test_google_developer_endpoint_precedence(base_url, environment, endpoint, source):
    resolved = resolve_llm_routing(
        "google", "gemini-3.5-flash", base_url, environment=environment
    )

    assert resolved.endpoint.value == endpoint
    assert resolved.endpoint.source == source
    assert resolved.protocol.value == "gemini_developer"
    assert resolved.api_version.value == "v1beta"


@pytest.mark.unit
@pytest.mark.parametrize(
    "environment,expected",
    [
        ({"GOOGLE_API_KEY": "primary", "GEMINI_API_KEY": "fallback"}, "GOOGLE_API_KEY"),
        ({"GEMINI_API_KEY": "fallback"}, "GEMINI_API_KEY"),
    ],
)
def test_google_credential_order_matches_the_sdk(environment, expected):
    resolved = resolve_llm_routing(
        "google", "gemini-3.5-flash", environment=environment
    )

    assert resolved.credential.environment_variable == expected
    assert "primary" not in repr(resolved)
    assert "fallback" not in repr(resolved)


@pytest.mark.unit
@pytest.mark.parametrize(
    "environment,kwargs",
    [
        ({"GOOGLE_GENAI_USE_VERTEXAI": "true"}, {}),
        ({"GOOGLE_GENAI_USE_VERTEXAI": "1"}, {}),
        ({}, {"project": "my-project"}),
        ({}, {"location": "us-central1"}),
        ({}, {"vertexai": True}),
        ({}, {"credentials": "adc-object"}),
    ],
)
def test_google_vertex_selection_is_reported_as_unsupported(environment, kwargs):
    # Vertex routing depends on project/credential state this client cannot read,
    # so it is rejected rather than guessed.
    with pytest.raises(UnsupportedRoutingSelectorError):
        resolve_llm_routing(
            "google", "gemini-3.5-flash", environment=environment, **kwargs
        )


@pytest.mark.unit
def test_google_client_construction_is_unchanged_when_vertex_is_selected(monkeypatch):
    from tradingagents.llm_clients import google_client

    captured = {}

    class StubGoogleChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        google_client, "NormalizedChatGoogleGenerativeAI", StubGoogleChat
    )
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

    google_client.GoogleClient("gemini-3.5-flash", api_key="explicit-key").get_llm()

    assert captured["google_api_key"] == "explicit-key"
    assert "base_url" not in captured


@pytest.mark.unit
def test_google_resolution_matches_the_installed_sdk(monkeypatch, clean_routing_env):
    # Fails loudly if langchain-google-genai changes the rule this mirrors.
    monkeypatch.setenv("LANGSMITH_GATEWAY", "https://gw.example")

    resolved = resolve_llm_routing(
        "google", "gemini-3.5-flash", environment=dict(os.environ)
    )
    llm = create_llm_client(
        "google", "gemini-3.5-flash", api_key="placeholder"
    ).get_llm()
    http_options = llm.client._api_client._http_options

    assert http_options.base_url == resolved.endpoint.value == "https://gw.example/gemini"
    assert http_options.api_version == resolved.api_version.value
