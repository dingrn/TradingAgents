"""The OpenAI-compatible provider registry is the single source of truth for the
family; this guards each provider's resolved config (base URL, subclass, auth,
Responses API) so a future edit can't silently break one.
"""
import os
import subprocess
import sys

import pytest

from tradingagents.llm_clients import resolve_llm_routing
from tradingagents.llm_clients.openai_client import (
    OPENAI_COMPATIBLE_PROVIDERS,
    DeepSeekChatOpenAI,
    LocalCompatibleChatOpenAI,
    MinimaxChatOpenAI,
    NormalizedChatOpenAI,
    is_openai_compatible,
)


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
        resolve_llm_routing("anthropic", "claude")
