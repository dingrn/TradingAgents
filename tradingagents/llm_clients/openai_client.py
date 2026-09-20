import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .api_key_env import get_api_key_env
from .base_client import BaseLLMClient, normalize_content
from .capabilities import get_capabilities
from .routing import (
    CredentialPresence,
    LLMRoutingResolution,
    ResolvedRoutingValue,
    RoutingResolutionError,
    UnsupportedRoutingSelectorError,
    ensure_non_secret_endpoint,
)
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output and capability-aware binding.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling.

    ``with_structured_output`` consults the per-model capability table
    (``capabilities.get_capabilities``) to pick the method and to decide
    whether ``tool_choice`` may be sent. Models that reject ``tool_choice``
    (e.g. DeepSeek V4 and reasoner — per their official tool-calling
    guide) still bind the schema as a tool, but no ``tool_choice``
    parameter is sent.

    Provider-specific quirks beyond structured-output (e.g. DeepSeek's
    reasoning_content roundtrip) live in subclasses so this base class
    stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        caps = get_capabilities(self.model_name)
        if caps.preferred_structured_method == "none":
            raise NotImplementedError(
                f"{self.model_name} has no structured-output method available; "
                f"agent factories will fall back to free-text generation."
            )
        method = method or caps.preferred_structured_method
        # When the model rejects tool_choice, suppress langchain's hardcoded
        # value. The schema is still bound as a tool — exactly what
        # DeepSeek's official tool-calling examples do.
        if method == "function_calling" and not caps.supports_tool_choice:
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


class LocalCompatibleChatOpenAI(NormalizedChatOpenAI):
    """OpenAI-compatible client for arbitrary local servers (LM Studio, vLLM,
    llama.cpp via the generic ``openai_compatible`` provider).

    Their tool-calling support varies, and many reject the object-form
    ``tool_choice`` langchain sends for function-calling structured output. Bind
    the schema as a tool but don't force tool_choice, so structured output works
    across local servers regardless of the model ID's capabilities (#1057).
    """

    def with_structured_output(self, schema, *, method=None, **kwargs):
        resolved = method or get_capabilities(self.model_name).preferred_structured_method
        if resolved == "function_calling":
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Thinking-mode round-trip is the only DeepSeek-specific behavior that
    stays here. When DeepSeek's thinking models return a response with
    ``reasoning_content``, that field must be echoed back as part of the
    assistant message on the next turn or the API fails with HTTP 400.
    ``_create_chat_result`` captures it on receive and
    ``_get_request_payload`` re-attaches it on send.

    Tool-choice handling for V4 and reasoner — those models reject the
    ``tool_choice`` parameter — is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_), strict=False):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", []), strict=False
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result


class MinimaxChatOpenAI(NormalizedChatOpenAI):
    """MiniMax-specific overrides on top of the OpenAI-compatible client.

    M2.x reasoning models embed ``<think>...</think>`` blocks directly in
    ``message.content`` by default, which would pollute saved reports.
    Per platform.minimax.io/docs/api-reference/text-openai-api,
    ``reasoning_split=True`` redirects the thinking block into
    ``reasoning_details`` so ``content`` stays clean. It is sent via
    ``extra_body`` (not a top-level kwarg) because the openai SDK validates
    top-level params and rejects unknown ones like reasoning_split (#826).

    The flag is gated by ``ModelCapabilities.requires_reasoning_split`` so
    only M2.x reasoning models receive it; non-reasoning MiniMax endpoints
    (Coding Plan, MiniMax-Text-01) never see it.

    Tool-choice handling for M2.x — those models accept only the string
    enum ``{"none", "auto"}`` and reject langchain's function-spec dict —
    is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if get_capabilities(self.model_name).requires_reasoning_split:
            # Pass via extra_body, not as a top-level kwarg: the openai SDK
            # (>=1.56) validates top-level params against Completions.create
            # and rejects unknown ones like reasoning_split (#826). extra_body
            # is forwarded into the request body untouched.
            extra_body = payload.setdefault("extra_body", {})
            extra_body.setdefault("reasoning_split", True)
        return payload


# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort", "temperature", "max_tokens",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# OpenAI's ``reasoning_effort`` is only accepted by reasoning models — GPT-5 and
# later, and the o-series. Non-reasoning models (gpt-4.1, gpt-4o, ...) 400 with
# "Unsupported parameter: 'reasoning.effort' is not supported with this model".
# Drop the kwarg for those rather than crash the run.
_OPENAI_REASONING_MODEL = re.compile(r"^(?:gpt-(?:[5-9]|[1-9]\d)|o[1-9])(?:[.-]|$)")


def _supports_reasoning_effort(model: str) -> bool:
    """Whether the (native OpenAI) model accepts ``reasoning_effort``."""
    return bool(_OPENAI_REASONING_MODEL.match(model.lower().strip()))


@dataclass(frozen=True)
class ProviderSpec:
    """Declarative config for one OpenAI-compatible provider.

    The OpenAI-compatible family (OpenAI, xAI, DeepSeek, Qwen, GLM, MiniMax,
    OpenRouter, Ollama, and any user endpoint) all speak the same Chat
    Completions API and differ only by these fields — so one row here replaces
    the former per-provider base-URL dict, auth handling, and client-class
    branches. Native Anthropic / Google use their own clients (genuinely
    different APIs) and are intentionally NOT in this registry.

    The API-key env var stays in ``api_key_env.PROVIDER_API_KEY_ENV`` (the single
    source consulted by both this client and the CLI prompt); only behavior that
    is provider-specific (base URL, key optionality, wire-format quirks via
    ``chat_class``) lives here.
    """

    chat_class: type = NormalizedChatOpenAI   # provider quirks live in the subclass
    base_url: str | None = None            # default endpoint (None -> SDK default)
    base_url_env: str | None = None        # env var that overrides base_url (e.g. OLLAMA_BASE_URL)
    key_optional: bool = False                # don't require/prompt; send a placeholder if unset
    placeholder_key: str = "EMPTY"            # sent when no key is available (keyless local servers)
    require_base_url: bool = False            # error if no base_url is resolved (generic endpoint)
    use_responses_api: bool = False           # native OpenAI Responses API


# Single source of truth for the OpenAI-compatible provider family. Dual-region
# providers (qwen/glm/minimax) keep separate endpoints because international and
# China accounts cannot share credentials (#758).
OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderSpec] = {
    "openai":     ProviderSpec(use_responses_api=True),
    "xai":        ProviderSpec(base_url="https://api.x.ai/v1"),
    "deepseek":   ProviderSpec(base_url="https://api.deepseek.com", chat_class=DeepSeekChatOpenAI),
    "qwen":       ProviderSpec(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "qwen-cn":    ProviderSpec(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm":        ProviderSpec(base_url="https://api.z.ai/api/paas/v4/"),
    "glm-cn":     ProviderSpec(base_url="https://open.bigmodel.cn/api/paas/v4/"),
    "minimax":    ProviderSpec(base_url="https://api.minimax.io/v1", chat_class=MinimaxChatOpenAI),
    "minimax-cn": ProviderSpec(base_url="https://api.minimaxi.com/v1", chat_class=MinimaxChatOpenAI),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1"),
    "mistral":    ProviderSpec(base_url="https://api.mistral.ai/v1"),
    "kimi":       ProviderSpec(base_url="https://api.moonshot.ai/v1"),
    "groq":       ProviderSpec(base_url="https://api.groq.com/openai/v1"),
    "nvidia":     ProviderSpec(base_url="https://integrate.api.nvidia.com/v1"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1", base_url_env="OLLAMA_BASE_URL",
                               key_optional=True, placeholder_key="ollama",
                               chat_class=LocalCompatibleChatOpenAI),
    # Generic endpoint: user supplies base_url; key optional (keyless local).
    "openai_compatible": ProviderSpec(
        require_base_url=True, key_optional=True, chat_class=LocalCompatibleChatOpenAI
    ),
}


def is_openai_compatible(provider: str) -> bool:
    """Whether ``provider`` is served by the OpenAI-compatible registry."""
    return provider.lower() in OPENAI_COMPATIBLE_PROVIDERS


def _is_native_openai_base_url(base_url: str | None) -> bool:
    """True when ``base_url`` is unset or points at api.openai.com.

    The Responses API (/v1/responses) only exists on native OpenAI. A custom
    base_url on the ``openai`` provider (a proxy, gateway, or local server)
    speaks only Chat Completions, so the Responses API must stay off there even
    though the provider spec enables it (#1024).
    """
    if not base_url:
        return True
    if "://" not in base_url:
        base_url = "https://" + base_url
    host = urlparse(base_url).hostname or ""
    return host == "api.openai.com" or host.endswith(".openai.com")


_OPENAI_SDK_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_UNSUPPORTED_ROUTING_KWARGS = frozenset(
    {
        "api_version",
        "azure_deployment",
        "azure_endpoint",
        "deployment_name",
        "openai_api_base",
        "use_responses_api",
    }
)


@dataclass(frozen=True)
class _OpenAISettings:
    routing: LLMRoutingResolution
    api_key: str | None
    client_base_url: str | None
    chat_class: type


def _resolve_openai_settings(
    provider: str,
    model: str,
    base_url: str | None,
    environment: Mapping[str, str],
    kwargs: Mapping[str, Any],
) -> _OpenAISettings:
    provider = provider.lower()
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    if spec is None:
        raise RoutingResolutionError(f"Unsupported OpenAI-compatible provider: {provider}")

    unsupported = sorted(_UNSUPPORTED_ROUTING_KWARGS.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise UnsupportedRoutingSelectorError(
            f"Provider '{provider}' routing cannot resolve unsupported selector(s): {names}"
        )

    provider_environment_url = environment.get(spec.base_url_env) if spec.base_url_env else None
    registry_base_url: str | None
    if base_url:
        registry_base_url = base_url
        endpoint = ResolvedRoutingValue(base_url, "explicit")
    elif provider_environment_url:
        registry_base_url = provider_environment_url
        endpoint = ResolvedRoutingValue(
            provider_environment_url, "environment", spec.base_url_env
        )
    elif spec.base_url:
        registry_base_url = spec.base_url
        endpoint = ResolvedRoutingValue(spec.base_url, "provider_default")
    elif provider == "openai" and environment.get("OPENAI_BASE_URL"):
        registry_base_url = None
        endpoint = ResolvedRoutingValue(
            environment["OPENAI_BASE_URL"], "sdk_environment", "OPENAI_BASE_URL"
        )
    elif provider == "openai":
        registry_base_url = None
        endpoint = ResolvedRoutingValue(_OPENAI_SDK_DEFAULT_BASE_URL, "sdk_default")
    else:
        raise RoutingResolutionError(
            f"Provider '{provider}' requires a base_url. Set it via backend_url / "
            "TRADINGAGENTS_LLM_BACKEND_URL to your endpoint, e.g. "
            "http://localhost:8000/v1 (vLLM) or http://localhost:1234/v1 (LM Studio)."
        )
    ensure_non_secret_endpoint(endpoint.value, "OpenAI-compatible endpoint")

    # Preserve the existing protocol rule: the provider-owned URL selection
    # controls whether native OpenAI uses Responses. OPENAI_BASE_URL remains an
    # SDK-level endpoint override and does not silently rewrite that selection.
    use_responses_api = spec.use_responses_api and _is_native_openai_base_url(
        registry_base_url
    )
    protocol = ResolvedRoutingValue(
        "responses" if use_responses_api else "chat_completions",
        "provider_registry",
    )

    api_key_environment = get_api_key_env(provider)
    explicit_api_key = kwargs.get("api_key")
    environment_api_key = (
        environment.get(api_key_environment) if api_key_environment else None
    )
    if explicit_api_key:
        api_key = explicit_api_key
        credential = CredentialPresence(True, "api_key", "explicit")
    elif environment_api_key:
        api_key = environment_api_key
        credential = CredentialPresence(
            True, "api_key", "environment", api_key_environment
        )
    elif spec.key_optional:
        api_key = spec.placeholder_key
        credential = CredentialPresence(
            False, "api_key", "optional", api_key_environment
        )
    else:
        api_key = None
        credential = CredentialPresence(
            False, "api_key", "missing", api_key_environment
        )

    routing_environment_variables = tuple(
        name
        for name in (spec.base_url_env, "OPENAI_BASE_URL" if provider == "openai" else None)
        if name
    )
    routing = LLMRoutingResolution(
        provider=provider,
        model=model,
        endpoint=endpoint,
        protocol=protocol,
        credential=credential,
        routing_environment_variables=routing_environment_variables,
    )
    client_base_url = None if endpoint.source == "sdk_default" else endpoint.value
    return _OpenAISettings(routing, api_key, client_base_url, spec.chat_class)


def resolve_openai_routing(
    provider: str,
    model: str,
    base_url: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> LLMRoutingResolution:
    """Resolve OpenAI-compatible routing without exposing credential values."""
    selected_environment = os.environ if environment is None else environment
    return _resolve_openai_settings(
        provider, model, base_url, selected_environment, kwargs
    ).routing


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return a configured ChatOpenAI instance, driven by the provider registry."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        settings = _resolve_openai_settings(
            self.provider, self.model, self.base_url, os.environ, self.kwargs
        )
        if settings.client_base_url:
            llm_kwargs["base_url"] = settings.client_base_url
        if settings.api_key:
            llm_kwargs["api_key"] = settings.api_key
        elif settings.routing.credential.environment_variable:
            api_key_environment = settings.routing.credential.environment_variable
            raise ValueError(
                f"API key for provider '{self.provider}' is not set. "
                f"Please set the {api_key_environment} environment variable "
                f"(e.g. add {api_key_environment}=your_key to your .env file)."
            )
        if settings.routing.protocol.value == "responses":
            llm_kwargs["use_responses_api"] = True

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "api_key":
                continue
            if key == "reasoning_effort" and not _supports_reasoning_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # The subclass (provider quirks) comes from the registry spec.
        return settings.chat_class(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
