import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .routing import (
    GATEWAY_ENDPOINT_ENVIRONMENT_VARIABLE,
    LLMRoutingResolution,
    ResolvedRoutingValue,
    UnsupportedRoutingSelectorError,
    ensure_non_secret_endpoint,
    resolve_gateway_routing,
)
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort",
)

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+,
# Sonnet 4.6+, and the Claude 5 family (Sonnet 5, Fable 5). Sonnet 4.5 and any
# Haiku version 400 with ``"This model does not support the effort parameter"``
# (#831). Versions may be dotted (``opus-4-8``) or single-number (``sonnet-5``,
# ``fable-5``); the per-family minimum below is forward-compatible.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
    "claude-mythos-5",        # Fable 5 twin (Project Glasswing); effort-capable
}
_EFFORT_MODEL = re.compile(r"^claude-(opus|sonnet|fable)-(\d+)(?:-(\d+))?$")
_EFFORT_MIN_VERSION = {"opus": (4, 5), "sonnet": (4, 6), "fable": (5, 0)}


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    if model_lc in _EFFORT_EXACT:
        return True
    match = _EFFORT_MODEL.match(model_lc)
    if not match:
        return False
    family = match.group(1)
    major = int(match.group(2))
    minor = int(match.group(3)) if match.group(3) else 0
    return (major, minor) >= _EFFORT_MIN_VERSION[family]


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


# ChatAnthropic reads these when no explicit base URL is supplied, in this order,
# before falling back to the LangSmith gateway and then its own default.
_ENDPOINT_ENVIRONMENT_VARIABLES = ("ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL")
_CREDENTIAL_ENVIRONMENT_VARIABLES = ("ANTHROPIC_API_KEY",)
_SDK_DEFAULT_ENDPOINT = "https://api.anthropic.com"
# A proxy redirects every call, but its URL can carry credentials, so the name is
# reported as a routing input instead of its value being resolved.
_PROXY_ENVIRONMENT_VARIABLE = "ANTHROPIC_PROXY"

# Selectors ChatAnthropic accepts but this client never forwards, so a caller
# supplying one would get routing that differs from the resolved identity.
_UNSUPPORTED_ROUTING_KWARGS = frozenset(
    {"anthropic_api_url", "anthropic_proxy", "default_headers"}
)


@dataclass(frozen=True)
class _AnthropicSettings:
    routing: LLMRoutingResolution
    client_base_url: str | None


def _resolve_anthropic_settings(
    model: str,
    base_url: str | None,
    environment: Mapping[str, str],
    kwargs: Mapping[str, Any],
) -> _AnthropicSettings:
    unsupported = sorted(_UNSUPPORTED_ROUTING_KWARGS.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise UnsupportedRoutingSelectorError(
            f"Anthropic routing cannot resolve unsupported selector(s): {names}"
        )

    gateway = resolve_gateway_routing(
        environment,
        provider_path="anthropic",
        credential_kind="api_key",
        explicit_endpoint=base_url,
        explicit_credential=kwargs.get("api_key"),
        endpoint_environment_variables=_ENDPOINT_ENVIRONMENT_VARIABLES,
        credential_environment_variables=_CREDENTIAL_ENVIRONMENT_VARIABLES,
        default_endpoint=_SDK_DEFAULT_ENDPOINT,
    )
    ensure_non_secret_endpoint(gateway.endpoint.value, "Anthropic endpoint")

    routing = LLMRoutingResolution(
        provider="anthropic",
        model=model,
        endpoint=gateway.endpoint,
        protocol=ResolvedRoutingValue("anthropic_messages", "provider_default"),
        credential=gateway.credential,
        routing_environment_variables=(
            *_ENDPOINT_ENVIRONMENT_VARIABLES,
            _PROXY_ENVIRONMENT_VARIABLE,
            GATEWAY_ENDPOINT_ENVIRONMENT_VARIABLE,
        ),
    )
    # Only a caller-supplied URL reaches ChatAnthropic: passing a resolved
    # default instead would suppress the SDK's own gateway precedence.
    return _AnthropicSettings(routing, base_url or None)


def resolve_anthropic_routing(
    model: str,
    base_url: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> LLMRoutingResolution:
    """Resolve Anthropic routing without exposing credential values."""
    selected_environment = os.environ if environment is None else environment
    return _resolve_anthropic_settings(
        model, base_url, selected_environment, kwargs
    ).routing


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        settings = _resolve_anthropic_settings(
            self.model, self.base_url, os.environ, self.kwargs
        )
        if settings.client_base_url:
            llm_kwargs["base_url"] = settings.client_base_url

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
