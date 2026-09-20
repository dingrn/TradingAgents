import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

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

_GEMINI_VERSION = re.compile(r"^gemini-(\d+)\.(\d+)")


def _accepts_minimal_thinking(model: str) -> bool:
    """Whether ``thinking_level="minimal"`` is accepted: numbered Flash models
    before 3.8. Pro, 3.8+ and version-less aliases (which move between
    generations) are treated as rejecting it."""
    model_lc = model.lower()
    match = _GEMINI_VERSION.match(model_lc)
    return bool(match) and "pro" not in model_lc and (
        (int(match.group(1)), int(match.group(2))) < (3, 8)
    )


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list of typed blocks.
    This normalizes to string for consistent downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


_CREDENTIAL_ENVIRONMENT_VARIABLES = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
_DEVELOPER_API_ENDPOINT = "https://generativelanguage.googleapis.com/"
_DEVELOPER_API_VERSION = "v1beta"
# ChatGoogleGenerativeAI picks its backend from this variable when no vertexai,
# credentials or project argument is supplied — and this client supplies none.
_BACKEND_ENVIRONMENT_VARIABLE = "GOOGLE_GENAI_USE_VERTEXAI"
_VERTEX_TRUE_VALUES = ("true", "1", "yes")
# Only consulted once Vertex AI is selected, which this client cannot resolve.
_VERTEX_ENVIRONMENT_VARIABLES = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")

# Selectors ChatGoogleGenerativeAI accepts but this client never forwards. Some
# of them (project, credentials, vertexai, location) would also switch the
# backend, so a caller supplying one would get routing that differs from the
# resolved identity.
_UNSUPPORTED_ROUTING_KWARGS = frozenset(
    {
        "additional_headers",
        "api_version",
        "client_args",
        "client_options",
        "credentials",
        "location",
        "project",
        "vertexai",
    }
)


@dataclass(frozen=True)
class _GoogleClientSettings:
    """The routing inputs this client hands to the SDK."""

    base_url: str | None
    google_api_key: str | None


def _resolve_google_client_settings(
    base_url: str | None, kwargs: Mapping[str, Any]
) -> _GoogleClientSettings:
    # Unified api_key maps to provider-specific google_api_key
    google_api_key = kwargs.get("api_key") or kwargs.get("google_api_key")
    return _GoogleClientSettings(base_url or None, google_api_key)


def _resolve_google_settings(
    model: str,
    base_url: str | None,
    environment: Mapping[str, str],
    kwargs: Mapping[str, Any],
) -> LLMRoutingResolution:
    unsupported = sorted(_UNSUPPORTED_ROUTING_KWARGS.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise UnsupportedRoutingSelectorError(
            f"Google routing cannot resolve unsupported selector(s): {names}"
        )

    if environment.get(_BACKEND_ENVIRONMENT_VARIABLE, "").lower() in _VERTEX_TRUE_VALUES:
        names = ", ".join(_VERTEX_ENVIRONMENT_VARIABLES)
        raise UnsupportedRoutingSelectorError(
            f"Google routing cannot be resolved while {_BACKEND_ENVIRONMENT_VARIABLE} "
            "selects Vertex AI: google-genai then derives the endpoint from "
            f"{names}, the API key and application-default credentials on disk, "
            "none of which this client owns or can read from the selected "
            "environment. Unset it to route through the Gemini Developer API."
        )

    client_settings = _resolve_google_client_settings(base_url, kwargs)
    gateway = resolve_gateway_routing(
        environment,
        provider_path="gemini",
        credential_kind="api_key",
        explicit_endpoint=client_settings.base_url,
        explicit_credential=client_settings.google_api_key,
        credential_environment_variables=_CREDENTIAL_ENVIRONMENT_VARIABLES,
        default_endpoint=_DEVELOPER_API_ENDPOINT,
    )
    ensure_non_secret_endpoint(gateway.endpoint.value, "Google endpoint")

    return LLMRoutingResolution(
        provider="google",
        model=model,
        endpoint=gateway.endpoint,
        protocol=ResolvedRoutingValue("gemini_developer", "sdk_default"),
        api_version=ResolvedRoutingValue(_DEVELOPER_API_VERSION, "sdk_default"),
        credential=gateway.credential,
        routing_environment_variables=(
            _BACKEND_ENVIRONMENT_VARIABLE,
            GATEWAY_ENDPOINT_ENVIRONMENT_VARIABLE,
        ),
    )


def resolve_google_routing(
    model: str,
    base_url: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> LLMRoutingResolution:
    """Resolve Google routing without exposing credential values."""
    selected_environment = os.environ if environment is None else environment
    return _resolve_google_settings(model, base_url, selected_environment, kwargs)


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Only the inputs this client owns are shared with routing resolution:
        # resolving the rest would reject a Vertex AI selection that the SDK
        # still accepts here.
        settings = _resolve_google_client_settings(self.base_url, self.kwargs)
        if settings.base_url:
            llm_kwargs["base_url"] = settings.base_url

        for key in ("timeout", "max_retries", "temperature", "max_output_tokens",
                    "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        if settings.google_api_key:
            llm_kwargs["google_api_key"] = settings.google_api_key

        # Gemini 3.x takes the string ``thinking_level`` (the integer
        # ``thinking_budget`` was for the now-retired 2.5 line). Pro, Gemini
        # 3.8+ and the -latest aliases reject "minimal" with a 400; "low" is
        # accepted everywhere, so it is the fallback.
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            if thinking_level == "minimal" and not _accepts_minimal_thinking(self.model):
                thinking_level = "low"
            llm_kwargs["thinking_level"] = thinking_level

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
