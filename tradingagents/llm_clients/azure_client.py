import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_openai import AzureChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .routing import (
    CredentialPresence,
    LLMRoutingResolution,
    ResolvedRoutingValue,
    RoutingResolutionError,
    UnsupportedRoutingSelectorError,
    ensure_non_secret_endpoint,
)

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "reasoning_effort", "temperature",
    "callbacks", "http_client", "http_async_client",
)


class NormalizedAzureChatOpenAI(AzureChatOpenAI):
    """AzureChatOpenAI with normalized content output."""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


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
class _AzureSettings:
    routing: LLMRoutingResolution
    api_key: str | None
    azure_ad_token: str | None


def _resolve_azure_settings(
    model: str,
    environment: Mapping[str, str],
    kwargs: Mapping[str, Any],
) -> _AzureSettings:
    unsupported = sorted(_UNSUPPORTED_ROUTING_KWARGS.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise UnsupportedRoutingSelectorError(
            f"Azure routing cannot resolve unsupported selector(s): {names}"
        )

    endpoint_value = environment.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint_value:
        raise RoutingResolutionError(
            "Azure routing requires AZURE_OPENAI_ENDPOINT in the selected environment"
        )
    endpoint = ResolvedRoutingValue(
        endpoint_value, "environment", "AZURE_OPENAI_ENDPOINT"
    )
    ensure_non_secret_endpoint(endpoint.value, "Azure endpoint")

    deployment_value = environment.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    deployment = ResolvedRoutingValue(
        deployment_value or model,
        "environment" if deployment_value else "model_default",
        "AZURE_OPENAI_DEPLOYMENT_NAME" if deployment_value else None,
    )

    api_version_value = environment.get("OPENAI_API_VERSION")
    if not api_version_value:
        raise RoutingResolutionError(
            "Azure routing requires OPENAI_API_VERSION in the selected environment"
        )
    api_version = ResolvedRoutingValue(
        api_version_value, "environment", "OPENAI_API_VERSION"
    )

    explicit_api_key = kwargs.get("api_key")
    environment_api_key = environment.get("AZURE_OPENAI_API_KEY")
    azure_ad_token = environment.get("AZURE_OPENAI_AD_TOKEN")
    # Match AzureChatOpenAI's existing precedence: an environment AD token
    # suppresses an API key, including an explicitly supplied one.
    if azure_ad_token:
        credential = CredentialPresence(
            True, "azure_ad_token", "environment", "AZURE_OPENAI_AD_TOKEN"
        )
    elif explicit_api_key:
        credential = CredentialPresence(True, "api_key", "explicit")
    elif environment_api_key:
        credential = CredentialPresence(
            True, "api_key", "environment", "AZURE_OPENAI_API_KEY"
        )
    else:
        credential = CredentialPresence(
            False, "api_key_or_azure_ad_token", "missing"
        )

    routing = LLMRoutingResolution(
        provider="azure",
        model=model,
        endpoint=endpoint,
        deployment=deployment,
        api_version=api_version,
        protocol=ResolvedRoutingValue("azure_chat_completions", "provider_default"),
        credential=credential,
        routing_environment_variables=(
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "OPENAI_API_VERSION",
        ),
    )
    return _AzureSettings(routing, explicit_api_key or environment_api_key, azure_ad_token)


def resolve_azure_routing(
    model: str,
    base_url: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> LLMRoutingResolution:
    """Resolve Azure routing without exposing credential values.

    ``base_url`` remains accepted for parity with ``create_llm_client``.  The
    Azure client has historically used ``AZURE_OPENAI_ENDPOINT`` instead, so
    that argument intentionally does not alter its routing precedence.
    """
    del base_url
    selected_environment = os.environ if environment is None else environment
    return _resolve_azure_settings(model, selected_environment, kwargs).routing


class AzureOpenAIClient(BaseLLMClient):
    """Client for Azure OpenAI deployments.

    Requires environment variables:
        AZURE_OPENAI_API_KEY: API key
        AZURE_OPENAI_ENDPOINT: Endpoint URL (e.g. https://<resource>.openai.azure.com/)
        AZURE_OPENAI_DEPLOYMENT_NAME: Deployment name
        OPENAI_API_VERSION: API version (e.g. 2025-03-01-preview)
    """

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured AzureChatOpenAI instance."""
        self.warn_if_unknown_model()

        settings = _resolve_azure_settings(self.model, os.environ, self.kwargs)
        llm_kwargs = {
            "model": self.model,
            "azure_endpoint": settings.routing.endpoint.value,
            "azure_deployment": settings.routing.deployment.value,
            "api_version": settings.routing.api_version.value,
        }
        if settings.api_key:
            llm_kwargs["api_key"] = settings.api_key
        if settings.azure_ad_token:
            llm_kwargs["azure_ad_token"] = settings.azure_ad_token

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                if key == "api_key":
                    continue
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedAzureChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Azure accepts any deployed model name."""
        return True
