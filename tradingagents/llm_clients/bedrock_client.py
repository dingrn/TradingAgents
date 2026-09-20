import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .base_client import BaseLLMClient, normalize_content
from .routing import (
    GATEWAY_ENDPOINT_ENVIRONMENT_VARIABLE,
    CredentialPresence,
    LLMRoutingResolution,
    ResolvedRoutingValue,
    UnsupportedRoutingSelectorError,
    ensure_non_secret_endpoint,
    first_environment_entry,
    gateway_endpoint_url,
)
from .validators import validate_model

# Bedrock has no global default region; us-west-2 hosts the broadest model set.
_DEFAULT_REGION = "us-west-2"
_BEDROCK_CLASS = None

_REGION_ENVIRONMENT_VARIABLES = ("AWS_REGION", "AWS_DEFAULT_REGION")
# botocore resolves the endpoint when langchain-aws passes none: a
# service-specific override first, then the global one.
_ENDPOINT_ENVIRONMENT_VARIABLES = (
    "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
    "AWS_ENDPOINT_URL",
)
# botocore also reads a ``services`` endpoint_url out of the shared config file,
# which these names select. That file is outside an environment mapping, so the
# names are reported as routing inputs a caller has to pin or remove rather than
# resolved here.
_CONFIGURED_ENDPOINT_ENVIRONMENT_VARIABLES = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_CONFIG_FILE",
)
_BEARER_TOKEN_ENVIRONMENT_VARIABLE = "AWS_BEARER_TOKEN_BEDROCK"
_GATEWAY_CREDENTIAL_ENVIRONMENT_VARIABLE = "LANGSMITH_GATEWAY_API_KEY"

# Selectors ChatBedrockConverse accepts but this client never forwards, so a
# caller supplying one would get routing that differs from the resolved identity.
_UNSUPPORTED_ROUTING_KWARGS = frozenset(
    {
        "bedrock_client",
        "client",
        "config",
        "credentials_profile_name",
        "endpoint_url",
        "provider",
        "region_name",
    }
)


@dataclass(frozen=True)
class _BedrockSettings:
    routing: LLMRoutingResolution
    bearer_token: str | None


def _resolve_bedrock_settings(
    model: str,
    environment: Mapping[str, str],
    kwargs: Mapping[str, Any],
) -> _BedrockSettings:
    unsupported = sorted(_UNSUPPORTED_ROUTING_KWARGS.intersection(kwargs))
    if unsupported:
        names = ", ".join(unsupported)
        raise UnsupportedRoutingSelectorError(
            f"Bedrock routing cannot resolve unsupported selector(s): {names}"
        )

    region_entry = first_environment_entry(environment, _REGION_ENVIRONMENT_VARIABLES)
    if region_entry is None:
        region = ResolvedRoutingValue(_DEFAULT_REGION, "client_default")
    else:
        region = ResolvedRoutingValue(region_entry[1], "environment", region_entry[0])

    # langchain-aws applies the gateway before botocore sees the request, so an
    # enabled gateway wins over the AWS endpoint variables.
    gateway_url = gateway_endpoint_url(environment, "bedrock")
    endpoint_entry = first_environment_entry(
        environment, _ENDPOINT_ENVIRONMENT_VARIABLES
    )
    if gateway_url is not None:
        endpoint = ResolvedRoutingValue(
            gateway_url, "gateway", GATEWAY_ENDPOINT_ENVIRONMENT_VARIABLE
        )
    elif endpoint_entry is not None:
        endpoint = ResolvedRoutingValue(
            endpoint_entry[1], "environment", endpoint_entry[0]
        )
    else:
        endpoint = ResolvedRoutingValue(
            f"https://bedrock-runtime.{region.value}.amazonaws.com", "sdk_default"
        )
    ensure_non_secret_endpoint(endpoint.value, "Bedrock endpoint")

    bearer_token = environment.get(_BEARER_TOKEN_ENVIRONMENT_VARIABLE)
    gateway_token = (
        environment.get(_GATEWAY_CREDENTIAL_ENVIRONMENT_VARIABLE)
        if gateway_url is not None
        else None
    )
    if bearer_token:
        credential = CredentialPresence(
            True, "bedrock_api_key", "environment", _BEARER_TOKEN_ENVIRONMENT_VARIABLE
        )
    elif gateway_token:
        credential = CredentialPresence(
            True,
            "bedrock_api_key",
            "environment",
            _GATEWAY_CREDENTIAL_ENVIRONMENT_VARIABLE,
        )
    else:
        # Without a bearer token langchain-aws falls back to the AWS credential
        # chain (env keys, shared profile, IAM role), which an environment
        # mapping alone cannot answer; presence stays unasserted.
        credential = CredentialPresence(
            False, "aws_credential_chain", "credential_chain"
        )

    routing = LLMRoutingResolution(
        provider="bedrock",
        model=model,
        endpoint=endpoint,
        protocol=ResolvedRoutingValue("bedrock_converse", "provider_default"),
        credential=credential,
        region=region,
        routing_environment_variables=(
            *_REGION_ENVIRONMENT_VARIABLES,
            *_ENDPOINT_ENVIRONMENT_VARIABLES,
            *_CONFIGURED_ENDPOINT_ENVIRONMENT_VARIABLES,
            GATEWAY_ENDPOINT_ENVIRONMENT_VARIABLE,
        ),
    )
    return _BedrockSettings(routing, bearer_token or None)


def resolve_bedrock_routing(
    model: str,
    base_url: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> LLMRoutingResolution:
    """Resolve Bedrock routing without exposing credential values.

    ``base_url`` is accepted for parity with ``create_llm_client``.  Like
    ``BedrockClient`` itself, which has never forwarded it, it does not take part
    in Bedrock's routing precedence.
    """
    del base_url
    selected_environment = os.environ if environment is None else environment
    return _resolve_bedrock_settings(model, selected_environment, kwargs).routing


def _bedrock_class():
    """Lazily import langchain-aws (the optional ``[bedrock]`` extra) and return a
    ChatBedrockConverse subclass with normalized content output.

    Imported on demand so the optional dependency (and boto3) isn't required by
    the rest of the package; cached after the first call.
    """
    global _BEDROCK_CLASS
    if _BEDROCK_CLASS is not None:
        return _BEDROCK_CLASS

    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as exc:
        raise ImportError(
            "AWS Bedrock support requires the optional 'langchain-aws' dependency. "
            'Install it with: pip install "tradingagents[bedrock]"'
        ) from exc

    class NormalizedChatBedrockConverse(ChatBedrockConverse):
        """ChatBedrockConverse with normalized (string) content output."""

        def invoke(self, input, config=None, **kwargs):
            return normalize_content(super().invoke(input, config, **kwargs))

    _BEDROCK_CLASS = NormalizedChatBedrockConverse
    return _BEDROCK_CLASS


class BedrockClient(BaseLLMClient):
    """Client for Amazon Bedrock via the Converse API (langchain-aws).

    Authentication is either a Bedrock API key (bearer token) via
    ``AWS_BEARER_TOKEN_BEDROCK`` — no AWS access keys required — or the standard
    AWS credential chain (env vars, ``~/.aws/credentials``, or an IAM role) with
    optional ``AWS_PROFILE``. Set ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` either
    way (the token carries no region). The model name is a Bedrock model ID or
    cross-region inference profile ID, e.g. ``us.anthropic.claude-opus-4-8-v1:0``.
    """

    def get_llm(self) -> Any:
        """Return a configured ChatBedrockConverse instance."""
        self.warn_if_unknown_model()
        chat_cls = _bedrock_class()

        settings = _resolve_bedrock_settings(self.model, os.environ, self.kwargs)
        llm_kwargs = {"model": self.model, "region_name": settings.routing.region.value}
        # A Bedrock API key authenticates without AWS access keys. Passing it as
        # api_key makes langchain-aws prefer bearer auth, so an ambient
        # AWS_PROFILE / SigV4 credentials can't override it (#1103).
        if settings.bearer_token:
            llm_kwargs["api_key"] = settings.bearer_token
        for key in ("temperature", "max_tokens", "max_retries", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Bedrock (any model ID accepted)."""
        return validate_model("bedrock", self.model)
