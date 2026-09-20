"""Shared, non-secret description of effective LLM routing.

Provider modules own resolution.  These small value objects give callers a
stable shape without importing any provider SDKs from the factory itself.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlparse


class RoutingResolutionError(ValueError):
    """Raised when effective provider routing cannot be determined safely."""


class UnsupportedRoutingSelectorError(RoutingResolutionError):
    """Raised when a selector is accepted by an SDK but not owned by TA."""


_CREDENTIAL_QUERY_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "key",
        "password",
        "sig",
        "signature",
        "token",
    }
)


def ensure_non_secret_endpoint(endpoint: str, selector: str) -> None:
    """Reject endpoint forms that would put credentials into saved identity."""
    parsed = urlparse(endpoint)
    query_names = {
        name.lower().replace("-", "_") for name, _value in parse_qsl(parsed.query)
    }
    if parsed.username or parsed.password or query_names.intersection(_CREDENTIAL_QUERY_NAMES):
        raise RoutingResolutionError(
            f"{selector} contains credential-bearing URL components and cannot be "
            "used as non-secret routing identity"
        )


@dataclass(frozen=True)
class ResolvedRoutingValue:
    """One non-secret routing value and the rule that selected it."""

    value: str
    source: str
    environment_variable: str | None = None


@dataclass(frozen=True)
class CredentialPresence:
    """Credential availability without the credential value itself."""

    present: bool
    kind: str
    source: str
    environment_variable: str | None = None


@dataclass(frozen=True)
class LLMRoutingResolution:
    """Effective non-secret routing identity for one configured model."""

    provider: str
    model: str
    endpoint: ResolvedRoutingValue
    protocol: ResolvedRoutingValue
    credential: CredentialPresence
    deployment: ResolvedRoutingValue | None = None
    api_version: ResolvedRoutingValue | None = None
    # Regional selector, where the family has one: a Bedrock AWS region today.
    region: ResolvedRoutingValue | None = None
    routing_environment_variables: tuple[str, ...] = ()


# The LangSmith gateway redirects a provider's base URL purely from the
# environment.  langchain-core owns the rules; its helper reads ``os.environ``
# directly, which resolution against an explicit mapping must not do, so the
# same precedence is mirrored here for the families whose installed SDK wires
# it up (Bedrock, Anthropic, Google).
GATEWAY_ENDPOINT_ENVIRONMENT_VARIABLE = "LANGSMITH_GATEWAY"
_GATEWAY_CREDENTIAL_ENVIRONMENT_VARIABLE = "LANGSMITH_GATEWAY_API_KEY"
_LANGSMITH_CREDENTIAL_ENVIRONMENT_VARIABLE = "LANGSMITH_API_KEY"
_GATEWAY_DEFAULT_BASE_URL = "https://gateway.smith.langchain.com"
_GATEWAY_TRUE_VALUES = ("true", "1", "yes")
_GATEWAY_FALSE_VALUES = ("false", "0", "no")


@dataclass(frozen=True)
class GatewayRouting:
    """Endpoint and credential presence after LangSmith gateway precedence."""

    endpoint: ResolvedRoutingValue | None
    credential: CredentialPresence
    from_gateway: bool


def first_environment_entry(
    environment: Mapping[str, str], names: Sequence[str]
) -> tuple[str, str] | None:
    """First ``(name, value)`` with a non-empty value, mirroring the SDK order."""
    for name in names:
        value = environment.get(name)
        if value:
            return name, value
    return None


def gateway_endpoint_url(
    environment: Mapping[str, str], provider_path: str
) -> str | None:
    """Provider base URL on the LangSmith gateway, or None when it is disabled.

    ``LANGSMITH_GATEWAY`` is either a boolean-ish string selecting the default
    gateway host or an explicit gateway base URL.
    """
    raw = environment.get(GATEWAY_ENDPOINT_ENVIRONMENT_VARIABLE)
    if not raw or raw.lower() in _GATEWAY_FALSE_VALUES:
        return None
    base = (
        _GATEWAY_DEFAULT_BASE_URL
        if raw.lower() in _GATEWAY_TRUE_VALUES
        else raw.rstrip("/")
    )
    return f"{base}/{provider_path}"


def resolve_gateway_routing(
    environment: Mapping[str, str],
    *,
    provider_path: str,
    credential_kind: str,
    explicit_endpoint: str | None = None,
    explicit_credential: str | None = None,
    endpoint_environment_variables: Sequence[str] = (),
    credential_environment_variables: Sequence[str] = (),
    default_endpoint: str | None = None,
) -> GatewayRouting:
    """Resolve an endpoint and credential source the way the installed SDK does.

    Endpoint precedence is explicit value, then the provider's own environment
    names, then the gateway, then the SDK default.  The credential follows the
    SDK's provenance flip: the gateway key wins only when the endpoint itself
    came from the gateway, otherwise the provider key does.
    """
    gateway_url = gateway_endpoint_url(environment, provider_path)

    from_gateway = False
    endpoint: ResolvedRoutingValue | None
    provider_endpoint = first_environment_entry(
        environment, endpoint_environment_variables
    )
    if explicit_endpoint:
        endpoint = ResolvedRoutingValue(explicit_endpoint, "explicit")
    elif provider_endpoint is not None:
        endpoint = ResolvedRoutingValue(
            provider_endpoint[1], "environment", provider_endpoint[0]
        )
    elif gateway_url is not None:
        endpoint = ResolvedRoutingValue(
            gateway_url, "gateway", GATEWAY_ENDPOINT_ENVIRONMENT_VARIABLE
        )
        from_gateway = True
    elif default_endpoint:
        endpoint = ResolvedRoutingValue(default_endpoint, "sdk_default")
    else:
        endpoint = None

    if explicit_credential:
        return GatewayRouting(
            endpoint, CredentialPresence(True, credential_kind, "explicit"), from_gateway
        )

    gateway_credential = None
    if gateway_url is not None:
        gateway_credential = first_environment_entry(
            environment, (_GATEWAY_CREDENTIAL_ENVIRONMENT_VARIABLE,)
        )
    if gateway_credential is None and from_gateway:
        gateway_credential = first_environment_entry(
            environment, (_LANGSMITH_CREDENTIAL_ENVIRONMENT_VARIABLE,)
        )
    provider_credential = first_environment_entry(
        environment, credential_environment_variables
    )
    chosen = (
        (gateway_credential or provider_credential)
        if from_gateway
        else (provider_credential or gateway_credential)
    )
    if chosen is None:
        credential = CredentialPresence(
            False,
            credential_kind,
            "missing",
            credential_environment_variables[0]
            if credential_environment_variables
            else None,
        )
    else:
        credential = CredentialPresence(True, credential_kind, "environment", chosen[0])
    return GatewayRouting(endpoint, credential, from_gateway)
