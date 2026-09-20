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
_GATEWAY_DEFAULT_BASE_URL = "https://gateway.smith.langchain.com"
_GATEWAY_TRUE_VALUES = ("true", "1", "yes")
_GATEWAY_FALSE_VALUES = ("false", "0", "no")


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
