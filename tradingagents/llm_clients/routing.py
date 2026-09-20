"""Shared, non-secret description of effective LLM routing.

Provider modules own resolution.  These small value objects give callers a
stable shape without importing any provider SDKs from the factory itself.
"""

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
    routing_environment_variables: tuple[str, ...] = ()
