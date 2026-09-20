"""Amazon Bedrock — first-class native client via the optional langchain-aws extra.

Auth uses the AWS credential chain (no single key env); the model is a Bedrock
model ID / inference profile ID; langchain-aws is imported lazily with a clear
install hint when the [bedrock] extra is absent.
"""
import os
import subprocess
import sys

import pytest

from tradingagents.llm_clients import resolve_llm_routing
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.validators import validate_model

_ROUTING_ENV_VARS = (
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
    "AWS_BEARER_TOKEN_BEDROCK",
    "LANGSMITH_GATEWAY",
    "LANGSMITH_GATEWAY_API_KEY",
    "LANGSMITH_API_KEY",
)


@pytest.fixture()
def clean_routing_env(monkeypatch):
    """Drop ambient AWS/gateway routing so precedence assertions are exact."""
    for name in _ROUTING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
def test_factory_routes_bedrock():
    client = create_llm_client("bedrock", "us.anthropic.claude-opus-4-8-v1:0")
    assert type(client).__name__ == "BedrockClient"


@pytest.mark.unit
def test_bedrock_any_model_and_no_key_env():
    assert validate_model("bedrock", "any.model-id:0") is True
    # Bedrock uses the AWS credential chain, so there is no single key env.
    assert get_api_key_env("bedrock") is None


@pytest.mark.unit
def test_helpful_error_when_langchain_aws_absent(monkeypatch):
    import tradingagents.llm_clients.bedrock_client as bc
    monkeypatch.setattr(bc, "_BEDROCK_CLASS", None)
    monkeypatch.setitem(sys.modules, "langchain_aws", None)  # force ImportError on import
    with pytest.raises(ImportError, match=r"bedrock"):
        create_llm_client("bedrock", "m").get_llm()


def _capture_kwargs(monkeypatch):
    """Stub _bedrock_class so the constructor kwargs are testable without the
    optional langchain-aws extra installed."""
    import tradingagents.llm_clients.bedrock_client as bc
    captured = {}

    class _FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(bc, "_bedrock_class", lambda: _FakeChat)
    return captured


@pytest.mark.unit
def test_bearer_token_passed_as_api_key(monkeypatch):
    # #1103: a Bedrock API key authenticates without AWS access keys.
    captured = _capture_kwargs(monkeypatch)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bt-secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    create_llm_client("bedrock", "us.anthropic.claude-opus-4-8-v1:0").get_llm()
    assert captured["api_key"] == "bt-secret"
    assert captured["region_name"] == "us-east-1"


@pytest.mark.unit
def test_no_bearer_token_omits_api_key(monkeypatch):
    # Without a token, fall back to the AWS credential chain (no api_key kwarg).
    captured = _capture_kwargs(monkeypatch)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    create_llm_client("bedrock", "us.anthropic.claude-opus-4-8-v1:0").get_llm()
    assert "api_key" not in captured


@pytest.mark.unit
def test_construction_when_extra_installed(monkeypatch):
    pytest.importorskip("langchain_aws")
    import tradingagents.llm_clients.bedrock_client as bc
    monkeypatch.setattr(bc, "_BEDROCK_CLASS", None)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    llm = create_llm_client("bedrock", "us.anthropic.claude-sonnet-5").get_llm()
    assert type(llm).__name__ == "NormalizedChatBedrockConverse"
    assert llm.region_name == "eu-west-1"


@pytest.mark.unit
@pytest.mark.parametrize(
    "environment,region,source,variable",
    [
        (
            {"AWS_REGION": "eu-central-1", "AWS_DEFAULT_REGION": "us-east-2"},
            "eu-central-1",
            "environment",
            "AWS_REGION",
        ),
        (
            {"AWS_DEFAULT_REGION": "us-east-2"},
            "us-east-2",
            "environment",
            "AWS_DEFAULT_REGION",
        ),
        ({}, "us-west-2", "client_default", None),
    ],
)
def test_region_precedence_reaches_the_regional_endpoint(
    environment, region, source, variable
):
    resolved = resolve_llm_routing(
        "bedrock", "us.anthropic.claude-sonnet-5", environment=environment
    )

    assert resolved.region.value == region
    assert resolved.region.source == source
    assert resolved.region.environment_variable == variable
    # An unset endpoint override is not proof of a fixed destination: botocore
    # still resolves the regional host from the region we pass.
    assert resolved.endpoint.value == f"https://bedrock-runtime.{region}.amazonaws.com"
    assert resolved.endpoint.source == "sdk_default"
    assert resolved.protocol.value == "bedrock_converse"


@pytest.mark.unit
@pytest.mark.parametrize(
    "environment,endpoint,variable",
    [
        (
            {
                "AWS_ENDPOINT_URL": "https://global.example",
                "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "https://service.example",
            },
            "https://service.example",
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        ),
        (
            {"AWS_ENDPOINT_URL": "https://global.example"},
            "https://global.example",
            "AWS_ENDPOINT_URL",
        ),
    ],
)
def test_endpoint_environment_overrides_the_regional_default(
    environment, endpoint, variable
):
    resolved = resolve_llm_routing("bedrock", "model-id", environment=environment)

    assert resolved.endpoint.value == endpoint
    assert resolved.endpoint.source == "environment"
    assert resolved.endpoint.environment_variable == variable


@pytest.mark.unit
def test_gateway_redirect_wins_over_aws_endpoint_variables():
    # langchain-aws applies the gateway before botocore reads its own variables.
    resolved = resolve_llm_routing(
        "bedrock",
        "model-id",
        environment={
            "LANGSMITH_GATEWAY": "true",
            "AWS_ENDPOINT_URL": "https://global.example",
        },
    )

    assert resolved.endpoint.value == "https://gateway.smith.langchain.com/bedrock"
    assert resolved.endpoint.source == "gateway"
    assert "LANGSMITH_GATEWAY" in resolved.routing_environment_variables


@pytest.mark.unit
def test_bearer_token_is_reported_as_presence_only():
    resolved = resolve_llm_routing(
        "bedrock",
        "model-id",
        environment={"AWS_BEARER_TOKEN_BEDROCK": "bt-secret"},
    )

    assert resolved.credential.present is True
    assert resolved.credential.kind == "bedrock_api_key"
    assert resolved.credential.environment_variable == "AWS_BEARER_TOKEN_BEDROCK"
    assert "bt-secret" not in repr(resolved)


@pytest.mark.unit
def test_credential_chain_presence_is_not_asserted():
    # Without a bearer token the AWS chain (profile, IAM role, ...) decides, and
    # an environment mapping cannot answer that.
    resolved = resolve_llm_routing("bedrock", "model-id", environment={})

    assert resolved.credential.kind == "aws_credential_chain"
    assert resolved.credential.source == "credential_chain"
    assert resolved.credential.present is False


@pytest.mark.unit
def test_client_and_resolution_share_the_same_region(monkeypatch, clean_routing_env):
    captured = _capture_kwargs(monkeypatch)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")

    resolved = resolve_llm_routing(
        "bedrock", "us.anthropic.claude-sonnet-5", environment=dict(os.environ)
    )
    create_llm_client("bedrock", "us.anthropic.claude-sonnet-5").get_llm()

    assert captured["region_name"] == resolved.region.value == "ap-southeast-2"


@pytest.mark.unit
def test_gateway_endpoint_matches_the_installed_sdk(monkeypatch, clean_routing_env):
    # Fails loudly if langchain-aws changes the rule this resolution mirrors.
    pytest.importorskip("langchain_aws")
    import tradingagents.llm_clients.bedrock_client as bc
    monkeypatch.setattr(bc, "_BEDROCK_CLASS", None)
    monkeypatch.setenv("LANGSMITH_GATEWAY", "https://gw.example")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")

    resolved = resolve_llm_routing(
        "bedrock", "us.anthropic.claude-sonnet-5", environment=dict(os.environ)
    )
    llm = create_llm_client("bedrock", "us.anthropic.claude-sonnet-5").get_llm()

    assert llm.endpoint_url == resolved.endpoint.value == "https://gw.example/bedrock"


@pytest.mark.unit
def test_selectors_the_client_never_forwards_are_not_guessed():
    with pytest.raises(ValueError, match="region_name"):
        resolve_llm_routing(
            "bedrock", "model-id", environment={}, region_name="us-east-1"
        )


@pytest.mark.unit
def test_backend_url_does_not_alter_bedrock_routing():
    # BedrockClient has never forwarded backend_url, so routing must not claim it.
    resolved = resolve_llm_routing(
        "bedrock",
        "model-id",
        "https://ignored.example/v1",
        environment={"AWS_REGION": "us-east-1"},
    )

    assert resolved.endpoint.value == "https://bedrock-runtime.us-east-1.amazonaws.com"


@pytest.mark.unit
def test_routing_resolution_does_not_import_the_optional_sdk():
    # A missing optional SDK must only affect a provider that is actually used.
    code = (
        "import sys; from tradingagents.llm_clients import resolve_llm_routing; "
        "resolve_llm_routing('bedrock', 'model-id', environment={}); "
        "assert 'langchain_aws' not in sys.modules"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)
