"""An exhausted required-data vendor chain must stay typed through the tools.

The router raising ``VendorChainRateLimitError`` is only half the contract: a
tool runs inside a ``ToolNode``, and a node that turned the exception into an
error ``ToolMessage`` would leave every caller parsing prose to tell "all
vendors are throttled, retry later" from "this symbol has no data". These tests
drive the production tool nodes from ``TradingAgentsGraph._create_tool_nodes``
and a compiled graph, so the assertion is about the installed LangGraph, not a
stand-in.

Observed behavior of the installed SDK (langgraph-prebuilt 1.1.0): ``ToolNode``
defaults to ``_default_handle_tool_errors``, which returns a message only for
``ToolInvocationError`` (bad tool arguments) and re-raises everything else. The
tests below pin both halves of that policy, so a future default that swallows
tool exceptions fails here rather than silently downgrading the throttle.
Nothing here calls a provider or the network.
"""

from __future__ import annotations

import copy
from unittest import mock

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorChainRateLimitError,
    VendorRateLimitError,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph

TICKER = "AAPL"
TRADE_DATE = "2026-05-08"


class _State(MessagesState):
    trade_date: str


@pytest.fixture(autouse=True)
def _clean_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _tool_node(name: str):
    """A production tool node. ``_create_tool_nodes`` does not use ``self``."""
    return TradingAgentsGraph._create_tool_nodes(None)[name]


def _call(name: str, args: dict) -> dict:
    return {
        "messages": [AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c1"}])],
        "trade_date": TRADE_DATE,
    }


def _through_graph(node, payload: dict):
    """The same node, reached the way a run reaches it: through a compiled graph."""
    graph = StateGraph(_State)
    graph.add_node("tools", node)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile().invoke(payload)


def _vendors(method: str, vendors: dict):
    return mock.patch.dict(interface.VENDOR_METHODS, {method: vendors}, clear=False)


def _raises(exc):
    def impl(*a, **k):
        raise exc
    return impl


def _throttled(message: str, retry_after: float | None = None):
    return _raises(VendorRateLimitError(message, retry_after=retry_after))


def _price_args() -> dict:
    return {"symbol": TICKER, "start_date": "2026-05-01", "end_date": TRADE_DATE}


@pytest.mark.unit
def test_the_production_tool_nodes_are_the_installed_toolnode():
    # The rest of this file asserts an error policy that belongs to the
    # installed ToolNode; if TA ever wrapped its tools in something else, those
    # assertions would be about the wrapper instead.
    for name in ("market", "social", "news", "fundamentals"):
        assert isinstance(_tool_node(name), ToolNode)


@pytest.mark.unit
def test_a_sole_throttled_required_vendor_raises_through_the_graph():
    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    with _vendors("get_stock_data", {"yfinance": _throttled("Yahoo Finance is throttling")}), \
            pytest.raises(VendorChainRateLimitError):
        _through_graph(_tool_node("market"), _call("get_stock_data", _price_args()))


@pytest.mark.unit
def test_an_exhausted_required_chain_keeps_its_evidence_through_the_graph():
    set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})
    with _vendors(
        "get_stock_data",
        {
            "alpha_vantage": _throttled("Alpha Vantage rate limit exceeded"),
            "yfinance": _throttled("Yahoo Finance is throttling", retry_after=45.0),
        },
    ), pytest.raises(VendorChainRateLimitError) as caught:
        _through_graph(_tool_node("market"), _call("get_stock_data", _price_args()))

    # The evidence a caller schedules a retry on, intact after the tool node.
    assert caught.value.vendors == ["alpha_vantage", "yfinance"]
    assert caught.value.retry_after == 45.0
    assert isinstance(caught.value.__cause__, VendorRateLimitError)


@pytest.mark.unit
def test_the_throttle_is_not_rendered_into_a_tool_message():
    # The failure mode this guards: an error ToolMessage looks like a completed
    # step, so the run continues and the only trace of the throttle is prose.
    set_config({"data_vendors": {"fundamental_data": "yfinance"}})
    with _vendors("get_fundamentals", {"yfinance": _throttled("Yahoo Finance is throttling")}):
        try:
            result = _through_graph(
                _tool_node("fundamentals"),
                _call("get_fundamentals", {"ticker": TICKER, "curr_date": TRADE_DATE}),
            )
        except VendorChainRateLimitError:
            return
    pytest.fail(f"the throttle was swallowed into state: {result['messages'][-1]}")


@pytest.mark.unit
def test_a_successful_fallback_returns_data_through_the_tool_node():
    # A throttled vendor earlier in the chain is diagnostic activity: it must
    # not fail a call another configured vendor can serve.
    set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})
    with _vendors(
        "get_stock_data",
        {"alpha_vantage": _throttled("slow down"), "yfinance": lambda *a, **k: "OHLCV_ROWS"},
    ):
        state = _through_graph(_tool_node("market"), _call("get_stock_data", _price_args()))
    message = state["messages"][-1]
    assert message.content == "OHLCV_ROWS"
    assert message.status != "error"


@pytest.mark.unit
def test_no_data_still_reaches_the_model_as_an_ordinary_result():
    # No-data is a verdict about the instrument the analyst must read and
    # report; raising it would end a run that has nothing to retry.
    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    with _vendors(
        "get_stock_data", {"yfinance": _raises(NoMarketDataError(TICKER, TICKER, "no rows"))}
    ):
        state = _through_graph(_tool_node("market"), _call("get_stock_data", _price_args()))
    assert "NO_DATA_AVAILABLE" in state["messages"][-1].content


@pytest.mark.unit
def test_optional_enrichment_exhaustion_does_not_escape_the_tool_node():
    # Macro context is optional: an entirely throttled optional chain degrades
    # to its sentinel rather than failing the analysis or earning a paid retry.
    set_config({"data_vendors": {"macro_data": "fred"}})
    with _vendors("get_macro_indicators", {"fred": _throttled("FRED is throttling")}):
        state = _through_graph(
            _tool_node("news"),
            _call("get_macro_indicators", {"indicator": "cpi", "curr_date": TRADE_DATE}),
        )
    message = state["messages"][-1]
    assert "DATA_UNAVAILABLE" in message.content
    assert message.status != "error"


@pytest.mark.unit
def test_invalid_tool_arguments_keep_their_existing_error_message():
    # The other half of the installed policy, pinned so the throttle fix can't
    # be read as license to make every tool failure end the run.
    state = _through_graph(_tool_node("market"), _call("get_stock_data", {"symbol": TICKER}))
    message = state["messages"][-1]
    assert message.status == "error"
    assert "start_date" in message.content
