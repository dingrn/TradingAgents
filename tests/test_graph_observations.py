"""The generic graph observation adapter (design 11.2, first half of TA04).

These tests drive the real seams: an analyst/tool/message-clear path and the
research/risk nodes through the installed LangGraph and ``ToolNode``, with the
handler attached the way a caller attaches it — through the graph invocation
config. Nothing here calls a provider or the network, and nothing here changes
how ``propagate`` runs.

Observed behavior of the installed SDKs (langgraph 1.2.11 / langgraph-prebuilt
1.1.0 / langchain-core 1.6.3) that this relies on: LangGraph reports each node's
own run to the invocation config's callbacks, tagged ``graph:step:<n>`` with
``langgraph_node`` metadata, while the conditional router, chat model and tools
nested in that node are tagged ``seq:step:<n>``; a node's ``on_chain_end``
carries the state update it returned; and a handler that raises is absorbed by
langchain-core rather than ending the run.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import create_msg_delete
from tradingagents.graph.observation import (
    NODE_COMPLETED,
    NODE_FAILED,
    NODE_STARTED,
    REPORT_AVAILABLE,
    STAGE_COMPLETED,
    TOOL_COMPLETED,
    TOOL_STARTED,
    GraphObservation,
    GraphObservationHandler,
    emit,
)
from tradingagents.graph.propagation import Propagator

_REPORT = "Market report: the trend is up."
_TICKER = "AAPL"
_DATE = "2026-05-08"


@tool
def fake_indicators(symbol: str) -> str:
    """Offline stand-in for a market tool."""
    return f"indicators for {symbol}"


def _collect():
    """An observer that records every observation it is handed."""
    seen: list[GraphObservation] = []
    return seen, seen.append


def _initial_state():
    return Propagator().create_initial_state(
        _TICKER, _DATE, instrument_context=f"{_TICKER} (Apple Inc.)"
    )


def _analyst_workflow(*, crash_at_clear=False, model_callbacks=None):
    """The real analyst shape: agent -> tool node -> agent -> message clear."""
    model = GenericFakeChatModel(
        messages=iter([
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fake_indicators", "args": {"symbol": _TICKER}, "id": "call-1"}
                ],
            ),
            AIMessage(content=_REPORT),
        ]),
        callbacks=model_callbacks,
    )
    clear = create_msg_delete()

    def market_analyst(state):
        result = model.invoke(state["messages"])
        update = {"messages": [result]}
        if not result.tool_calls:
            update["market_report"] = result.content
        return update

    def route(state):
        return "tools_market" if state["messages"][-1].tool_calls else "Msg Clear Market"

    def clear_node(state):
        if crash_at_clear:
            raise RuntimeError("simulated crash before the analyst milestone")
        return clear(state)

    workflow = StateGraph(AgentState)
    workflow.add_node("Market Analyst", market_analyst)
    workflow.add_node("tools_market", ToolNode([fake_indicators]))
    workflow.add_node("Msg Clear Market", clear_node)
    workflow.add_edge(START, "Market Analyst")
    workflow.add_conditional_edges(
        "Market Analyst", route, ["tools_market", "Msg Clear Market"]
    )
    workflow.add_edge("tools_market", "Market Analyst")
    workflow.add_edge("Msg Clear Market", END)
    return workflow


def _run(workflow, handler):
    return workflow.compile().invoke(
        _initial_state(), config={"callbacks": [handler], "recursion_limit": 25}
    )


@pytest.mark.unit
def test_analyst_tool_and_clear_nodes_are_visible():
    seen, observer = _collect()

    result = _run(_analyst_workflow(), GraphObservationHandler(observer))

    assert result["market_report"] == _REPORT
    assert [(o.stage, o.node) for o in seen if o.type == NODE_STARTED] == [
        ("market", "Market Analyst"),
        ("market", "tools_market"),
        ("market", "Market Analyst"),
        ("market", "Msg Clear Market"),
    ]
    # The conditional router runs inside the analyst node; it is not a node and
    # must not be reported as one.
    assert {o.node for o in seen if o.node} == {
        "Market Analyst",
        "tools_market",
        "Msg Clear Market",
    }
    assert [o.type for o in seen].count(NODE_COMPLETED) == 4

    tools = [o for o in seen if o.type in (TOOL_STARTED, TOOL_COMPLETED)]
    assert [(o.type, o.node, o.payload["tool"]) for o in tools] == [
        (TOOL_STARTED, "tools_market", "fake_indicators"),
        (TOOL_COMPLETED, "tools_market", "fake_indicators"),
    ]


@pytest.mark.unit
def test_a_tool_round_does_not_look_like_a_finished_report():
    """The report is announced from its state field, the stage from the clear node."""
    seen, observer = _collect()

    _run(_analyst_workflow(), GraphObservationHandler(observer))

    reports = [o for o in seen if o.type == REPORT_AVAILABLE]
    assert len(reports) == 1  # the tool turn filled no report field
    assert reports[0].stage == "market"
    assert reports[0].payload == {"report_key": "market_report", "report": _REPORT}

    completed = [o for o in seen if o.type == STAGE_COMPLETED]
    assert [(o.stage, o.node) for o in completed] == [("market", "Msg Clear Market")]
    assert seen.index(reports[0]) < seen.index(completed[0])


@pytest.mark.unit
def test_a_failing_node_is_reported_and_not_left_open():
    seen, observer = _collect()

    with pytest.raises(RuntimeError):
        _run(_analyst_workflow(crash_at_clear=True), GraphObservationHandler(observer))

    failures = [o for o in seen if o.type == NODE_FAILED]
    assert [(o.stage, o.node) for o in failures] == [("market", "Msg Clear Market")]
    assert failures[0].payload["error_type"] == "RuntimeError"
    # The failed node did not also report completion.
    assert [o.node for o in seen if o.type == NODE_COMPLETED] == [
        "Market Analyst",
        "tools_market",
        "Market Analyst",
    ]


def _debate_workflow():
    """Research and risk nodes returning the debate state they really return."""

    def bull(state):
        return {"investment_debate_state": {"count": 1, "current_response": "Bull: buy"}}

    def bear(state):
        return {"investment_debate_state": {"count": 2, "current_response": "Bear: sell"}}

    def research_manager(state):
        return {
            "investment_debate_state": {"count": 2, "judge_decision": "hold the line"},
            "investment_plan": "PLAN",
        }

    def trader(state):
        return {"trader_investment_plan": "TRADE"}

    def conservative(state):
        return {"risk_debate_state": {"count": 1, "latest_speaker": "Conservative"}}

    def portfolio_manager(state):
        return {
            "risk_debate_state": {"count": 3, "latest_speaker": "Conservative"},
            "final_trade_decision": "DECISION",
        }

    workflow = StateGraph(AgentState)
    order = [
        ("Bull Researcher", bull),
        ("Bear Researcher", bear),
        ("Research Manager", research_manager),
        ("Trader", trader),
        ("Conservative Analyst", conservative),
        ("Portfolio Manager", portfolio_manager),
    ]
    for name, fn in order:
        workflow.add_node(name, fn)
    workflow.add_edge(START, order[0][0])
    for (name, _), (nxt, _) in zip(order, order[1:], strict=False):
        workflow.add_edge(name, nxt)
    workflow.add_edge(order[-1][0], END)
    return workflow


@pytest.mark.unit
def test_debate_rounds_come_from_the_real_turn_counters():
    seen, observer = _collect()
    handler = GraphObservationHandler(
        observer, max_debate_rounds=2, max_risk_discuss_rounds=1
    )

    _debate_workflow().compile().invoke(
        _initial_state(), config={"callbacks": [handler], "recursion_limit": 25}
    )

    rounds = {
        o.node: o.payload for o in seen if o.type == NODE_COMPLETED and "total_turns" in o.payload
    }
    # 2 * max_debate_rounds and 3 * max_risk_discuss_rounds are the same totals
    # ConditionalLogic routes on.
    assert rounds["Bull Researcher"]["completed_turns"] == 1
    assert rounds["Bull Researcher"]["total_turns"] == 4
    assert rounds["Bear Researcher"]["completed_turns"] == 2
    assert rounds["Conservative Analyst"] == {
        "run_id": rounds["Conservative Analyst"]["run_id"],
        "completed_turns": 1,
        "total_turns": 3,
        "latest_speaker": "Conservative",
    }

    assert {(o.stage, o.node) for o in seen if o.type == STAGE_COMPLETED} == {
        ("research", "Research Manager"),
        ("trading", "Trader"),
        ("risk", "Portfolio Manager"),
    }
    assert {(o.stage, o.payload["report_key"]) for o in seen if o.type == REPORT_AVAILABLE} == {
        ("research", "investment_plan"),
        ("trading", "trader_investment_plan"),
        ("risk", "final_trade_decision"),
    }


@pytest.mark.unit
def test_observing_the_graph_does_not_count_a_model_response_twice():
    """A usage counter keeps counting each response once (design 11.2)."""
    from cli.stats_handler import StatsCallbackHandler

    stats = StatsCallbackHandler()
    seen, observer = _collect()

    _run(
        _analyst_workflow(model_callbacks=[stats]),
        GraphObservationHandler(observer),
    )

    # The usage counter is bound to the model; the observation handler travels
    # through the graph config. They are separate objects and separate counts.
    assert stats.get_stats()["llm_calls"] == 2  # one per analyst turn
    assert [o.type for o in seen].count(TOOL_STARTED) == 1

    # The CLI hands one handler to both the LLM constructor and the graph
    # config; langchain-core 1.6.3 still invokes it once per run.
    shared = StatsCallbackHandler()
    _run(_analyst_workflow(model_callbacks=[shared]), shared)
    assert shared.get_stats()["llm_calls"] == 2


@pytest.mark.unit
def test_a_failing_listener_does_not_end_the_analysis(caplog):
    def observer(observation):
        raise RuntimeError("listener down")

    with caplog.at_level(logging.WARNING, logger="tradingagents.graph.observation"):
        result = _run(_analyst_workflow(), GraphObservationHandler(observer))

    assert result["market_report"] == _REPORT
    assert "Observation listener failed" in caplog.text


@pytest.mark.unit
def test_emit_without_an_observer_is_a_no_op():
    emit(None, GraphObservation(type=NODE_STARTED, node="Market Analyst"))
