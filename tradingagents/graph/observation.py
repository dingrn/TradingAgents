# TradingAgents/graph/observation.py

"""Generic observations of a run's progress.

A caller that wants live visibility into a run passes ``observer=`` to
:class:`~tradingagents.graph.trading_graph.TradingAgentsGraph`, which forwards a
:class:`GraphObservationHandler` through the graph invocation config and also
reports the outer preparation, memory, identity, resume and completion steps it
performs around the graph. A caller driving LangGraph itself can build the
handler directly and hand it to that config, which is where LangGraph reports
node runs.

The records are plain TA data: a type, the stage and node they belong to, and a
payload of values TA already had. TA does not know what an observer does with
them, and an observer that raises never ends a run that is already paying for
model calls (:func:`emit`).

Graph events come from LangGraph's own run metadata (``langgraph_node``, plus
the ``graph:step:`` run tag that marks a node's own run rather than the router,
model and tool runs nested inside it) and from the state each node returns —
never from agent prose. An analyst report is announced only once its state
field is actually filled, and an analyst stage completes on its message-clear
node, so an intermediate tool round can never look like a finished report.

This handler is deliberately separate from the ``callbacks`` bound to the LLM
clients to count model usage: they stay distinct handler objects, so each
concern sees a model response once.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .analyst_execution import ANALYST_NODE_SPECS

logger = logging.getLogger(__name__)

# Outer lifecycle: the work propagate() owns around the graph.
PREPARING = "preparing"
MEMORY_RESOLVED = "memory_resolved"
IDENTITY_RESOLVED = "identity_resolved"
CHECKPOINT_INITIALIZED = "checkpoint_initialized"
ANALYSIS_COMPLETED = "analysis_completed"
REPORTS_SAVED = "reports_saved"

# Graph lifecycle: reported from LangGraph run metadata and node state.
NODE_STARTED = "node_started"
NODE_COMPLETED = "node_completed"
NODE_FAILED = "node_failed"
TOOL_STARTED = "tool_started"
TOOL_COMPLETED = "tool_completed"
TOOL_FAILED = "tool_failed"
REPORT_AVAILABLE = "report_available"
STAGE_COMPLETED = "stage_completed"

RESEARCH_STAGE = "research"
TRADING_STAGE = "trading"
RISK_STAGE = "risk"

# LangGraph tags a node's own run "graph:step:<n>"; the conditional router, the
# model and the tools nested inside that node are tagged "seq:step:<n>".
_NODE_RUN_TAG = "graph:step:"

_RESEARCH_NODES = ("Bull Researcher", "Bear Researcher", "Research Manager")
_RISK_NODES = (
    "Aggressive Analyst",
    "Conservative Analyst",
    "Neutral Analyst",
    "Portfolio Manager",
)

# An analyst's stage is its analyst key, so a caller can match observations to
# the selection it asked for; all three of its nodes share that stage.
_STAGE_BY_NODE: dict[str, str] = {
    **dict.fromkeys(_RESEARCH_NODES, RESEARCH_STAGE),
    **dict.fromkeys(_RISK_NODES, RISK_STAGE),
    "Trader": TRADING_STAGE,
}

# The state field each node fills when its section of the report is final.
_REPORT_KEY_BY_NODE: dict[str, str] = {
    "Research Manager": "investment_plan",
    "Trader": "trader_investment_plan",
    "Portfolio Manager": "final_trade_decision",
}

# Nodes whose completion ends a stage: an analyst's message-clear node (the
# milestone that separates a finished report from another tool round) and the
# three deciding agents.
_STAGE_COMPLETING_NODES: set[str] = {"Research Manager", "Trader", "Portfolio Manager"}

for _spec in ANALYST_NODE_SPECS.values():
    _STAGE_BY_NODE[_spec.agent_node] = _spec.key
    _STAGE_BY_NODE[_spec.tool_node] = _spec.key
    _STAGE_BY_NODE[_spec.clear_node] = _spec.key
    _REPORT_KEY_BY_NODE[_spec.agent_node] = _spec.report_key
    _STAGE_COMPLETING_NODES.add(_spec.clear_node)


def stage_for_node(node: str | None) -> str | None:
    """The stage a graph node belongs to, or ``None`` for an unknown node."""
    return _STAGE_BY_NODE.get(node) if node else None


@dataclass(frozen=True)
class GraphObservation:
    """One observation of TA's own progress.

    ``node`` is the LangGraph node name, which for an agent node is also that
    agent's label ("Conservative Analyst"). ``payload`` carries TA-native values
    only: no consumer identifiers, no credentials.
    """

    type: str
    stage: str | None = None
    node: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


Observer = Callable[[GraphObservation], None]


def emit(observer: Observer | None, observation: GraphObservation) -> None:
    """Hand one observation to ``observer``, absorbing its failures.

    Observation is a side channel. A display or journal that breaks must not end
    an analysis that is already paying for model calls, so the failure is logged
    and the run continues.
    """
    if observer is None:
        return
    try:
        observer(observation)
    except Exception as exc:
        logger.warning(
            "Observation listener failed for %s (analysis continues): %s",
            observation.type,
            exc,
        )


class GraphObservationHandler(BaseCallbackHandler):
    """Turn LangGraph's node and tool callbacks into :class:`GraphObservation`.

    Forwarded through the graph invocation config (``Propagator.get_graph_args``),
    which is where LangGraph reports node runs; the LLM-bound constructor
    callbacks never see them.
    """

    def __init__(
        self,
        observer: Observer | None,
        max_debate_rounds: int = 1,
        max_risk_discuss_rounds: int = 1,
    ) -> None:
        super().__init__()
        self._observer = observer
        # The same totals ConditionalLogic routes on, so a reported round ends
        # when the debate actually ends.
        self._debate_turns = 2 * max_debate_rounds
        self._risk_turns = 3 * max_risk_discuss_rounds
        # Only the start of a run carries the node/tool identity, so it is kept
        # against the run id until that run ends. LangGraph may run nodes
        # concurrently, so the map is locked like cli.stats_handler's counters.
        self._lock = threading.Lock()
        self._nodes: dict[str, str] = {}
        self._tools: dict[str, tuple[str | None, str | None]] = {}

    def _emit(self, obs_type: str, stage=None, node=None, **payload: Any) -> None:
        emit(
            self._observer,
            GraphObservation(type=obs_type, stage=stage, node=node, payload=payload),
        )

    @staticmethod
    def _node_run(kwargs: dict[str, Any]) -> str | None:
        """The node name when this run is a node's own run, else ``None``."""
        node = (kwargs.get("metadata") or {}).get("langgraph_node")
        if not node:
            return None
        tags = kwargs.get("tags") or ()
        if not any(str(tag).startswith(_NODE_RUN_TAG) for tag in tags):
            return None
        return node

    def _turns(self, node: str, update: dict[str, Any]) -> dict[str, Any]:
        """Real debate counters from the state the node just returned."""
        if node in _RESEARCH_NODES:
            debate = update.get("investment_debate_state") or {}
            if "count" in debate:
                return {
                    "completed_turns": debate["count"],
                    "total_turns": self._debate_turns,
                }
        elif node in _RISK_NODES:
            debate = update.get("risk_debate_state") or {}
            if "count" in debate:
                return {
                    "completed_turns": debate["count"],
                    "total_turns": self._risk_turns,
                    "latest_speaker": debate.get("latest_speaker"),
                }
        return {}

    def on_chain_start(self, serialized, inputs, **kwargs: Any) -> None:
        node = self._node_run(kwargs)
        if node is None:
            return
        run_id = str(kwargs.get("run_id"))
        with self._lock:
            self._nodes[run_id] = node
        self._emit(
            NODE_STARTED,
            stage_for_node(node),
            node,
            run_id=run_id,
            step=(kwargs.get("metadata") or {}).get("langgraph_step"),
        )

    def on_chain_end(self, outputs, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id"))
        with self._lock:
            node = self._nodes.pop(run_id, None)
        if node is None:
            return
        stage = stage_for_node(node)
        update = outputs if isinstance(outputs, dict) else {}
        self._emit(
            NODE_COMPLETED, stage, node, run_id=run_id, **self._turns(node, update)
        )

        report_key = _REPORT_KEY_BY_NODE.get(node)
        report = update.get(report_key) if report_key else None
        if report:
            # The node filled its report field on this turn, so that section is
            # final even while later stages keep running.
            self._emit(
                REPORT_AVAILABLE, stage, node, report_key=report_key, report=report
            )

        if node in _STAGE_COMPLETING_NODES:
            self._emit(STAGE_COMPLETED, stage, node)

    def on_chain_error(self, error, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id"))
        with self._lock:
            node = self._nodes.pop(run_id, None)
        if node is None:
            return
        self._emit(
            NODE_FAILED,
            stage_for_node(node),
            node,
            run_id=run_id,
            error=str(error),
            error_type=type(error).__name__,
        )

    def on_tool_start(self, serialized, input_str, **kwargs: Any) -> None:
        node = (kwargs.get("metadata") or {}).get("langgraph_node")
        # The tool name arrives in ``serialized`` on start and as ``name`` on
        # end, and the node is only in the start metadata, so both are carried
        # across on the run id.
        tool = (serialized or {}).get("name") or kwargs.get("name")
        run_id = str(kwargs.get("run_id"))
        with self._lock:
            self._tools[run_id] = (tool, node)
        self._emit(TOOL_STARTED, stage_for_node(node), node, run_id=run_id, tool=tool)

    def _take_tool(self, kwargs: dict[str, Any]) -> tuple[str, str | None, str | None]:
        run_id = str(kwargs.get("run_id"))
        with self._lock:
            tool, node = self._tools.pop(run_id, (None, None))
        return run_id, tool or kwargs.get("name"), node

    def on_tool_end(self, output, **kwargs: Any) -> None:
        run_id, tool, node = self._take_tool(kwargs)
        self._emit(TOOL_COMPLETED, stage_for_node(node), node, run_id=run_id, tool=tool)

    def on_tool_error(self, error, **kwargs: Any) -> None:
        run_id, tool, node = self._take_tool(kwargs)
        self._emit(
            TOOL_FAILED,
            stage_for_node(node),
            node,
            run_id=run_id,
            tool=tool,
            error=str(error),
            error_type=type(error).__name__,
        )
