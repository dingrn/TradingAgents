from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from tradingagents.dataflows.date_window import as_of
from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot
from tradingagents.dataflows.symbol_utils import NoMarketDataError


@tool
def get_verified_market_snapshot(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[
        int, "number of recent trading rows to include for sanity-checking"
    ] = 30,
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """Deterministic verification snapshot for exact market-data claims.

    Returns the latest OHLCV row on or before curr_date, common technical
    indicators, and recent closes. Call this before making exact claims about
    price levels, Bollinger bands, RSI, MACD, moving averages, support /
    resistance, or historical comparisons, and treat it as the source of truth.
    """
    effective_date = as_of(curr_date, trade_date)
    try:
        return build_verified_market_snapshot(symbol, effective_date, look_back_days)
    except NoMarketDataError as exc:
        return (
            f"NO_DATA_AVAILABLE: Verified market snapshot unavailable for {symbol} "
            f"on {effective_date}: {exc}. This does not establish that the ticker is invalid. "
            "Do not estimate or fabricate prices or indicators. Report the verification "
            "gap and retry later or use an earlier analysis date explicitly."
        )
