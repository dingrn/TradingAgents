"""Vendor router must respect the configured chain and never silently hide a
broken primary.

Regressions for #988 (explicit single-vendor config still fell back to others),
#289 (fallback ran for unchosen vendors), and #989 (serious primary failures
were swallowed without a trace).
"""
import copy
import unittest
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.alpha_vantage_common import (
    AlphaVantageNotConfiguredError,
    AlphaVantageRateLimitError,
)
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    VendorChainRateLimitError,
    VendorRateLimitError,
)
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _reset_config():
    # Hard reset: set_config() merges, so empty DEFAULT dicts (e.g. tool_vendors)
    # don't clear keys leaked by other tests. Replace the global outright.
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _no_data(symbol, *a, **k):
    raise NoMarketDataError(symbol, symbol, "no rows")


def _returns(value):
    def impl(symbol, *a, **k):
        return value
    return impl


def _raises(exc):
    def impl(symbol, *a, **k):
        raise exc
    return impl


@pytest.mark.unit
class VendorRoutingTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def _route(self, vendors_for_get_stock_data):
        return mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": vendors_for_get_stock_data},
            clear=False,
        )

    def test_explicit_single_vendor_does_not_fall_back(self):
        # #988: with yfinance pinned, a healthy alpha_vantage must NOT be used.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        av = mock.Mock(side_effect=_returns("AV_DATA"))
        with self._route({"yfinance": _no_data, "alpha_vantage": av}):
            result = interface.route_to_vendor("get_stock_data", "FAKE", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        av.assert_not_called()  # the unchosen vendor was never tried

    def test_explicit_multi_vendor_falls_back_within_chain(self):
        # Listing both vendors opts in to ordered fallback.
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def test_primary_error_is_logged_not_masked(self):
        # #989: primary errors + fallback no-data -> NO_DATA, but the failure
        # must be visible in logs (broken primary not hidden).
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _raises(ValueError("boom")), "alpha_vantage": _no_data}), \
                self.assertLogs("tradingagents.dataflows.interface", level="WARNING") as cm:
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        joined = "\n".join(cm.output)
        self.assertIn("boom", joined)            # the real error surfaced in logs
        self.assertIn("yfinance", joined)

    def test_unknown_configured_vendor_raises(self):
        set_config({"data_vendors": {"core_stock_apis": "bogus_vendor"}})
        with self.assertRaises(ValueError) as ctx:
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("bogus_vendor", str(ctx.exception))

    def test_default_sentinel_uses_all_vendors(self):
        # No explicit choice ("default") keeps the resilient full-chain behavior.
        set_config({"data_vendors": {"core_stock_apis": "default"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def _route_method(self, method, vendors):
        return mock.patch.dict(interface.VENDOR_METHODS, {method: vendors}, clear=False)

    def test_optional_category_degrades_instead_of_raising(self):
        # An optional enrichment vendor (FRED macro) that raises must NOT abort
        # the run — the router returns a sentinel so the analysis proceeds.
        set_config({"data_vendors": {"macro_data": "fred"}})
        with self._route_method(
            "get_macro_indicators", {"fred": _raises(ValueError("FRED 400: bad series"))}
        ):
            result = interface.route_to_vendor("get_macro_indicators", "cpi", "2026-01-01")
        self.assertIn("DATA_UNAVAILABLE", result)
        self.assertIn("macro_data", result)

    def test_core_category_still_raises_on_error(self):
        # A core category (single configured vendor) propagates the error so a
        # broken primary is loud, not silently degraded.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        with self._route({"yfinance": _raises(ValueError("boom"))}), \
                self.assertRaises(ValueError):
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")


@pytest.mark.unit
class ExhaustedThrottlingTests(unittest.TestCase):
    """An exhausted required chain is a retryable fact about the vendors.

    It used to return a DATA_UNAVAILABLE string, which the analyst read as
    "this instrument has no data" and which no caller could tell apart from a
    genuine verdict without parsing prose. Required-data exhaustion now raises
    the typed condition; everything else about the chain is unchanged.
    """

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def _route(self, method, vendors):
        return mock.patch.dict(interface.VENDOR_METHODS, {method: vendors}, clear=False)

    def test_sole_throttled_required_vendor_raises_typed(self):
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        with self._route("get_stock_data", {"yfinance": _raises(VendorRateLimitError("slow down"))}), \
                self.assertRaises(VendorChainRateLimitError) as ctx:
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIsInstance(ctx.exception, VendorRateLimitError)
        self.assertEqual(ctx.exception.vendors, ["yfinance"])
        self.assertEqual(ctx.exception.method, "get_stock_data")

    def test_every_required_vendor_throttled_keeps_each_vendors_evidence(self):
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})
        av = AlphaVantageRateLimitError("Alpha Vantage rate limit exceeded")
        yf = VendorRateLimitError("Yahoo Finance is unreachable")
        with self._route(
            "get_stock_data", {"alpha_vantage": _raises(av), "yfinance": _raises(yf)}
        ), self.assertRaises(VendorChainRateLimitError) as ctx:
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")

        exhausted = ctx.exception
        # Configured order, every vendor, and the original typed errors: enough
        # for a caller to report what happened without re-running the chain.
        self.assertEqual(exhausted.vendors, ["alpha_vantage", "yfinance"])
        self.assertIs(exhausted.vendor_errors["alpha_vantage"], av)
        self.assertIs(exhausted.__cause__, yf)  # the last throttle, preserved
        self.assertIn("Alpha Vantage rate limit exceeded", str(exhausted))
        self.assertIn("unreachable", str(exhausted))

    def test_a_throttled_vendor_does_not_abort_a_working_fallback(self):
        # The chain must run to the end: a throttle is diagnostic activity, not
        # a terminal condition, while another configured vendor can still serve.
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})
        yfinance = mock.Mock(side_effect=_returns("YF_DATA"))
        with self._route(
            "get_stock_data",
            {"alpha_vantage": _raises(VendorRateLimitError("slow down")), "yfinance": yfinance},
        ):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "YF_DATA")
        yfinance.assert_called_once()

    def test_mixed_no_data_and_throttle_stays_no_data(self):
        # A vendor that answered "no rows" saw the instrument; the throttled one
        # saw nothing. Relabelling this as pure throttling would turn a real
        # no-data verdict into a paid retry.
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})
        with self._route(
            "get_stock_data",
            {"alpha_vantage": _raises(VendorRateLimitError("slow down")), "yfinance": _no_data},
        ):
            result = interface.route_to_vendor("get_stock_data", "FAKE", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)

    def test_authentication_failure_is_not_hidden_by_a_later_throttle(self):
        # A bad key is actionable and stays broken until someone fixes it;
        # answering "try again later" would hide it behind endless retries.
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})
        with self._route(
            "get_stock_data",
            {
                "alpha_vantage": _raises(AlphaVantageNotConfiguredError("API key invalid")),
                "yfinance": _raises(VendorRateLimitError("slow down")),
            },
        ), self.assertRaises(AlphaVantageNotConfiguredError):
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")

    def test_optional_category_exhaustion_still_degrades(self):
        # Optional enrichment must never end an otherwise usable analysis, even
        # when its whole chain is throttled.
        set_config({"data_vendors": {"macro_data": "fred"}})
        with self._route(
            "get_macro_indicators", {"fred": _raises(VendorRateLimitError("FRED is throttling"))}
        ):
            result = interface.route_to_vendor("get_macro_indicators", "cpi", "2026-01-01")
        self.assertIn("DATA_UNAVAILABLE", result)
        self.assertIn("throttling", result)

    def test_structured_retry_timing_survives_the_chain(self):
        # Only what a vendor stated structurally: the prose-only throttle
        # contributes no timing, so the caller's own backoff decides.
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})
        with self._route(
            "get_stock_data",
            {
                "alpha_vantage": _raises(VendorRateLimitError("wait 90 seconds please")),
                "yfinance": _raises(VendorRateLimitError("throttled", retry_after=30.0)),
            },
        ), self.assertRaises(VendorChainRateLimitError) as ctx:
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        # 30 from the header, not 90 read out of the other vendor's sentence.
        self.assertEqual(ctx.exception.retry_after, 30.0)

    def test_no_structured_timing_leaves_retry_after_unset(self):
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        with self._route(
            "get_stock_data", {"yfinance": _raises(VendorRateLimitError("retry in 60 seconds"))}
        ), self.assertRaises(VendorChainRateLimitError) as ctx:
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIsNone(ctx.exception.retry_after)


if __name__ == "__main__":
    unittest.main()
