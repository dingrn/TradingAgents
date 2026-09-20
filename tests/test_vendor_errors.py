"""The vendor data-error hierarchy: every "vendor couldn't return usable data"
condition derives from VendorError, so the router catches base types and any
vendor slots in without new handling.
"""
import copy
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.alpha_vantage_common import (
    AlphaVantageNotConfiguredError,
    AlphaVantageRateLimitError,
)
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorChainRateLimitError,
    VendorError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from tradingagents.dataflows.fred import FredNotConfiguredError
from tradingagents.dataflows.utils import retry_after_seconds


@pytest.mark.unit
class HierarchyTests(unittest.TestCase):
    def test_all_conditions_derive_from_vendor_error(self):
        for cls in (NoMarketDataError, VendorRateLimitError, VendorNotConfiguredError):
            self.assertTrue(issubclass(cls, VendorError))

    def test_not_configured_is_still_a_value_error(self):
        # Back-compat: existing `except ValueError` callers keep working.
        self.assertTrue(issubclass(VendorNotConfiguredError, ValueError))

    def test_vendor_named_errors_subclass_the_generic_bases(self):
        self.assertTrue(issubclass(AlphaVantageRateLimitError, VendorRateLimitError))
        self.assertTrue(issubclass(AlphaVantageNotConfiguredError, VendorNotConfiguredError))
        self.assertTrue(issubclass(FredNotConfiguredError, VendorNotConfiguredError))
        # ... and therefore still ValueErrors
        self.assertTrue(issubclass(FredNotConfiguredError, ValueError))

    def test_symbol_utils_reexports_no_market_data_error(self):
        from tradingagents.dataflows.symbol_utils import (
            NoMarketDataError as ReExported,
        )
        self.assertIs(ReExported, NoMarketDataError)

    def test_exhausted_chain_is_still_a_rate_limit(self):
        # Callers that only care "was this throttled?" keep one check; callers
        # scheduling a retry can ask for the terminal form specifically.
        exhausted = VendorChainRateLimitError(
            "get_stock_data", {"yfinance": VendorRateLimitError("slow down")}
        )
        self.assertIsInstance(exhausted, VendorRateLimitError)
        self.assertIsInstance(exhausted, VendorError)

    def test_retry_after_is_absent_unless_a_vendor_stated_it(self):
        self.assertIsNone(VendorRateLimitError("slow down").retry_after)
        self.assertEqual(VendorRateLimitError("slow down", retry_after=12.0).retry_after, 12.0)

    def test_chain_retry_after_is_the_earliest_a_vendor_offered(self):
        # Any one vendor returning is enough to serve the call.
        exhausted = VendorChainRateLimitError(
            "get_stock_data",
            {
                "alpha_vantage": VendorRateLimitError("throttled", retry_after=120.0),
                "yfinance": VendorRateLimitError("throttled", retry_after=45.0),
            },
        )
        self.assertEqual(exhausted.retry_after, 45.0)


@pytest.mark.unit
class RetryAfterHeaderTests(unittest.TestCase):
    """Structured ``Retry-After`` only: never a number read out of prose."""

    def _response(self, value):
        return mock.Mock(headers={"Retry-After": value} if value is not None else {})

    def test_delta_seconds(self):
        self.assertEqual(retry_after_seconds(self._response("30")), 30.0)

    def test_http_date(self):
        when = datetime.now(timezone.utc) + timedelta(seconds=60)
        seconds = retry_after_seconds(self._response(format_datetime(when, usegmt=True)))
        self.assertIsNotNone(seconds)
        self.assertAlmostEqual(seconds, 60.0, delta=5.0)

    def test_a_past_http_date_is_zero_not_negative(self):
        when = datetime.now(timezone.utc) - timedelta(seconds=60)
        self.assertEqual(retry_after_seconds(self._response(format_datetime(when, usegmt=True))), 0.0)

    def test_unparseable_or_missing_header_yields_nothing(self):
        for value in (None, "", "   ", "in a little while"):
            self.assertIsNone(retry_after_seconds(self._response(value)))
        self.assertIsNone(retry_after_seconds(None))

    def test_sec_edgar_carries_the_header_into_the_typed_error(self):
        from tradingagents.dataflows import sec_edgar

        response = mock.Mock(status_code=429, headers={"Retry-After": "600"})
        failure = requests.HTTPError("429 Too Many Requests")
        failure.response = response
        with mock.patch.object(sec_edgar.requests, "get", side_effect=failure), \
                self.assertRaises(VendorRateLimitError) as ctx:
            sec_edgar._fetch_json("https://data.sec.gov/anything.json")
        self.assertEqual(ctx.exception.retry_after, 600.0)


@pytest.mark.unit
class RouterHandlesBaseTypesTests(unittest.TestCase):
    def setUp(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def tearDown(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def test_rate_limit_subclass_caught_by_base(self):
        # A vendor-named rate-limit error skips to the next vendor in the chain.
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})

        def _throttled(*a, **k):
            raise AlphaVantageRateLimitError("slow down")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": _throttled, "yfinance": lambda *a, **k: "YF"}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(out, "YF")

    def test_not_configured_falls_through_to_next_vendor(self):
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})

        def _unconfigured(*a, **k):
            raise AlphaVantageNotConfiguredError("no key")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": _unconfigured, "yfinance": lambda *a, **k: "YF"}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(out, "YF")

    def test_sole_unconfigured_vendor_surfaces_the_error(self):
        # With no fallback, the not-configured condition must surface (not vanish).
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage"}})

        def _unconfigured(*a, **k):
            raise AlphaVantageNotConfiguredError("no key")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": _unconfigured}},
            clear=False,
        ), self.assertRaises(AlphaVantageNotConfiguredError):
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")


if __name__ == "__main__":
    unittest.main()
