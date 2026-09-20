"""Vendor data-error taxonomy.

A single hierarchy so the routing layer reacts by *behavior*, not by vendor:
every condition where a vendor cannot return usable data derives from
``VendorError``, and the router catches the base types. A new vendor raises
these (or a thin vendor-named subclass) and needs no new ``except`` clause.

    VendorError
    ├── NoMarketDataError          no usable rows (empty result OR stale data)
    ├── VendorRateLimitError       transient throttle -> skip to next vendor
    │   └── VendorChainRateLimitError  every vendor throttled -> required call fails
    └── VendorNotConfiguredError   missing API key/config -> vendor unavailable

The number of types is the number of distinct router reactions, not the number
of human-describable causes: empty and stale data get identical handling, so
they share ``NoMarketDataError`` and differ only in the free-text ``detail``.
"""

from __future__ import annotations

from collections.abc import Mapping


class VendorError(Exception):
    """Base for any condition where a vendor could not return usable data."""


class NoMarketDataError(VendorError):
    """A vendor returned no usable rows for a symbol (empty result or stale data).

    Carries both the symbol the user requested and the canonical symbol the
    vendor was actually queried with, plus a free-text ``detail``, so callers
    can build a clear message instead of emitting a vendor-specific empty
    string into the data channel.
    """

    def __init__(self, symbol: str, canonical: str | None = None, detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class VendorRateLimitError(VendorError):
    """A vendor throttled the request; the router skips to the next vendor.

    ``retry_after`` carries the vendor's own structured wait in seconds (an
    HTTP ``Retry-After`` header, an SDK field), so a caller scheduling a retry
    uses what the vendor actually said. It stays ``None`` when the vendor only
    described the throttle in prose: an invented number is worse than falling
    back to the caller's own bounded backoff.
    """

    def __init__(self, *args, retry_after: float | None = None):
        super().__init__(*args)
        self.retry_after = retry_after


class VendorChainRateLimitError(VendorRateLimitError):
    """Every vendor in a required-data chain throttled the request.

    Distinct from a single vendor's throttle because it is terminal: the
    fallback chain is exhausted and the call cannot be served now. Raising it
    (rather than returning a sentinel string) is what lets a caller tell
    "every vendor is throttled, retry later" from "this symbol has no data",
    which is a fact about the instrument, without reading message text.
    """

    def __init__(self, method: str, vendor_errors: Mapping[str, VendorRateLimitError]):
        self.method = method
        self.vendor_errors = dict(vendor_errors)
        self.vendors = list(self.vendor_errors)
        waits = [
            e.retry_after for e in self.vendor_errors.values() if e.retry_after is not None
        ]
        detail = "; ".join(f"{vendor}: {err}" for vendor, err in self.vendor_errors.items())
        super().__init__(
            f"All configured vendors for {method!r} are rate limited ({detail})",
            # One vendor coming back is enough to serve the call, so the
            # earliest advertised wait is the first moment a retry can work.
            retry_after=min(waits) if waits else None,
        )


class VendorNotConfiguredError(VendorError, ValueError):
    """A vendor was selected but its API key/configuration is missing.

    Also a ``ValueError`` so existing callers that catch ``ValueError`` keep
    working while the routing layer can treat it as "vendor unavailable".
    """
