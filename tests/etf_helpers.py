"""Builders for synthetic raw API data (the same shape RIT returns), shared by ETF tests."""

from __future__ import annotations

from etf_config import (
    FALLBACK_CURRENCY,
    FALLBACK_FEE_PER_SHARE,
    FALLBACK_LIMIT_WEIGHTS,
    FALLBACK_MAX_TRADE_SIZE,
    FALLBACK_REBATE_PER_SHARE,
    TICKERS,
)
from snapshot import build_snapshot

DEEP = 1_000_000
LIMIT_NAME = "LIMIT-STOCK"


def book(bids=(), asks=()) -> dict:
    """bids/asks as (price, quantity) pairs, best first. Uses RIT's `bids`/`asks` keys."""
    return {
        "bids": [{"price": p, "quantity": q, "quantity_filled": 0} for p, q in bids],
        "asks": [{"price": p, "quantity": q, "quantity_filled": 0} for p, q in asks],
    }


def quote(bid: float, ask: float, depth: float = DEEP) -> dict:
    return book([(bid, depth)], [(ask, depth)])


def standard_books(**overrides) -> dict:
    """Fairly priced, deep books: 25.00 USD x 1.40 = 35.00 CAD = 14.00 + 21.00."""
    books = {
        "RITC": quote(24.99, 25.01),
        "BULL": quote(13.99, 14.01),
        "BEAR": quote(20.99, 21.01),
        "USD": quote(1.3999, 1.4001, 100_000_000),
    }
    books.update(overrides)
    return books


def security(ticker: str, position: float = 0.0, fee: float | None = None, units: float | None = None) -> dict:
    row = {
        "ticker": ticker,
        "position": position,
        "currency": FALLBACK_CURRENCY[ticker],
        "trading_fee": FALLBACK_FEE_PER_SHARE[ticker] if fee is None else fee,
        "limit_order_rebate": FALLBACK_REBATE_PER_SHARE[ticker],
        "max_trade_size": FALLBACK_MAX_TRADE_SIZE[ticker],
        "quoted_decimals": 4 if ticker in ("USD", "CAD") else 2,
        "is_tradeable": True,
    }
    weight = FALLBACK_LIMIT_WEIGHTS.get(ticker, 0.0) if units is None else units
    if weight:
        row["limits"] = [{"name": LIMIT_NAME, "units": weight}]
    return row


def tender(
    tender_id: int = 1,
    action: str = "SELL",
    quantity: float = 600,
    price: float = 25.10,
    expires: int = 40,
    tick: int = 10,
    fixed: bool = True,
) -> dict:
    return {
        "tender_id": tender_id,
        "period": 1,
        "tick": tick,
        "expires": expires,
        "caption": f"Tender {tender_id}",
        "quantity": quantity,
        "action": action,
        "is_fixed_bid": fixed,
        "price": price if fixed else None,
    }


def make_raw(
    books: dict | None = None,
    positions: dict | None = None,
    fees: dict | None = None,
    tick: int = 10,
    status: str = "ACTIVE",
    gross_limit: float = 250_000,
    net_limit: float = 100_000,
    tenders=(),
    orders=(),
) -> dict:
    positions = positions or {}
    fees = fees or {}
    securities = [security(t, positions.get(t, 0.0), fees.get(t)) for t in TICKERS]
    gross = sum(abs(positions.get(t, 0.0)) * w for t, w in FALLBACK_LIMIT_WEIGHTS.items())
    net = sum(positions.get(t, 0.0) * w for t, w in FALLBACK_LIMIT_WEIGHTS.items())
    return {
        "case": {
            "name": "RITC - Algorithmic ETF Arbitrage (synthetic)",
            "period": 1,
            "tick": tick,
            "ticks_per_period": 300,
            "total_periods": 1,
            "status": status,
        },
        "books": books if books is not None else standard_books(),
        "securities": securities,
        "limits": [
            {
                "name": LIMIT_NAME,
                "gross": gross,
                "net": net,
                "gross_limit": gross_limit,
                "net_limit": net_limit,
                "gross_fine": 0,
                "net_fine": 0,
            }
        ],
        "tenders": list(tenders),
        "orders": list(orders),
        "started_at": 1_000.0,
        "finished_at": 1_000.2,
    }


def make_snapshot(**kwargs):
    return build_snapshot(make_raw(**kwargs))


# --- the guide's and training slides' worked examples -------------------------------


def guide_basket_raw(fees: dict | None = None) -> dict:
    """Guide: sell RITC at 18.70 USD, FX 1.35, buy BULL 10.02 and BEAR 15.03 CAD -> 0.195 CAD/unit."""
    return make_raw(
        books={
            "RITC": book([(18.70, 300), (18.60, 10_000)], [(18.72, 10_000)]),
            "BULL": quote(10.01, 10.02),
            "BEAR": quote(15.02, 15.03),
            "USD": quote(1.35, 1.3502, 100_000_000),
        },
        fees=fees,
    )


def guide_losing_tender_raw() -> dict:
    """Guide: sell 600 RITC to an institution at 25.10; asks 200 @ 25.00 then 400 @ 25.16."""
    return make_raw(
        books=standard_books(RITC=book([(24.98, DEEP)], [(25.00, 200), (25.16, 400)])),
        tenders=[tender(1, "SELL", 600, 25.10)],
    )


def slide10_tender_raw(asks, gross_limit: float = 300_000, net_limit: float = 300_000) -> dict:
    """Slide 10: an institution buys 98,000 RITC from you at 25.15 (you SELL).
    98,000 ETF units count 196,000 toward the limits, so the default limits leave room."""
    return make_raw(
        books=standard_books(RITC=book([(25.06, DEEP)], asks)),
        tenders=[tender(7, "SELL", 98_000, 25.15)],
        gross_limit=gross_limit,
        net_limit=net_limit,
    )
