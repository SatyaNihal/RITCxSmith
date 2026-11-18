"""Turn raw RIT API responses into one timestamped market snapshot.

`collect_raw()` makes the HTTP calls and returns the untouched JSON (safe to
record: it holds market data and positions, never credentials).
`build_snapshot()` parses that raw dict, so live data, recorded files and test
fixtures all go through the same code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from etf_config import (
    BOOK_TICKERS,
    DEFAULT_TICKS_PER_PERIOD,
    ETF,
    FALLBACK_CURRENCY,
    FALLBACK_FEE_PER_SHARE,
    FALLBACK_LIMIT_WEIGHTS,
    FALLBACK_MAX_TRADE_SIZE,
    FALLBACK_REBATE_PER_SHARE,
    TICKERS,
)

Level = tuple[float, float]  # (price, available quantity)


def limit_weight(units: float) -> float:
    """Limit units one share uses, read conservatively from /securities `limits[].units`.

    On the practice server RIT reports RITC at 0.5 and the stocks at 1.0, while the
    case packet says each ETF unit counts twice. So 0.5 most likely means "half a
    share per limit unit". Until a real position confirms it (bot.py compares its
    exposure with /limits every loop), take the larger of units and 1/units.
    """
    return max(units, 1.0 / units)


@dataclass(frozen=True)
class Book:
    ticker: str
    bids: tuple[Level, ...]  # highest price first
    asks: tuple[Level, ...]  # lowest price first
    bids_truncated: bool = False  # RIT returned as many orders as we asked for
    asks_truncated: bool = False

    def truncated(self, action: str) -> bool:
        """True when the side you'd trade against may continue past what RIT returned."""
        return self.asks_truncated if action == "BUY" else self.bids_truncated

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    def levels_for(self, action: str) -> tuple[Level, ...]:
        """Levels you trade against: asks when you BUY, bids when you SELL."""
        return self.asks if action == "BUY" else self.bids

    def depth(self, action: str) -> float:
        return sum(quantity for _, quantity in self.levels_for(action))


@dataclass(frozen=True)
class Security:
    ticker: str
    position: float
    bid: float | None
    ask: float | None
    last: float | None
    currency: str
    trading_fee: float
    limit_order_rebate: float
    max_trade_size: float
    quoted_decimals: int | None
    is_tradeable: bool
    limit_units: dict[str, float]  # limit name -> units counted per share


@dataclass(frozen=True)
class Limit:
    name: str
    gross: float
    net: float
    gross_limit: float
    net_limit: float


@dataclass(frozen=True)
class Tender:
    tender_id: int
    period: int
    tick: int
    expires: int
    caption: str
    quantity: float
    action: str
    is_fixed_bid: bool
    price: float | None
    ticker: str


@dataclass(frozen=True)
class OpenOrder:
    order_id: int
    ticker: str
    action: str
    order_type: str
    quantity: float
    quantity_filled: float
    price: float | None

    @property
    def remaining(self) -> float:
        return max(0.0, self.quantity - self.quantity_filled)


@dataclass(frozen=True)
class Snapshot:
    case_name: str
    period: int
    tick: int
    status: str
    ticks_per_period: int
    securities: dict[str, Security]
    books: dict[str, Book]
    limits: tuple[Limit, ...]
    tenders: tuple[Tender, ...]
    open_orders: tuple[OpenOrder, ...]
    started_at: float
    finished_at: float

    # -- convenience ------------------------------------------------------

    def position(self, ticker: str) -> float:
        security = self.securities.get(ticker)
        return security.position if security else 0.0

    def positions(self) -> dict[str, float]:
        return {ticker: security.position for ticker, security in self.securities.items()}

    @property
    def ticks_left(self) -> int:
        return max(0, self.ticks_per_period - self.tick)

    def age_s(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.started_at

    @property
    def collection_span_s(self) -> float:
        return self.finished_at - self.started_at

    def fee(self, ticker: str) -> float:
        security = self.securities.get(ticker)
        return security.trading_fee if security else FALLBACK_FEE_PER_SHARE.get(ticker, 0.0)

    def rebate(self, ticker: str) -> float:
        security = self.securities.get(ticker)
        return security.limit_order_rebate if security else FALLBACK_REBATE_PER_SHARE.get(ticker, 0.0)

    def max_size(self, ticker: str) -> float:
        security = self.securities.get(ticker)
        return security.max_trade_size if security else FALLBACK_MAX_TRADE_SIZE.get(ticker, 0.0)

    def currency(self, ticker: str) -> str:
        security = self.securities.get(ticker)
        return security.currency if security else FALLBACK_CURRENCY.get(ticker, "CAD")

    def limit_weights(self) -> dict[str, dict[str, float]]:
        """Per limit name, how many limit units one share of each ticker uses."""
        weights: dict[str, dict[str, float]] = {}
        for limit in self.limits:
            reported = {
                ticker: limit_weight(security.limit_units[limit.name])
                for ticker, security in self.securities.items()
                if limit.name in security.limit_units
            }
            weights[limit.name] = reported or dict(FALLBACK_LIMIT_WEIGHTS)
        return weights

    def missing(self) -> list[str]:
        """Reasons this snapshot can't support a decision (empty = complete)."""
        problems = [f"no /securities row for {t}" for t in TICKERS if t not in self.securities]
        for ticker in BOOK_TICKERS:
            book = self.books.get(ticker)
            if book is None:
                problems.append(f"no book for {ticker}")
            elif not book.bids or not book.asks:
                problems.append(f"{ticker} book has an empty side")
        return problems

    def is_complete(self) -> bool:
        return not self.missing()


# -- parsing ---------------------------------------------------------------


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_present(data: Any, *keys: str) -> list:
    if isinstance(data, dict):
        for key in keys:
            if data.get(key) is not None:
                return data[key]
    return []


def _aggregate(entries: list, highest_first: bool) -> tuple[Level, ...]:
    totals: dict[float, float] = {}
    for entry in entries or []:
        price = _optional_float(entry.get("price")) if isinstance(entry, dict) else None
        if price is None:
            continue
        available = _float(entry.get("quantity")) - _float(entry.get("quantity_filled"))
        if available <= 0:
            continue
        key = round(price, 6)
        totals[key] = totals.get(key, 0.0) + available
    return tuple(sorted(totals.items(), key=lambda level: level[0], reverse=highest_first))


def parse_book(ticker: str, data: Any, depth_limit: int | None = None) -> Book:
    """Group individual book orders by price. Accepts `bids`/`asks` (what RIT
    returns) and `bid`/`ask` (what the API spec documents). With `depth_limit`,
    flags a side that came back with that many orders as possibly cut off."""
    bids = _first_present(data, "bids", "bid")
    asks = _first_present(data, "asks", "ask")
    return Book(
        ticker=ticker,
        bids=_aggregate(bids, highest_first=True),
        asks=_aggregate(asks, highest_first=False),
        bids_truncated=bool(depth_limit) and len(bids) >= depth_limit,
        asks_truncated=bool(depth_limit) and len(asks) >= depth_limit,
    )


def parse_security(data: dict) -> Security:
    ticker = str(data.get("ticker"))
    fee = data.get("trading_fee")
    rebate = data.get("limit_order_rebate")
    max_size = data.get("max_trade_size")
    decimals = data.get("quoted_decimals")
    return Security(
        ticker=ticker,
        position=_float(data.get("position")),
        bid=_optional_float(data.get("bid")),
        ask=_optional_float(data.get("ask")),
        last=_optional_float(data.get("last")),
        currency=str(data.get("currency") or FALLBACK_CURRENCY.get(ticker, "CAD")),
        trading_fee=_float(fee, FALLBACK_FEE_PER_SHARE.get(ticker, 0.0)) if fee is not None
        else FALLBACK_FEE_PER_SHARE.get(ticker, 0.0),
        limit_order_rebate=_float(rebate) if rebate is not None
        else FALLBACK_REBATE_PER_SHARE.get(ticker, 0.0),
        max_trade_size=_float(max_size) if max_size else FALLBACK_MAX_TRADE_SIZE.get(ticker, 0.0),
        quoted_decimals=int(decimals) if decimals is not None else None,
        is_tradeable=bool(data.get("is_tradeable", True)),
        limit_units={
            str(item["name"]): _float(item.get("units"))
            for item in data.get("limits") or []
            if isinstance(item, dict) and item.get("name") is not None and _float(item.get("units")) > 0
        },
    )


def parse_limits(data: Any) -> tuple[Limit, ...]:
    limits = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        gross_limit = _float(item.get("gross_limit"))
        net_limit = _float(item.get("net_limit"))
        limits.append(
            Limit(
                name=str(item.get("name", "limit")),
                gross=_float(item.get("gross")),
                net=_float(item.get("net")),
                gross_limit=gross_limit if gross_limit > 0 else float("inf"),
                net_limit=net_limit if net_limit > 0 else float("inf"),
            )
        )
    return tuple(limits)


def parse_tender(data: dict) -> Tender:
    return Tender(
        tender_id=int(data.get("tender_id")),
        period=int(_float(data.get("period"))),
        tick=int(_float(data.get("tick"))),
        expires=int(_float(data.get("expires"))),
        caption=str(data.get("caption") or ""),
        quantity=_float(data.get("quantity")),
        action=str(data.get("action") or "").upper(),
        is_fixed_bid=bool(data.get("is_fixed_bid")),
        price=_optional_float(data.get("price")),
        ticker=str(data.get("ticker") or ETF),
    )


def parse_open_order(data: dict) -> OpenOrder:
    return OpenOrder(
        order_id=int(data.get("order_id")),
        ticker=str(data.get("ticker")),
        action=str(data.get("action") or "").upper(),
        order_type=str(data.get("type") or ""),
        quantity=_float(data.get("quantity")),
        quantity_filled=_float(data.get("quantity_filled")),
        price=_optional_float(data.get("price")),
    )


def build_snapshot(raw: dict) -> Snapshot:
    case = raw.get("case") or {}
    securities = {}
    for row in raw.get("securities") or []:
        if isinstance(row, dict) and row.get("ticker"):
            security = parse_security(row)
            securities[security.ticker] = security
    depth_limit = int(_float(raw.get("book_depth"))) or None
    books = {ticker: parse_book(ticker, data, depth_limit) for ticker, data in (raw.get("books") or {}).items()}
    tenders = tuple(
        parse_tender(row) for row in raw.get("tenders") or [] if isinstance(row, dict) and "tender_id" in row
    )
    open_orders = tuple(
        parse_open_order(row)
        for row in raw.get("orders") or []
        if isinstance(row, dict) and row.get("order_id") is not None
    )
    started_at = _float(raw.get("started_at"), time.time())
    return Snapshot(
        case_name=str(case.get("name") or ""),
        period=int(_float(case.get("period"))),
        tick=int(_float(case.get("tick"))),
        status=str(case.get("status") or "UNKNOWN"),
        ticks_per_period=int(_float(case.get("ticks_per_period"), DEFAULT_TICKS_PER_PERIOD))
        or DEFAULT_TICKS_PER_PERIOD,
        securities=securities,
        books=books,
        limits=parse_limits(raw.get("limits")),
        tenders=tenders,
        open_orders=open_orders,
        started_at=started_at,
        finished_at=_float(raw.get("finished_at"), started_at),
    )


# -- collection (HTTP, read-only) ---------------------------------------------


def collect_raw(client: Any, book_depth: int, case: dict | None = None) -> dict:
    """Read everything one decision needs. Only GET requests."""
    raw: dict[str, Any] = {"started_at": time.time(), "book_depth": book_depth}
    raw["case"] = case if case is not None else client.case()
    raw["books"] = {ticker: client.book(ticker, limit=book_depth) for ticker in BOOK_TICKERS}
    raw["securities"] = client.securities()
    raw["limits"] = client.limits()
    raw["tenders"] = client.tenders()
    raw["orders"] = client.orders("OPEN")
    raw["finished_at"] = time.time()
    return raw


# -- display ---------------------------------------------------------------


def _quote(book: Book | None, decimals: int = 2) -> str:
    if book is None or book.best_bid is None or book.best_ask is None:
        return "--/--"
    return f"{book.best_bid:.{decimals}f}/{book.best_ask:.{decimals}f}"


def format_market_line(snap: Snapshot) -> str:
    fx_decimals = snap.securities.get("USD").quoted_decimals if "USD" in snap.securities else None
    positions = " ".join(
        f"{ticker} {snap.position(ticker):+,.0f}" for ticker in ("RITC", "BULL", "BEAR", "USD")
    )
    limits = " ".join(
        f"{limit.name} g{limit.gross:,.0f}/{limit.gross_limit:,.0f} n{limit.net:+,.0f}/{limit.net_limit:,.0f}"
        for limit in snap.limits
    )
    return (
        f"p{snap.period} t{snap.tick:>3} | RITC {_quote(snap.books.get('RITC'))} "
        f"BULL {_quote(snap.books.get('BULL'))} BEAR {_quote(snap.books.get('BEAR'))} "
        f"USD {_quote(snap.books.get('USD'), fx_decimals or 4)} | {positions} | {limits}"
    )
