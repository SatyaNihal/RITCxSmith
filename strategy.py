"""Pricing and risk decisions for the ETF case.

Pure functions: no HTTP and no clocks. Each takes a Snapshot (or plain numbers)
and returns numbers or a Decision with a readable reason, so any logged line can
be reproduced by hand from the prices in that snapshot.

Conventions
- Buy at the ask, sell at the bid, walking down the book for size.
- USD you receive converts to CAD at the USD bid; USD you need costs the USD ask.
- Fees are charged per share in the instrument's quote currency (VERIFY: check
  the COMMISSION rows in /transactions for RITC).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

from etf_config import ETF, FALLBACK_LIMIT_WEIGHTS, FX, STOCKS, StrategyConfig
from snapshot import Level, Snapshot, Tender

EPS = 1e-9
RICH = "ETF_RICH"    # sell RITC, buy BULL + BEAR
CHEAP = "ETF_CHEAP"  # buy RITC, sell BULL + BEAR
BASKET_ACTIONS = {
    RICH: {ETF: "SELL", "BULL": "BUY", "BEAR": "BUY"},
    CHEAP: {ETF: "BUY", "BULL": "SELL", "BEAR": "SELL"},
}


# -- order book ---------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    requested: float
    filled: float
    cost: float               # sum(price x quantity) in the book's quote currency
    worst_price: float | None

    @property
    def complete(self) -> bool:
        return self.filled >= self.requested - EPS

    @property
    def vwap(self) -> float | None:
        return self.cost / self.filled if self.filled > EPS else None


def sweep_book(
    levels: Sequence[Level],
    quantity: float,
    *,
    limit_price: float | None = None,
    buying: bool | None = None,
) -> Fill:
    """Walk price levels (best first) to estimate filling `quantity` now.

    With `limit_price`, stop at levels worse than it (`buying` says which way is worse).
    An incomplete Fill means the visible book can't cover the quantity.
    """
    if limit_price is not None and buying is None:
        raise ValueError("buying must be given with limit_price")
    remaining = max(0.0, float(quantity))
    filled = cost = 0.0
    worst = None
    for price, available in levels:
        if remaining <= EPS:
            break
        if limit_price is not None:
            if buying and price > limit_price + EPS:
                break
            if not buying and price < limit_price - EPS:
                break
        take = min(remaining, available)
        if take <= 0:
            continue
        filled += take
        cost += take * price
        remaining -= take
        worst = price
    return Fill(float(quantity), filled, cost, worst)


# -- currency -----------------------------------------------------------------


def to_cad(snap: Snapshot, amount: float, currency: str) -> float:
    """Convert a cash amount to CAD on the executable side of the USD book."""
    if currency == "CAD" or abs(amount) < EPS:
        return amount
    if currency != "USD":
        raise ValueError(f"unsupported currency {currency!r}")
    book = snap.books.get(FX)
    rate = (book.best_bid if amount > 0 else book.best_ask) if book else None
    if rate is None:
        raise ValueError("USD book has no quote on the needed side")
    return amount * rate


def cost_in_cad(snap: Snapshot, cost: float, currency: str) -> float:
    """A cost (positive number) in `currency`, expressed in CAD."""
    return -to_cad(snap, -cost, currency)


def fee_cad(snap: Snapshot, ticker: str, quantity: float) -> float:
    return cost_in_cad(snap, snap.fee(ticker) * abs(quantity), snap.currency(ticker))


# -- exposure and limits --------------------------------------------------------


def weighted_exposure(positions: dict[str, float], weights: dict[str, float]) -> tuple[float, float]:
    gross = sum(abs(positions.get(ticker, 0.0)) * weight for ticker, weight in weights.items())
    net = sum(positions.get(ticker, 0.0) * weight for ticker, weight in weights.items())
    return gross, net


def exposure(positions: dict[str, float], weights: dict[str, float] | None = None) -> tuple[float, float]:
    """(gross, net) with the case weights: stocks x1, ETF x2."""
    return weighted_exposure(positions, weights or FALLBACK_LIMIT_WEIGHTS)


def _apply(positions: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
    result = dict(positions)
    for ticker, change in delta.items():
        result[ticker] = result.get(ticker, 0.0) + change
    return result


def open_order_delta(snap: Snapshot) -> dict[str, float]:
    delta: dict[str, float] = {}
    for order in snap.open_orders:
        sign = 1.0 if order.action == "BUY" else -1.0
        delta[order.ticker] = delta.get(order.ticker, 0.0) + sign * order.remaining
    return delta


def limit_violation(snap: Snapshot, delta: dict[str, float], headroom: float, label: str) -> str | None:
    """Why applying `delta` (and also all open orders filling) would break a limit, or None.

    A change that leaves exposure lower than now is always allowed, so the bot can
    still reduce a position that is already above the headroom.
    """
    if not snap.limits:
        return "no limits reported by /limits"
    base = snap.positions()
    scenarios = [(label, delta)]
    pending = open_order_delta(snap)
    if any(abs(v) > EPS for v in pending.values()):
        scenarios.append((f"{label} + open orders", _apply(dict(delta), pending)))
    weights_by_limit = snap.limit_weights()
    for limit in snap.limits:
        weights = weights_by_limit[limit.name]
        gross_now, net_now = weighted_exposure(base, weights)
        gross_cap = headroom * limit.gross_limit
        net_cap = headroom * limit.net_limit
        for name, change in scenarios:
            gross, net = weighted_exposure(_apply(base, change), weights)
            if gross > gross_cap + EPS and gross > gross_now + EPS:
                return (
                    f"{name}: gross {gross:,.0f} > {gross_cap:,.0f} "
                    f"({headroom:.0%} of {limit.name} gross limit {limit.gross_limit:,.0f})"
                )
            if abs(net) > net_cap + EPS and abs(net) > abs(net_now) + EPS:
                return (
                    f"{name}: net {net:+,.0f} beyond {net_cap:,.0f} "
                    f"({headroom:.0%} of {limit.name} net limit {limit.net_limit:,.0f})"
                )
    return None


def exposure_mismatch(snap: Snapshot, tolerance: float = 1.0) -> str | None:
    """Compare our weighted exposure with what /limits reports (a live self-check)."""
    weights_by_limit = snap.limit_weights()
    for limit in snap.limits:
        gross, net = weighted_exposure(snap.positions(), weights_by_limit[limit.name])
        if abs(gross - limit.gross) > tolerance or abs(net - limit.net) > tolerance:
            return (
                f"{limit.name}: computed gross {gross:,.0f} net {net:+,.0f} but RIT reports "
                f"gross {limit.gross:,.0f} net {limit.net:+,.0f}; check the limit weights"
            )
    return None


# -- decisions ------------------------------------------------------------------


@dataclass(frozen=True)
class Leg:
    ticker: str
    action: str
    quantity: float
    est_vwap: float | None = None
    limit_price: float | None = None  # deepest price the estimate used


@dataclass(frozen=True)
class Decision:
    kind: str                  # "basket" or "tender"
    name: str                  # ETF_RICH, ETF_CHEAP, or "tender <id>"
    go: bool
    quantity: float
    est_profit_cad: float | None
    reason: str
    legs: tuple[Leg, ...] = ()
    details: dict = field(default_factory=dict)

    @property
    def verb(self) -> str:
        if not self.go:
            return "SKIP"
        return "ACCEPT" if self.kind == "tender" else "TRADE"

    def summary(self) -> str:
        return f"{self.verb:<6} {self.name:<10} {self.reason}"

    def to_dict(self) -> dict:
        return asdict(self)


# -- ETF vs basket ----------------------------------------------------------------


@dataclass(frozen=True)
class BasketEstimate:
    direction: str
    quantity: float
    complete: bool
    gross_cad: float
    fees_cad: float
    buffer_cad: float
    usd_amount: float
    fills: dict[str, Fill]
    fx_fill: Fill | None

    @property
    def net_cad(self) -> float:
        return self.gross_cad - self.fees_cad - self.buffer_cad


def estimate_basket(snap: Snapshot, direction: str, quantity: float, cfg: StrategyConfig) -> BasketEstimate:
    """Net CAD edge of trading `quantity` ETF units against the basket right now."""
    actions = BASKET_ACTIONS[direction]
    fills = {
        ticker: sweep_book(snap.books[ticker].levels_for(action), quantity)
        for ticker, action in actions.items()
    }
    usd_amount = fills[ETF].cost  # USD received (rich) or paid (cheap)
    fx_book = snap.books[FX]
    fx_fill = None
    if direction == RICH:
        if cfg.convert_fx:
            fx_fill = sweep_book(fx_book.bids, usd_amount)
            etf_cad = fx_fill.cost
        else:
            etf_cad = usd_amount * (fx_book.best_bid or 0.0)
        gross = etf_cad - fills["BULL"].cost - fills["BEAR"].cost
    else:
        if cfg.convert_fx:
            fx_fill = sweep_book(fx_book.asks, usd_amount)
            etf_cad = fx_fill.cost
        else:
            etf_cad = usd_amount * (fx_book.best_ask or 0.0)
        gross = fills["BULL"].cost + fills["BEAR"].cost - etf_cad
    fees = sum(fee_cad(snap, ticker, quantity) for ticker in actions)
    if cfg.convert_fx:
        fees += fee_cad(snap, FX, usd_amount)
    complete = all(fill.complete for fill in fills.values()) and (fx_fill is None or fx_fill.complete)
    return BasketEstimate(
        direction=direction,
        quantity=quantity,
        complete=complete,
        gross_cad=gross,
        fees_cad=fees,
        buffer_cad=cfg.basket_buffer_per_unit_cad * quantity,
        usd_amount=usd_amount,
        fills=fills,
        fx_fill=fx_fill,
    )


def _cumulative(levels: Sequence[Level]) -> list[float]:
    total, out = 0.0, []
    for _, quantity in levels:
        total += quantity
        out.append(total)
    return out


def _largest_feasible(quantity: int, lot: int, violation: Callable[[int], str | None]) -> tuple[int, str | None]:
    """Largest multiple of `lot` <= quantity with no violation (feasible set is an interval from 0)."""
    first = violation(quantity)
    if first is None:
        return quantity, None
    # Search in lots: `low` is feasible (0 lots trades nothing), `high` is not.
    low, high = 0, quantity // lot
    while high - low > 1:
        middle = (low + high) // 2
        if violation(middle * lot) is None:
            low = middle
        else:
            high = middle
    return low * lot, first


def basket_opportunity(snap: Snapshot, direction: str, cfg: StrategyConfig) -> Decision:
    actions = BASKET_ACTIONS[direction]

    def skip(reason: str, **details: float) -> Decision:
        return Decision("basket", direction, False, 0, None, reason, details=details)

    for ticker in (*actions, FX):
        book = snap.books.get(ticker)
        if book is None or not book.bids or not book.asks:
            return skip(f"{ticker} book is missing a side")

    unit = estimate_basket(snap, direction, 1, cfg)
    details = {
        "ritc": unit.fills[ETF].vwap,
        "bull": unit.fills["BULL"].vwap,
        "bear": unit.fills["BEAR"].vwap,
        "fx": snap.books[FX].best_bid if direction == RICH else snap.books[FX].best_ask,
        "gross_per_unit": unit.gross_cad,
        "fees_per_unit": unit.fees_cad,
        "buffer_per_unit": unit.buffer_cad,
        "net_per_unit": unit.net_cad,
    }
    head = (
        f"top edge {unit.gross_cad:+.4f} - fees {unit.fees_cad:.4f} - buffer {unit.buffer_cad:.4f} "
        f"= {unit.net_cad:+.4f} CAD/unit"
    )
    threshold = cfg.basket_min_edge_per_unit_cad
    if unit.net_cad < threshold:
        return skip(f"{head} < {threshold:+.4f}", **details)

    cap = min(
        cfg.basket_max_qty,
        *(snap.max_size(ticker) for ticker in actions),
        *(snap.books[ticker].depth(action) for ticker, action in actions.items()),
    )
    breakpoints = sorted(
        {min(level, cap) for ticker, action in actions.items() for level in _cumulative(snap.books[ticker].levels_for(action))}
        | {cap}
    )
    best, previous_qty, previous_net = 0.0, 0.0, 0.0
    for quantity in breakpoints:
        if quantity <= previous_qty + EPS:
            continue
        estimate = estimate_basket(snap, direction, quantity, cfg)
        if not estimate.complete:
            break
        marginal = (estimate.net_cad - previous_net) / (quantity - previous_qty)
        if marginal < threshold:
            break
        best, previous_qty, previous_net = quantity, quantity, estimate.net_cad

    lot = max(1, cfg.basket_lot)
    quantity = int(best // lot) * lot
    if quantity <= 0:
        return skip(f"{head}, but the edge holds for only {best:,.0f} units (< lot {lot})", **details)

    # Legs go thinnest book first; check limits after each leg alone and after all.
    order = sorted(actions, key=lambda ticker: snap.books[ticker].depth(actions[ticker]))

    def violation(qty: int) -> str | None:
        delta: dict[str, float] = {}
        for index, ticker in enumerate(order, start=1):
            delta[ticker] = qty if actions[ticker] == "BUY" else -qty
            label = "after all legs" if index == len(order) else f"after {'+'.join(order[:index])}"
            problem = limit_violation(snap, delta, cfg.limit_headroom, label)
            if problem:
                return problem
        return None

    quantity, limit_reason = _largest_feasible(quantity, lot, violation)
    if quantity <= 0:
        return skip(f"{head}, but {limit_reason}", **details)

    estimate = estimate_basket(snap, direction, quantity, cfg)
    if estimate.net_cad / quantity < threshold:
        return skip(f"{head}, but {quantity:,} units average only {estimate.net_cad / quantity:+.4f}", **details)

    legs = [
        Leg(ticker, actions[ticker], quantity, estimate.fills[ticker].vwap, estimate.fills[ticker].worst_price)
        for ticker in order
    ]
    if cfg.convert_fx and estimate.fx_fill is not None:
        legs.append(
            Leg(
                FX,
                "SELL" if direction == RICH else "BUY",
                round(estimate.usd_amount),
                estimate.fx_fill.vwap,
                estimate.fx_fill.worst_price,
            )
        )
    capped = f"; capped by {limit_reason}" if limit_reason else ""
    vwaps = " ".join(f"{leg.ticker} {leg.est_vwap:.4f}" for leg in legs if leg.est_vwap is not None)
    reason = (
        f"{quantity:,} units est {estimate.net_cad:+,.2f} CAD ({estimate.net_cad / quantity:+.4f}/unit) "
        f"after fees {estimate.fees_cad:,.2f} and buffer {estimate.buffer_cad:,.2f}; VWAP {vwaps}{capped}"
    )
    details.update(
        quantity=quantity,
        gross_cad=estimate.gross_cad,
        fees_cad=estimate.fees_cad,
        buffer_cad=estimate.buffer_cad,
        net_cad=estimate.net_cad,
    )
    return Decision("basket", direction, True, quantity, estimate.net_cad, reason, tuple(legs), details)


# -- tenders ------------------------------------------------------------------------


def tender_position_delta(tender: Tender, cfg: StrategyConfig) -> float:
    """Change in YOUR RITC position if you accept. VERIFY the sign in practice."""
    if tender.action == "BUY":
        sign = 1.0
    elif tender.action == "SELL":
        sign = -1.0
    else:
        return 0.0
    if not cfg.tender_action_is_trader_side:
        sign = -sign
    return sign * tender.quantity


def hedge_target(positions: dict[str, float]) -> float:
    """RITC position that BULL/BEAR holdings already hedge (0 when there is no basket)."""
    bull, bear = positions.get("BULL", 0.0), positions.get("BEAR", 0.0)
    if bull * bear <= 0:
        return 0.0
    hedged = min(abs(bull), abs(bear))
    return -hedged if bull > 0 else hedged


def trade_value(snap: Snapshot, ticker: str, signed_quantity: float) -> tuple[float, Fill, float]:
    """Cash from trading `signed_quantity` now (+ buy, - sell), net of fees, in the
    ticker's quote currency. Quantity beyond the visible book is priced at the worst
    visible level and the returned Fill is marked incomplete."""
    if abs(signed_quantity) < EPS:
        return 0.0, Fill(0.0, 0.0, 0.0, None), 0.0
    action = "BUY" if signed_quantity > 0 else "SELL"
    quantity = abs(signed_quantity)
    fill = sweep_book(snap.books[ticker].levels_for(action), quantity)
    cost = fill.cost
    if not fill.complete and fill.worst_price is not None:
        cost += (quantity - fill.filled) * fill.worst_price
    fee = snap.fee(ticker) * quantity
    value = (-cost if action == "BUY" else cost) - fee
    return value, fill, fee


def evaluate_tender(tender: Tender, snap: Snapshot, cfg: StrategyConfig) -> Decision:
    """Accept only if exiting the incremental position at visible prices still pays."""
    name = f"tender {tender.tender_id}"

    def skip(reason: str, **details: float) -> Decision:
        return Decision("tender", name, False, tender.quantity, details.get("net_cad"), reason, details=details)

    if tender.ticker != ETF:
        return skip(f"tender is for {tender.ticker}, not {ETF}")
    if not tender.is_fixed_bid:
        return skip("variable-price tender: bidding is not implemented")
    if tender.price is None or tender.quantity <= 0:
        return skip("tender has no price or quantity")
    if tender.expires - snap.tick < cfg.min_ticks_to_tender_expiry:
        return skip(f"expires at tick {tender.expires}, now tick {snap.tick}")
    if snap.ticks_left < cfg.min_heat_ticks_left_for_tender:
        return skip(f"only {snap.ticks_left} ticks left in the heat")
    delta = tender_position_delta(tender, cfg)
    if delta == 0:
        return skip(f"unknown tender action {tender.action!r}")
    for ticker in (ETF, FX):
        book = snap.books.get(ticker)
        if book is None or not book.bids or not book.asks:
            return skip(f"{ticker} book is missing a side")

    problem = limit_violation(snap, {ETF: delta}, cfg.tender_limit_headroom, "after accepting")
    if problem:
        return skip(problem)

    positions = snap.positions()
    position = positions.get(ETF, 0.0)
    new_position = position + delta
    target = hedge_target(positions)
    exit_quantity = target - new_position
    after_value, after_fill, after_fee = trade_value(snap, ETF, exit_quantity)
    if not after_fill.complete:
        cut_off = snap.books[ETF].truncated("BUY" if exit_quantity > 0 else "SELL")
        return skip(
            f"exit book covers only {after_fill.filled:,.0f} of {abs(exit_quantity):,.0f} RITC "
            f"(worst visible {after_fill.worst_price})"
            + ("; RIT cut the book off at the request limit, so raise book_depth" if cut_off else "")
        )
    before_value, _, before_fee = trade_value(snap, ETF, target - position)

    currency = snap.currency(ETF)
    tender_cash = -delta * tender.price
    incremental = tender_cash + after_value - before_value
    incremental_cad = to_cad(snap, incremental, currency)
    fees_cad = cost_in_cad(snap, after_fee - before_fee, currency)
    buffer_cad = cfg.tender_buffer_per_unit_cad * abs(delta)
    net_cad = incremental_cad - buffer_cad
    per_unit = net_cad / abs(delta)

    stocks = [snap.books[ticker].mid for ticker in STOCKS if ticker in snap.books]
    fx_mid = snap.books[FX].mid
    fair = sum(stocks) / fx_mid if len(stocks) == 2 and None not in stocks and fx_mid else None
    fair_edge = None if fair is None else (tender.price - fair if delta < 0 else fair - tender.price)

    side = "sell" if delta < 0 else "buy"
    if abs(exit_quantity) > EPS:
        exit_text = (
            f"{'buy back' if exit_quantity > 0 else 'sell'} {abs(exit_quantity):,.0f} at VWAP "
            f"{after_fill.vwap:.4f} (worst {after_fill.worst_price:.2f})"
        )
    else:
        exit_text = "offsets the current position (no exit needed)"
    fair_text = f"; basket fair {fair:.3f} ({fair_edge:+.3f}/unit)" if fair is not None else ""
    estimate = (
        f"{side} {abs(delta):,.0f} @ {tender.price:.2f}, {exit_text}: est {net_cad:+,.0f} CAD "
        f"({per_unit:+.4f}/unit) after fees {fees_cad:,.0f} and buffer {buffer_cad:,.0f}{fair_text}"
    )
    details = {
        "delta": delta,
        "position": position,
        "target": target,
        "exit_quantity": exit_quantity,
        "exit_vwap": after_fill.vwap,
        "exit_worst": after_fill.worst_price,
        "incremental_quote": incremental,
        "fees_cad": fees_cad,
        "buffer_cad": buffer_cad,
        "net_cad": net_cad,
        "fair": fair,
    }
    if net_cad < cfg.min_tender_profit_cad or per_unit < cfg.min_tender_edge_per_unit_cad:
        return skip(
            f"{estimate}; needs >= {cfg.min_tender_profit_cad:,.0f} CAD and "
            f">= {cfg.min_tender_edge_per_unit_cad:.3f}/unit",
            **details,
        )
    legs = ()
    if abs(exit_quantity) > EPS:
        legs = (
            Leg(ETF, "BUY" if exit_quantity > 0 else "SELL", abs(exit_quantity), after_fill.vwap, after_fill.worst_price),
        )
    return Decision("tender", name, True, tender.quantity, net_cad, estimate, legs, details)


def unwind_step(
    snap: Snapshot,
    target: float,
    reference_price: float,
    ticks_since_progress: int,
    cfg: StrategyConfig,
) -> tuple[Leg | None, str]:
    """Next slice for working an ETF position back to `target`, or None and why not.

    The price limit starts at break-even against the tender price (after the exit
    fee) and relaxes to `unwind_max_loss_per_unit` over `unwind_patience_ticks`
    without progress. Near the end of the heat it stops crossing the spread and
    leaves the rest to the close-out.
    """
    residual = snap.position(ETF) - target
    if abs(residual) < 1:
        return None, "at target"
    if snap.ticks_left <= cfg.unwind_stop_crossing_ticks:
        return None, f"{snap.ticks_left} ticks left: leaving {residual:+,.0f} RITC to the close-out"
    action = "SELL" if residual > 0 else "BUY"
    fee = snap.fee(ETF)
    relax = min(1.0, ticks_since_progress / max(1, cfg.unwind_patience_ticks))
    tolerance = cfg.unwind_max_loss_per_unit * relax
    if action == "SELL":
        limit = reference_price + fee - tolerance
    else:
        limit = reference_price - fee + tolerance
    book = snap.books.get(ETF)
    if book is None:
        return None, "no RITC book"
    want = min(abs(residual), cfg.unwind_slice_max, snap.max_size(ETF))
    fill = sweep_book(book.levels_for(action), want, limit_price=limit, buying=action == "BUY")
    quantity = math.floor(fill.filled)
    if quantity < 1:
        best = book.best_bid if action == "SELL" else book.best_ask
        side = "bid" if action == "SELL" else "ask"
        return None, f"waiting: best {side} {best} is past the {limit:.2f} limit ({residual:+,.0f} left)"
    leg = Leg(ETF, action, quantity, fill.vwap, fill.worst_price)
    return leg, f"{action} {quantity:,} up to {fill.worst_price:.2f} (limit {limit:.2f}, {residual:+,.0f} left)"
