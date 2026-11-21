"""Pricing and risk math for the RITCxSmith Volatility Trading case.

Pure functions only: no HTTP, no printing. `vol_tool.py` feeds these with data
read from RIT. Case facts used here (Volatility case packet, pp. 2-4):

- 300 ticks = one month = 20 trading days of a 240-day year, so one tick is
  1/3600 of a year and T = (300 - tick) / 3600.
- Four weeks of 75 ticks (1-75, 76-150, 151-225, 226-300). The realized vol
  is announced at the start of each week; an estimate (a range) for next week
  is announced mid-week.
- Options are European, cash-settled, 100 shares per contract, $2.00 fee per
  contract. Delta must stay within +/-7,000 shares; every second beyond that
  costs (|delta| - 7,000) x $0.10.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

UNDERLYING = "RTM"
TOTAL_TICKS = 300
TICKS_PER_YEAR = 3600
TICKS_PER_WEEK = 75
CONTRACT_MULTIPLIER = 100
OPTION_FEE_PER_CONTRACT = 2.00
RTM_FEE_PER_SHARE = 0.02

DELTA_LIMIT = 7_000
DELTA_FINE_PER_SHARE_SECOND = 0.10
DEFAULT_WARN_BAND = 5_000

OPTION_GROSS_LIMIT = 2_500  # contracts
OPTION_NET_LIMIT = 1_000  # contracts
RTM_LIMIT = 50_000  # shares, gross and net
MAX_OPTION_ORDER = 100  # contracts per order
MAX_RTM_ORDER = 10_000  # shares per order

MIN_YEARS = 1e-6  # floor for T at or after expiry
IV_LOW = 1e-4
IV_HIGH = 5.0


# ---------------------------------------------------------------- Black-Scholes


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def years_to_expiry(tick: float, total_ticks: int = TOTAL_TICKS, ticks_per_year: int = TICKS_PER_YEAR) -> float:
    """Time to expiry in years, floored at a tiny positive value near expiry."""
    return max((total_ticks - tick) / ticks_per_year, MIN_YEARS)


def _discounted_strike(k: float, t: float, r: float) -> float:
    return k * math.exp(-r * max(t, 0.0))


def _d1_d2(s: float, k: float, t: float, r: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _check_inputs(s: float, k: float) -> None:
    if s <= 0 or k <= 0:
        raise ValueError(f"underlying and strike must be positive (S={s}, K={k})")


def bs_price(is_call: bool, s: float, k: float, t: float, r: float, sigma: float) -> float:
    """European option price (no dividends)."""
    _check_inputs(s, k)
    if t <= 0 or sigma <= 0:
        pv_k = _discounted_strike(k, t, r)
        return max(0.0, s - pv_k) if is_call else max(0.0, pv_k - s)
    d1, d2 = _d1_d2(s, k, t, r, sigma)
    pv_k = _discounted_strike(k, t, r)
    if is_call:
        return s * norm_cdf(d1) - pv_k * norm_cdf(d2)
    return pv_k * norm_cdf(-d2) - s * norm_cdf(-d1)


def bs_delta(is_call: bool, s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Delta per share of underlying (call in [0, 1], put in [-1, 0])."""
    _check_inputs(s, k)
    if t <= 0 or sigma <= 0:
        pv_k = _discounted_strike(k, t, r)
        call_delta = 1.0 if s > pv_k else (0.5 if s == pv_k else 0.0)
    else:
        d1, _ = _d1_d2(s, k, t, r, sigma)
        call_delta = norm_cdf(d1)
    return call_delta if is_call else call_delta - 1.0


def price_bounds(is_call: bool, s: float, k: float, t: float, r: float) -> tuple[float, float]:
    """No-arbitrage (lower, upper) bounds for a European option price."""
    pv_k = _discounted_strike(k, t, r)
    if is_call:
        return max(0.0, s - pv_k), s
    return max(0.0, pv_k - s), pv_k


def implied_vol(
    price: float,
    is_call: bool,
    s: float,
    k: float,
    t: float,
    r: float = 0.0,
    *,
    low: float = IV_LOW,
    high: float = IV_HIGH,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float | None:
    """Implied vol by bisection, or None if the price has no IV in [low, high]."""
    if price is None or not math.isfinite(price) or t <= MIN_YEARS:
        return None
    _check_inputs(s, k)
    lower, upper = price_bounds(is_call, s, k, t, r)
    if price <= lower or price >= upper:
        return None
    if bs_price(is_call, s, k, t, r, low) > price or bs_price(is_call, s, k, t, r, high) < price:
        return None
    lo, hi = low, high
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if bs_price(is_call, s, k, t, r, mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- tickers


@dataclass(frozen=True)
class OptionContract:
    ticker: str
    underlying: str
    strike: float
    is_call: bool

    @property
    def kind(self) -> str:
        return "C" if self.is_call else "P"


_OPTION_TICKER = re.compile(r"^([A-Z]+?)(\d+(?:\.\d+)?)([CP])$")


def parse_option_ticker(ticker: str) -> OptionContract | None:
    """'RTM48C' -> strike 48 call; 'RTM52P' -> strike 52 put; 'RTM' -> None."""
    match = _OPTION_TICKER.match((ticker or "").strip().upper())
    if not match:
        return None
    underlying, strike, kind = match.groups()
    return OptionContract(ticker.strip().upper(), underlying, float(strike), kind == "C")


# ---------------------------------------------------------------- news


@dataclass(frozen=True)
class VolNews:
    """A parsed volatility announcement. Vols are decimals (0.20 = 20%)."""

    kind: str  # "current" (this week's realized vol) or "next" (estimate for next week)
    low: float
    high: float

    @property
    def value(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def is_range(self) -> bool:
        return self.high != self.low


_NUMBER = r"(\d+(?:\.\d+)?)"
_RANGE = re.compile(_NUMBER + r"\s*%?\s*(?:-|–|—|~|to|and)\s*" + _NUMBER + r"\s*%")
_SINGLE = re.compile(_NUMBER + r"\s*%")
_KEYWORD = re.compile(r"volatil|\brates?\b")
_NEXT_WEEK = re.compile(r"\bnext\s+week\b")


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _labelled_percentages(lowered: str) -> list[tuple[str, float, float, str]]:
    """Each percentage (or range) left to right, labelled by the nearest keyword
    before it: ("volatil" | "rate" | "", low, high, text before it).

    Live news puts the rate first: "Risk free rate and current annualized
    volatility of RTM. The current risk free rate is 0% and the current
    annualized volatility is 25%". The 0% follows "rate"; the 25% follows
    "volatility", so each number gets the right meaning.
    """
    tokens = []
    position = 0
    while True:
        matches = [m for m in (_RANGE.search(lowered, position), _SINGLE.search(lowered, position)) if m]
        if not matches:
            return tokens
        # Earliest match; at the same start, the longer one (a range beats its first number).
        match = min(matches, key=lambda m: (m.start(), -(m.end() - m.start())))
        if match.re is _RANGE:
            low, high = sorted((float(match.group(1)), float(match.group(2))))
        else:
            low = high = float(match.group(1))
        before = lowered[position:match.start()]
        keywords = _KEYWORD.findall(before)
        label = ("volatil" if keywords[-1].startswith("volatil") else "rate") if keywords else ""
        tokens.append((label, low, high, before))
        position = match.end()


def parse_vol_news(text: str) -> VolNews | None:
    """Parse a volatility announcement; None if the text has no volatility figure.

    Handles the live format ("current annualized volatility is 25%", rate
    mentioned first), "...for this week will be 20%" and "...for next week
    will be between 27-30%" (also 27% - 30%, 27~30%, 27–30%, 27 to 30%).
    A figure with no week stated is treated as the current one.
    """
    lowered = _normalize(text)
    if "volatil" not in lowered:
        return None
    for label, low, high, before in _labelled_percentages(lowered):
        if label != "volatil":
            continue
        if not (0 < low <= high < 500):
            return None
        names_current = re.search(r"\b(this|current)\b", lowered) is not None
        is_next = _NEXT_WEEK.search(before) is not None or (
            _NEXT_WEEK.search(lowered) is not None and not names_current
        )
        return VolNews("next" if is_next else "current", low / 100.0, high / 100.0)
    return None


def parse_rate_news(text: str) -> float | None:
    """Risk-free rate as a decimal from a news item ("risk free rate is 0%" -> 0.0), else None."""
    for label, low, _, _ in _labelled_percentages(_normalize(text)):
        if label == "rate":
            return low / 100.0
    return None


# ---------------------------------------------------------------- vol schedule


def week_of(tick: int, ticks_per_week: int = TICKS_PER_WEEK, total_ticks: int = TOTAL_TICKS) -> int:
    """Week number (1-based) containing `tick`; tick 0 counts as week 1."""
    weeks = max(1, math.ceil(total_ticks / ticks_per_week))
    return min(weeks, max(1, math.ceil(tick / ticks_per_week)))


def announcement_week(news_tick: int, ticks_per_week: int = TICKS_PER_WEEK, total_ticks: int = TOTAL_TICKS) -> int:
    """Week a 'this week' announcement at `news_tick` refers to.

    Uses the week of the *next* tick, so news posted on a boundary tick (e.g.
    tick 75) counts toward the week that is starting (week 2). Confirm against
    the practice server's news timing.
    """
    return week_of(news_tick + 1, ticks_per_week, total_ticks)


def remaining_ticks_by_week(
    tick: int, ticks_per_week: int = TICKS_PER_WEEK, total_ticks: int = TOTAL_TICKS
) -> dict[int, int]:
    """How many of the remaining price steps (tick+1 .. total) fall in each week."""
    counts: dict[int, int] = {}
    for step in range(max(int(tick), 0) + 1, total_ticks + 1):
        week = week_of(step, ticks_per_week, total_ticks)
        counts[week] = counts.get(week, 0) + 1
    return counts


def effective_vol(
    tick: int,
    week_vols: Mapping[int, float],
    ticks_per_week: int = TICKS_PER_WEEK,
    total_ticks: int = TOTAL_TICKS,
) -> float | None:
    """Variance-weighted vol over the option's remaining life.

    `week_vols` maps week -> vol (decimal). Weeks with no value carry forward the
    latest earlier known week (or the earliest known week if none earlier).
    Returns None if no vol is known at all.
    """
    if not week_vols:
        return None
    counts = remaining_ticks_by_week(tick, ticks_per_week, total_ticks)
    if not counts:
        return resolve_week_vol(week_of(tick, ticks_per_week, total_ticks), week_vols)
    total = sum(counts.values())
    variance = sum(n * resolve_week_vol(week, week_vols) ** 2 for week, n in counts.items()) / total
    return math.sqrt(variance)


def resolve_week_vol(week: int, week_vols: Mapping[int, float]) -> float:
    """Vol for `week`: its own value, else the latest earlier one, else the earliest known."""
    if week in week_vols:
        return week_vols[week]
    earlier = [w for w in week_vols if w < week]
    if earlier:
        return week_vols[max(earlier)]
    return week_vols[min(week_vols)]


@dataclass
class VolSchedule:
    """Announced vols by week, built from news. Actual values beat estimates."""

    actual: dict[int, float] = field(default_factory=dict)
    estimates: dict[int, tuple[float, float]] = field(default_factory=dict)
    ticks_per_week: int = TICKS_PER_WEEK
    total_ticks: int = TOTAL_TICKS

    def apply(self, news: VolNews, news_tick: int) -> int:
        """Record a parsed announcement; returns the week it applies to."""
        week = announcement_week(news_tick, self.ticks_per_week, self.total_ticks)
        if news.kind == "current":
            self.actual[week] = news.value
            return week
        target = week + 1
        self.estimates[target] = (news.low, news.high)
        return target

    def week_vols(self) -> dict[int, float]:
        vols = {week: 0.5 * (low + high) for week, (low, high) in self.estimates.items()}
        vols.update(self.actual)
        return vols

    def forecast(self, tick: int) -> float | None:
        return effective_vol(tick, self.week_vols(), self.ticks_per_week, self.total_ticks)

    def describe(self, tick: int) -> str:
        """Short human-readable source of the forecast, e.g. 'wk1 20.0% x38, wk2 28.5% est x75'."""
        vols = self.week_vols()
        if not vols:
            return "no volatility news yet"
        parts = []
        for week, count in sorted(remaining_ticks_by_week(tick, self.ticks_per_week, self.total_ticks).items()):
            vol = resolve_week_vol(week, vols)
            if week in self.actual:
                tag = ""
            elif week in self.estimates:
                low, high = self.estimates[week]
                tag = f" est {low:.0%}-{high:.0%}"
            else:
                tag = " carried"
            parts.append(f"wk{week} {vol:.1%}{tag} x{count}")
        return ", ".join(parts)


# ---------------------------------------------------------------- signals and risk


@dataclass(frozen=True)
class OptionSignal:
    ticker: str
    strike: float
    is_call: bool
    bid: float | None
    ask: float | None
    market_iv: float | None
    fair: float
    delta: float
    buy_edge: float | None  # $ per contract after fee: (fair - ask) x 100 - fee
    sell_edge: float | None  # $ per contract after fee: (bid - fair) x 100 - fee
    signal: str  # "BUY", "SELL" or ""


def _valid_quote(value: float | None) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def option_signals(
    quotes: Iterable[tuple[str, float | None, float | None]],
    s: float,
    tick: int,
    sigma: float,
    r: float = 0.0,
    *,
    min_edge: float = 0.0,
    fee_per_contract: float = OPTION_FEE_PER_CONTRACT,
    multiplier: int = CONTRACT_MULTIPLIER,
    hedge_cost_per_share: float = 0.0,
    total_ticks: int = TOTAL_TICKS,
    ticks_per_year: int = TICKS_PER_YEAR,
) -> list[OptionSignal]:
    """Fair value at `sigma` vs the executable side for each (ticker, bid, ask).

    BUY when (fair - ask) x 100 - fee > min_edge, SELL when
    (bid - fair) x 100 - fee > min_edge. `hedge_cost_per_share` optionally
    charges |delta| x 100 shares of RTM hedging cost per contract (default 0).
    Tickers that aren't options are skipped. Sorted by strike, calls first.
    """
    t = years_to_expiry(tick, total_ticks, ticks_per_year)
    signals: list[OptionSignal] = []
    for ticker, raw_bid, raw_ask in quotes:
        contract = parse_option_ticker(ticker)
        if contract is None:
            continue
        bid, ask = _valid_quote(raw_bid), _valid_quote(raw_ask)
        fair = bs_price(contract.is_call, s, contract.strike, t, r, sigma)
        delta = bs_delta(contract.is_call, s, contract.strike, t, r, sigma)
        cost = fee_per_contract + abs(delta) * multiplier * hedge_cost_per_share
        buy_edge = (fair - ask) * multiplier - cost if ask is not None else None
        sell_edge = (bid - fair) * multiplier - cost if bid is not None else None
        mid = 0.5 * (bid + ask) if bid is not None and ask is not None else None
        market_iv = implied_vol(mid, contract.is_call, s, contract.strike, t, r) if mid is not None else None
        signal = ""
        if buy_edge is not None and buy_edge > min_edge and (sell_edge is None or buy_edge >= sell_edge):
            signal = "BUY"
        elif sell_edge is not None and sell_edge > min_edge:
            signal = "SELL"
        signals.append(
            OptionSignal(contract.ticker, contract.strike, contract.is_call, bid, ask, market_iv,
                         fair, delta, buy_edge, sell_edge, signal)
        )
    signals.sort(key=lambda sig: (sig.strike, not sig.is_call))
    return signals


def portfolio_delta(
    positions: Mapping[str, float],
    deltas: Mapping[str, float],
    *,
    underlying: str = UNDERLYING,
    multipliers: Mapping[str, float] | None = None,
) -> float:
    """Shares of delta: RTM position + sum(contracts x multiplier x delta)."""
    total = float(positions.get(underlying, 0.0) or 0.0)
    for ticker, delta in deltas.items():
        if ticker == underlying:
            continue
        contracts = float(positions.get(ticker, 0.0) or 0.0)
        multiplier = (multipliers or {}).get(ticker) or CONTRACT_MULTIPLIER
        total += contracts * multiplier * delta
    return total


def hedge_shares(delta: float) -> int:
    """RTM shares to trade to bring delta to zero (positive = buy)."""
    return -int(round(delta))


def split_order(quantity: float, max_order: int = MAX_RTM_ORDER) -> list[int]:
    """Split |quantity| into order sizes no larger than max_order."""
    remaining = abs(int(round(quantity)))
    sizes = []
    while remaining > 0:
        size = min(remaining, max_order)
        sizes.append(size)
        remaining -= size
    return sizes


def delta_status(delta: float, warn_band: float = DEFAULT_WARN_BAND, limit: float = DELTA_LIMIT) -> str:
    """'OK', 'WARN' (beyond the internal band) or 'ALERT' (beyond the CRO limit)."""
    magnitude = abs(delta)
    if magnitude > limit:
        return "ALERT"
    if magnitude > warn_band:
        return "WARN"
    return "OK"


def delta_penalty_per_second(delta: float, limit: float = DELTA_LIMIT,
                             fine: float = DELTA_FINE_PER_SHARE_SECOND) -> float:
    """CRO fine for one second at this delta: (|delta| - limit) x $0.10, floored at 0."""
    return max(0.0, abs(delta) - limit) * fine


@dataclass(frozen=True)
class LimitUsage:
    options_gross: float
    options_net: float
    rtm: float

    def headroom(self) -> dict[str, float]:
        return {
            "options_gross": OPTION_GROSS_LIMIT - self.options_gross,
            "options_net": OPTION_NET_LIMIT - abs(self.options_net),
            "rtm": RTM_LIMIT - abs(self.rtm),
        }


def limit_usage(positions: Mapping[str, float], underlying: str = UNDERLYING) -> LimitUsage:
    """Contracts and shares counted against the case's packet limits."""
    gross = net = 0.0
    for ticker, position in positions.items():
        if parse_option_ticker(ticker) is None:
            continue
        position = float(position or 0.0)
        gross += abs(position)
        net += position
    return LimitUsage(gross, net, float(positions.get(underlying, 0.0) or 0.0))


def normalize_rate(value: float | None) -> float:
    """Interest rate as a decimal.

    RIT's `interest_rate` units are unconfirmed for this case. Guess: values
    above 0.20 in magnitude are percents (2 -> 0.02); values at or below 0.20
    are already decimals (0.02 -> 0.02). Missing/invalid -> 0 (the packet's
    start-up assumption). Confirm on the practice server.
    """
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(rate):
        return 0.0
    return rate / 100.0 if abs(rate) > 0.20 else rate


def rate_from_securities(securities: Iterable[Mapping], underlying: str = UNDERLYING) -> float:
    """Pick the underlying's interest_rate (else the first non-zero one) and normalize it."""
    securities = list(securities)
    for security in securities:
        if security.get("ticker") == underlying and security.get("interest_rate") is not None:
            return normalize_rate(security.get("interest_rate"))
    for security in securities:
        if security.get("interest_rate"):
            return normalize_rate(security.get("interest_rate"))
    return 0.0
