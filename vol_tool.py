"""Read-only dashboard for the RITCxSmith Volatility Trading case.

It never places orders: the RitClient is created with execute=False, so any
write raises WriteBlocked. Trade manually in RIT and use this screen for fair
values, cheap/expensive options after costs, portfolio delta and the RTM hedge.

    .venv/bin/python vol_tool.py                  # practice port 16625
    .venv/bin/python vol_tool.py --port 16xxx     # competition port when announced

Each teammate runs it with their own Trader ID. Ctrl+C exits.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from typing import Any, Iterable, Mapping

import vol_model as vm
from rit_client import DEFAULT_HOST, PRACTICE_PORTS, AuthError, RitClient, RitError, prompt_credentials

CLEAR_SCREEN = "\033[2J\033[H"
NEWS_LINES = 3


def _num(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value: float | None, spec: str, width: int) -> str:
    return f"{value:{spec}}".rjust(width) if value is not None else "-".rjust(width)


def _pct(value: float | None, width: int) -> str:
    return f"{value * 100:.1f}%".rjust(width) if value is not None else "-".rjust(width)


def _underlying_price(security: Mapping) -> float | None:
    bid, ask = _num(security.get("bid")), _num(security.get("ask"))
    if bid and ask and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    last = _num(security.get("last"))
    return last if last and last > 0 else None


def _contract_multiplier(security: Mapping) -> float:
    """Shares per contract. Packet says 100; use RIT's `size`/`unit_multiplier` if either is > 1."""
    for key in ("size", "unit_multiplier"):
        value = _num(security.get(key))
        if value and value > 1:
            return value
    return float(vm.CONTRACT_MULTIPLIER)


class Dashboard:
    """Polls RIT (read-only) and renders one text screen per refresh."""

    def __init__(
        self,
        client: Any,
        *,
        min_edge: float = 0.0,
        warn_band: float = vm.DEFAULT_WARN_BAND,
        fixed_vol: float | None = None,
    ):
        self.client = client
        self.min_edge = min_edge
        self.warn_band = warn_band
        self.fixed_vol = fixed_vol  # --vol: overrides the news-based forecast
        self.total_ticks = vm.TOTAL_TICKS
        self._reset(None)

    # -- state ----------------------------------------------------------

    def _reset(self, period: Any) -> None:
        self.period = period
        self.last_tick: int | None = None
        self.last_news_id: int | None = None
        self.schedule = vm.VolSchedule(
            ticks_per_week=max(1, self.total_ticks // 4), total_ticks=self.total_ticks
        )
        self.headlines: list[str] = []
        self.news_rate: float | None = None

    def _sync_case(self, case: Mapping) -> int:
        tick = int(_num(case.get("tick")) or 0)
        total = int(_num(case.get("ticks_per_period")) or vm.TOTAL_TICKS)
        new_heat = (
            case.get("period") != self.period
            or (self.last_tick is not None and tick < self.last_tick)
            or total != self.total_ticks
        )
        if new_heat:
            self.total_ticks = total
            self._reset(case.get("period"))
        self.last_tick = tick
        return tick

    def ingest_news(self, items: Iterable[Mapping]) -> None:
        for item in sorted(items or [], key=lambda n: _num(n.get("news_id")) or 0):
            news_id = item.get("news_id")
            if news_id is not None:
                news_id = int(news_id)
                if self.last_news_id is not None and news_id <= self.last_news_id:
                    continue
                self.last_news_id = news_id
            text = " ".join(str(item.get(key) or "") for key in ("headline", "body")).strip()
            tick = int(_num(item.get("tick")) or 0)
            parsed = vm.parse_vol_news(text)
            rate = vm.parse_rate_news(text)
            notes = []
            if parsed is not None:
                week = self.schedule.apply(parsed, tick)
                vols = f"{parsed.low:.1%}-{parsed.high:.1%}" if parsed.is_range else f"{parsed.value:.1%}"
                notes.append(f"wk{week} {parsed.kind} vol {vols}")
            if rate is not None:
                self.news_rate = rate
                notes.append(f"r {rate:.2%}")
            note = f" [{', '.join(notes)}]" if notes else " [not parsed]"
            line = f"t{tick}{note}: {' '.join(text.split())}"
            self.headlines.append(line if len(line) <= 114 else line[:111] + "...")
        self.headlines = self.headlines[-NEWS_LINES:]

    # -- polling --------------------------------------------------------

    def refresh(self, now: datetime | None = None) -> str:
        case = self.client.case()
        tick = self._sync_case(case)
        # Rebuild from the full news list on every refresh, keeping only items up to the
        # current tick. Between heats the server still lists the previous heat's news, and
        # news ids restart each heat, so incremental `after=` reads go stale (seen live).
        news = self.client.news(limit=100) or []
        self.schedule = vm.VolSchedule(ticks_per_week=max(1, self.total_ticks // 4), total_ticks=self.total_ticks)
        self.headlines = []
        self.news_rate = None
        self.last_news_id = None
        self.ingest_news([item for item in news if (_num(item.get("tick")) or 0) <= tick])
        status = case.get("status")
        if status != "ACTIVE":
            return "\n".join([
                self._header(case, tick, now),
                f"Idle: case status is {status}. Waiting for the heat to start (Ctrl+C to quit).",
                *self._news_lines(),
            ])
        securities = self.client.securities()
        limits = self.client.limits()
        return self.render(case, securities, limits, now)

    # -- rendering ------------------------------------------------------

    def _header(self, case: Mapping, tick: int, now: datetime | None) -> str:
        clock = (now or datetime.now()).strftime("%H:%M:%S")
        return (
            f"RITC Volatility | {case.get('name', '?')} | period {case.get('period', '?')} | "
            f"tick {tick}/{self.total_ticks} | {case.get('status', '?')} | {clock} | read-only"
        )

    def _news_lines(self) -> list[str]:
        return ["News: " + line for line in self.headlines] or ["News: none yet"]

    def render(
        self,
        case: Mapping,
        securities: Iterable[Mapping],
        limits: Iterable[Mapping] | None,
        now: datetime | None = None,
    ) -> str:
        securities = list(securities or [])
        tick = int(_num(case.get("tick")) or 0)
        lines = [self._header(case, tick, now)]
        if self.total_ticks != vm.TOTAL_TICKS:
            lines.append(f"Note: ticks_per_period is {self.total_ticks}; the month is scaled to that length.")

        by_ticker = {sec.get("ticker"): sec for sec in securities if sec.get("ticker")}
        rtm = by_ticker.get(vm.UNDERLYING)
        s = _underlying_price(rtm) if rtm else None
        if s is None:
            lines.append(f"{vm.UNDERLYING} quote not available in /securities; cannot price options.")
            return "\n".join(lines + self._news_lines())

        ticks_per_year = 12 * self.total_ticks
        r = self.news_rate if self.news_rate is not None else vm.rate_from_securities(securities)
        t = vm.years_to_expiry(tick, self.total_ticks, ticks_per_year)
        sigma = self.fixed_vol if self.fixed_vol is not None else self.schedule.forecast(tick)
        positions = {ticker: _num(sec.get("position")) or 0.0 for ticker, sec in by_ticker.items()}
        option_secs = {ticker: sec for ticker, sec in by_ticker.items() if vm.parse_option_ticker(ticker)}
        multipliers = {ticker: _contract_multiplier(sec) for ticker, sec in option_secs.items()}

        lines.append(
            f"{vm.UNDERLYING} {_fmt(_num(rtm.get('bid')), '.2f', 0)}/{_fmt(_num(rtm.get('ask')), '.2f', 0)} "
            f"mid {s:.3f} | pos {positions.get(vm.UNDERLYING, 0):+,.0f} | r {r:.2%} | T {t:.4f}y"
        )
        if sigma is None:
            lines.append(
                "Forecast vol: no volatility news parsed yet. Fair values need it: "
                "restart with --vol 25 (the announced %) as a fallback."
            )
        elif self.fixed_vol is not None:
            lines.append(f"Forecast vol {sigma:.2%}  (set by --vol; news ignored)")
        else:
            lines.append(f"Forecast vol {sigma:.2%}  ({self.schedule.describe(tick)})")

        quotes = [(ticker, _num(sec.get("bid")), _num(sec.get("ask"))) for ticker, sec in option_secs.items()]
        signals = {}
        if sigma is not None:
            signals = {
                sig.ticker: sig
                for sig in vm.option_signals(
                    quotes, s, tick, sigma, r, min_edge=self.min_edge,
                    total_ticks=self.total_ticks, ticks_per_year=ticks_per_year,
                )
            }

        lines.append("")
        lines.append(
            f"{'TICKER':<8}{'POS':>7}{'BID':>8}{'ASK':>8}{'MKT IV':>8}{'FAIR':>9}"
            f"{'BUY $/c':>9}{'SELL $/c':>10}{'SIGNAL':>8}{'DELTA':>8}"
        )
        deltas_fcst: dict[str, float] = {}
        deltas_iv: dict[str, float] = {}
        rows = sorted(
            option_secs, key=lambda tk: (vm.parse_option_ticker(tk).strike, not vm.parse_option_ticker(tk).is_call)
        )
        for ticker in rows:
            contract = vm.parse_option_ticker(ticker)
            bid, ask = _num(option_secs[ticker].get("bid")), _num(option_secs[ticker].get("ask"))
            mid = 0.5 * (bid + ask) if bid and ask and bid > 0 and ask > 0 else None
            market_iv = vm.implied_vol(mid, contract.is_call, s, contract.strike, t, r) if mid else None
            if market_iv is not None:
                deltas_iv[ticker] = vm.bs_delta(contract.is_call, s, contract.strike, t, r, market_iv)
            sig = signals.get(ticker)
            if sig is not None:
                deltas_fcst[ticker] = sig.delta
            lines.append(
                f"{ticker:<8}{positions.get(ticker, 0):>7,.0f}{_fmt(bid, '.2f', 8)}{_fmt(ask, '.2f', 8)}"
                f"{_pct(market_iv, 8)}{_fmt(sig.fair if sig else None, '.3f', 9)}"
                f"{_fmt(sig.buy_edge if sig else None, '+.1f', 9)}{_fmt(sig.sell_edge if sig else None, '+.1f', 10)}"
                f"{(sig.signal if sig else ''):>8}{_fmt(sig.delta if sig else deltas_iv.get(ticker), '+.3f', 8)}"
            )

        lines.append("")
        lines.extend(self._delta_lines(positions, deltas_fcst, deltas_iv, multipliers))
        lines.append(self._limit_line(positions, limits))
        lines.extend(self._news_lines())
        return "\n".join(lines)

    def _delta_lines(
        self,
        positions: Mapping[str, float],
        deltas_fcst: Mapping[str, float],
        deltas_iv: Mapping[str, float],
        multipliers: Mapping[str, float],
    ) -> list[str]:
        has_options = any(abs(positions.get(tk, 0)) > 0 for tk in multipliers)
        delta_f = vm.portfolio_delta(positions, deltas_fcst, multipliers=multipliers) if deltas_fcst else None
        delta_iv = vm.portfolio_delta(positions, deltas_iv, multipliers=multipliers) if deltas_iv else None
        if delta_f is None and delta_iv is None:
            delta_f = float(positions.get(vm.UNDERLYING, 0.0))
            if has_options:
                return [f"Delta: cannot compute option deltas yet (RTM shares alone {delta_f:+,.0f})."]
        reference = delta_f if delta_f is not None else delta_iv
        worst = max(abs(d) for d in (delta_f, delta_iv) if d is not None)
        status = vm.delta_status(worst, self.warn_band)
        parts = [f"Delta {status}:"]
        if delta_f is not None:
            parts.append(f"{delta_f:+,.0f} at forecast vol")
        if delta_iv is not None:
            parts.append(f"| {delta_iv:+,.0f} at market IV")
        parts.append(f"| band ±{self.warn_band:,.0f}, CRO limit ±{vm.DELTA_LIMIT:,}")
        lines = [" ".join(parts)]

        shares = vm.hedge_shares(reference)
        if shares == 0:
            lines.append("Hedge: none needed.")
        else:
            sizes = vm.split_order(shares, vm.MAX_RTM_ORDER)
            side = "BUY" if shares > 0 else "SELL"
            split = f" in {len(sizes)} orders ({' + '.join(f'{n:,}' for n in sizes)}; max {vm.MAX_RTM_ORDER:,}/order)" if len(sizes) > 1 else ""
            lines.append(f"Hedge to zero: {side} {abs(shares):,} {vm.UNDERLYING}{split}.")
        if status == "ALERT":
            penalty = vm.delta_penalty_per_second(worst)
            lines.append(f"ALERT: beyond ±{vm.DELTA_LIMIT:,} delta. Penalty about ${penalty:,.2f} per second until hedged.")
        elif status == "WARN":
            lines.append(f"WARN: |delta| above the internal ±{self.warn_band:,.0f} band.")
        return lines

    def _limit_line(self, positions: Mapping[str, float], limits: Iterable[Mapping] | None) -> str:
        usage = vm.limit_usage(positions)
        room = usage.headroom()
        text = (
            f"Limits (packet): options gross {usage.options_gross:,.0f}/{vm.OPTION_GROSS_LIMIT:,} "
            f"(room {room['options_gross']:,.0f}), net {usage.options_net:+,.0f}/±{vm.OPTION_NET_LIMIT:,} "
            f"(room {room['options_net']:,.0f}) | RTM {usage.rtm:+,.0f}/±{vm.RTM_LIMIT:,} | "
            f"{vm.MAX_OPTION_ORDER} contracts/order"
        )
        server = []
        for entry in limits or []:
            gross, gross_limit = _num(entry.get("gross")), _num(entry.get("gross_limit"))
            net, net_limit = _num(entry.get("net")), _num(entry.get("net_limit"))
            server.append(
                f"{entry.get('name', '?')} gross {_fmt(gross, ',.0f', 0)}/{_fmt(gross_limit, ',.0f', 0)} "
                f"net {_fmt(net, ',.0f', 0)}/{_fmt(net_limit, ',.0f', 0)}"
            )
        if server:
            text += "\nLimits (server): " + " | ".join(server)
        return text


def run(dashboard: Dashboard, refresh_s: float, clear: bool) -> None:
    last_screen = ""
    while True:
        started = time.monotonic()
        try:
            last_screen = dashboard.refresh()
            screen = last_screen
        except AuthError:
            raise
        except RitError as error:
            stamp = datetime.now().strftime("%H:%M:%S")
            screen = (last_screen + "\n" if last_screen else "") + f"[{stamp}] RIT error, retrying: {error}"
        print((CLEAR_SCREEN if clear else "") + screen, flush=True)
        time.sleep(max(0.0, refresh_s - (time.monotonic() - started)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Volatility case dashboard (never places orders).")
    parser.add_argument("--host", default=DEFAULT_HOST, help="RIT server host")
    parser.add_argument("--port", type=int, default=PRACTICE_PORTS["volatility"], help="DMA port (practice 16625)")
    parser.add_argument("--refresh", type=float, default=1.0, help="seconds between refreshes")
    parser.add_argument("--min-edge", type=float, default=0.0, help="minimum $ per contract after fees to flag a signal")
    parser.add_argument("--warn-band", type=float, default=vm.DEFAULT_WARN_BAND, help="internal |delta| warning band")
    parser.add_argument("--no-clear", action="store_true", help="don't clear the screen between refreshes")
    parser.add_argument(
        "--vol", type=float, help="fallback: use this annualized vol (25 or 0.25) instead of parsing news"
    )
    args = parser.parse_args(argv)
    fixed_vol = None
    if args.vol is not None:
        fixed_vol = args.vol / 100.0 if args.vol > 1 else args.vol

    print(f"Read-only Volatility dashboard: http://{args.host}:{args.port}/v1")
    credentials = prompt_credentials()
    client = RitClient(credentials, args.port, args.host, execute=False)
    dashboard = Dashboard(client, min_edge=args.min_edge, warn_band=args.warn_band, fixed_vol=fixed_vol)
    try:
        run(dashboard, max(args.refresh, 0.2), clear=not args.no_clear and sys.stdout.isatty())
    except KeyboardInterrupt:
        print("\nStopped. No orders were placed by this tool.")
        return 0
    except AuthError as error:
        print(f"{error}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
