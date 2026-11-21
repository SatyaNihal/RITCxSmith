"""Trading console for the Volatility case: you type a command, it sends the orders.

    .venv/bin/python vol_trade.py --port 16520

Commands (Market orders; options go out in 100-contract lots, calls and puts
alternating so delta stays balanced while they fill):
    p              show positions, forecast vol and delta
    h              hedge: trade RTM so delta is about zero
    b 50 100       BUY 100 RTM50C and 100 RTM50P (a straddle)
    s 50 100       SELL 100 RTM50C and 100 RTM50P
    c              close every option position
    q              quit

Nothing is sent until you type a command. Delta for hedging is the midpoint of
the forecast-vol delta and the market-IV delta, so both stay inside the limit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

import vol_model as vm
from rit_client import DEFAULT_HOST, AuthError, RitClient, RitError, SubmissionUncertain, prompt_credentials
from vol_tool import _contract_multiplier, _num, _underlying_price

COMPETITION_PORT = 16520
LOT = vm.MAX_OPTION_ORDER  # 100 contracts per order


@dataclass
class State:
    tick: int
    status: str
    spot: float | None
    sigma: float | None
    positions: dict[str, float]
    delta_forecast: float | None
    delta_market: float | None
    option_tickers: list[str]

    @property
    def hedge_delta(self) -> float | None:
        values = [d for d in (self.delta_forecast, self.delta_market) if d is not None]
        return sum(values) / len(values) if values else None

    def text(self) -> str:
        options = ", ".join(
            f"{t} {self.positions[t]:+,.0f}" for t in self.option_tickers if self.positions.get(t)
        ) or "no options"
        sigma = f"{self.sigma:.1%}" if self.sigma is not None else "unknown"
        df = f"{self.delta_forecast:+,.0f}" if self.delta_forecast is not None else "-"
        dm = f"{self.delta_market:+,.0f}" if self.delta_market is not None else "-"
        return (
            f"tick {self.tick} {self.status} | RTM {self.spot or 0:.2f} pos {self.positions.get(vm.UNDERLYING, 0):+,.0f} | "
            f"{options} | forecast vol {sigma} | delta {df} (forecast) {dm} (market)"
        )


def read_state(client: Any) -> State:
    case = client.case()
    tick = int(_num(case.get("tick")) or 0)
    total = int(_num(case.get("ticks_per_period")) or vm.TOTAL_TICKS)
    securities = list(client.securities() or [])
    news = client.news(limit=100) or []
    schedule = vm.VolSchedule(ticks_per_week=max(1, total // 4), total_ticks=total)
    rate = None
    for item in sorted(news, key=lambda n: _num(n.get("news_id")) or 0):
        news_tick = int(_num(item.get("tick")) or 0)
        if news_tick > tick:
            continue  # left over from a previous heat
        text = " ".join(str(item.get(key) or "") for key in ("headline", "body"))
        parsed = vm.parse_vol_news(text)
        if parsed is not None:
            schedule.apply(parsed, news_tick)
        parsed_rate = vm.parse_rate_news(text)
        if parsed_rate is not None:
            rate = parsed_rate
    sigma = schedule.forecast(tick)
    by_ticker = {sec.get("ticker"): sec for sec in securities if sec.get("ticker")}
    spot = _underlying_price(by_ticker.get(vm.UNDERLYING, {}))
    r = rate if rate is not None else vm.rate_from_securities(securities)
    t = vm.years_to_expiry(tick, total, 12 * total)
    positions = {ticker: _num(sec.get("position")) or 0.0 for ticker, sec in by_ticker.items()}
    options = sorted(
        (tk for tk in by_ticker if vm.parse_option_ticker(tk)),
        key=lambda tk: (vm.parse_option_ticker(tk).strike, not vm.parse_option_ticker(tk).is_call),
    )
    deltas_f: dict[str, float] = {}
    deltas_m: dict[str, float] = {}
    if spot:
        for ticker in options:
            contract = vm.parse_option_ticker(ticker)
            sec = by_ticker[ticker]
            if sigma is not None:
                deltas_f[ticker] = vm.bs_delta(contract.is_call, spot, contract.strike, t, r, sigma)
            bid, ask = _num(sec.get("bid")), _num(sec.get("ask"))
            iv = None
            if bid and ask and bid > 0 and ask > 0:
                iv = vm.implied_vol(0.5 * (bid + ask), contract.is_call, spot, contract.strike, t, r)
            if iv is not None:
                deltas_m[ticker] = vm.bs_delta(contract.is_call, spot, contract.strike, t, r, iv)
            elif ticker in deltas_f:
                deltas_m[ticker] = deltas_f[ticker]
            else:  # no vol known: intrinsic delta
                deltas_m[ticker] = vm.bs_delta(contract.is_call, spot, contract.strike, t, r, 0.0)
    multipliers = {tk: _contract_multiplier(by_ticker[tk]) for tk in options}
    delta_f = vm.portfolio_delta(positions, deltas_f, multipliers=multipliers) if deltas_f else None
    delta_m = vm.portfolio_delta(positions, deltas_m, multipliers=multipliers) if deltas_m else None
    return State(tick, str(case.get("status")), spot, sigma, positions, delta_f, delta_m, options)


def _send(client: Any, ticker: str, action: str, quantity: int, out=print) -> bool:
    try:
        order = client.submit_order(ticker, action, quantity, "MARKET")
        out(f"  sent {action} {quantity:,} {ticker}: filled {order.get('quantity_filled')} @ {order.get('vwap')}")
        return True
    except SubmissionUncertain as error:
        out(f"  {action} {quantity:,} {ticker}: NO REPLY ({error}). Check RIT before resending.")
    except RitError as error:
        out(f"  {action} {quantity:,} {ticker}: REJECTED ({error})")
    return False


def _lots(quantity: int) -> list[int]:
    return vm.split_order(quantity, LOT)


def straddle_orders(strike: str, quantity: int, action: str) -> list[tuple[str, str, int]]:
    """Calls and puts in alternating 100-lots."""
    calls = [(f"RTM{strike}C", action, q) for q in _lots(quantity)]
    puts = [(f"RTM{strike}P", action, q) for q in _lots(quantity)]
    orders = []
    for index in range(max(len(calls), len(puts))):
        orders += calls[index:index + 1] + puts[index:index + 1]
    return orders


def close_orders(state: State) -> list[tuple[str, str, int]]:
    """Every option position reversed, interleaving tickers so delta stays balanced."""
    queues = []
    for ticker in state.option_tickers:
        position = int(round(state.positions.get(ticker, 0)))
        if position:
            action = "SELL" if position > 0 else "BUY"
            queues.append([(ticker, action, q) for q in _lots(abs(position))])
    orders = []
    while any(queues):
        for queue in queues:
            if queue:
                orders.append(queue.pop(0))
    return orders


def hedge_orders(state: State) -> list[tuple[str, str, int]]:
    delta = state.hedge_delta
    if delta is None:
        return []
    shares = vm.hedge_shares(delta)
    if abs(shares) < 100:
        return []
    action = "BUY" if shares > 0 else "SELL"
    return [(vm.UNDERLYING, action, q) for q in vm.split_order(shares, vm.MAX_RTM_ORDER)]


def run_command(client: Any, line: str, out=print) -> bool:
    """Execute one command line. Returns False to quit."""
    parts = line.strip().lower().split()
    if not parts:
        return True
    command = parts[0]
    if command in ("q", "quit", "exit"):
        return False
    state = read_state(client)
    if command == "p":
        out(state.text())
        return True
    if command == "h":
        orders = hedge_orders(state)
        if not orders:
            out(f"  no hedge needed ({state.text()})")
    elif command == "c":
        orders = close_orders(state)
        if not orders:
            out("  no option positions to close")
    elif command in ("b", "s") and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        strike, quantity = parts[1], int(parts[2])
        if f"RTM{strike}C".upper() not in {t.upper() for t in state.option_tickers}:
            out(f"  no RTM{strike} options in this case")
            return True
        if not 0 < quantity <= 1000:
            out("  quantity must be 1 to 1,000 contracts")
            return True
        orders = straddle_orders(strike, quantity, "BUY" if command == "b" else "SELL")
    else:
        out("  commands: p | h | b <strike> <qty> | s <strike> <qty> | c | q")
        return True
    for ticker, action, quantity in orders:
        if not _send(client, ticker, action, quantity, out):
            out("  stopped: fix this before sending more")
            break
    out(read_state(client).text())
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=COMPETITION_PORT)
    args = parser.parse_args(argv)
    print(__doc__)
    credentials = prompt_credentials()
    client = RitClient(credentials, args.port, args.host, execute=True)
    if input(f"Orders will be sent to {client.base_url}. Type EXECUTE to continue: ").strip() != "EXECUTE":
        print("Not confirmed. Nothing sent.")
        return 1
    try:
        print(read_state(client).text())
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            try:
                if not run_command(client, line):
                    break
            except AuthError:
                raise
            except RitError as error:
                print(f"  RIT error: {error}")
    except KeyboardInterrupt:
        print("\nStopped.")
    except AuthError as error:
        print(error)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
