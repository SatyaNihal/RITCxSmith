"""Run the RIT volatility case using executable option quotes and case news."""

import argparse
import math
import time
from dataclasses import dataclass

from rit_client import DEFAULT_HOST, AuthError, RequestRejected, RitClient, RitError, prompt_credentials
from vol_tool import _contract_multiplier, _underlying_price
import vol_model as vm

LIQUIDATION_TICKS = 30
LAST_ENTRY_TICKS = 40
SIZE_INCREMENT = 25
SPOT_SCENARIOS = ((-3, .006), (-2, .061), (-1, .242), (0, .382),
                  (1, .242), (2, .061), (3, .006))
DELTA_MODEL_WEIGHTS = (.35, .35, .30)


@dataclass(frozen=True)
class Entry:
    pair: tuple[str, str]
    direction: int
    target: int
    expected: float
    offset: float
    market_order: bool
    aggressive: bool


class Session:
    def __init__(self, client, max_per_leg=500, idle_seconds=180):
        self.client = client
        self.max_per_leg = max_per_leg
        self.idle_seconds = idle_seconds
        self.close_reason_tick = None
        self.close_streak = 0
        self.closing = False
        self.traded = 0
        self.last_tick = -1
        self.last_display = None
        self.last_decision_tick = -1
        self.max_delta = 0
        self.needs_reconcile = False

    def snapshot(self, case):
        self.tick = case["tick"]
        self.total = case["ticks_per_period"]
        self.options = {}
        self.quotes = {q["ticker"]: q for q in self.client.securities()}
        self.spot = _underlying_price(self.quotes.get("RTM", {}))
        self.positions = {k: int(q["position"]) for k, q in self.quotes.items()}
        self.t = vm.years_to_expiry(self.tick, self.total, 12 * self.total)
        self.r = vm.rate_from_securities(self.quotes.values())
        self.sigma = None
        self.bounds = None
        self.week_sigma = None
        self.risk_horizon = 0
        if not self.spot:
            return
        schedule = vm.VolSchedule(ticks_per_week=max(1, self.total // 4), total_ticks=self.total)
        for item in sorted(self.client.news(limit=100), key=lambda n: n["news_id"]):
            if item["tick"] > self.tick or item.get("period") != case["period"]:
                continue
            body = item.get("body") or ""
            news = vm.parse_vol_news(body)
            if news:
                schedule.apply(news, item["tick"])
            rate = vm.parse_rate_news(body)
            if rate is not None:
                self.r = rate
        self.sigma = schedule.forecast(self.tick)
        if self.sigma is not None:
            week = vm.announcement_week(self.tick, schedule.ticks_per_week, self.total)
            self.week_sigma = vm.resolve_week_vol(week, schedule.week_vols())
            # The market maker reprices toward the upcoming week when that week's
            # forecast has arrived. Budget hedge costs through its announcement.
            next_week = week + 1
            next_open = week * schedule.ticks_per_week
            horizon = 14
            if next_week in schedule.estimates:
                horizon = max(horizon, next_open - self.tick + 10)
            self.risk_horizon = min(horizon, max(0, self.total - LIQUIDATION_TICKS - self.tick))
            self.bounds = tuple(
                vm.effective_vol(self.tick,
                    {**{k: v[i] for k, v in schedule.estimates.items()}, **schedule.actual},
                    schedule.ticks_per_week, self.total)
                for i in (0, 1)
            )
        for ticker, quote in self.quotes.items():
            option = vm.parse_option_ticker(ticker)
            if not option:
                continue
            mid = (quote["bid"] + quote["ask"]) / 2 if quote["bid"] and quote["ask"] else None
            iv = vm.implied_vol(mid, option.is_call, self.spot, option.strike, self.t, self.r) if mid else None
            self.options[ticker] = (option, iv, _contract_multiplier(quote))

    def delta(self, positions=None, spot=None):
        positions = self.positions if positions is None else positions
        spot = self.spot if spot is None else spot
        bases = [positions.get("RTM", 0)] * 3
        for ticker, (option, iv, unit) in self.options.items():
            quantity = positions[ticker]
            if quantity:
                for index, sigma in enumerate((self.sigma, iv, self.week_sigma)):
                    sigma = sigma or self.sigma or iv or 0.0
                    bases[index] += quantity * unit * vm.bs_delta(option.is_call, spot, option.strike, self.t, self.r, sigma)
        return bases

    def reconcile(self):
        for order in self.client.orders():
            try:
                self.client.cancel_order(order["order_id"])
            except RequestRejected as error:
                if error.status_code != 404:
                    raise
        if self.client.orders():
            raise RitError("An order is still open")
        self.needs_reconcile = False

    def order(self, ticker, change, entry=False):
        direction = "BUY" if change > 0 else "SELL"
        q = self.quotes[ticker]
        qty = min(abs(change), q["max_trade_size"])
        if entry and not self.cross_deep_book:
            field = "ask" if change > 0 else "bid"
            if not q[field] or q[field + "_size"] < qty:
                raise RitError(f"Insufficient displayed depth in {ticker}")
            # A small price allowance keeps both legs executable across quote updates.
            offset = self.entry_offset
            price = round(q[field] + (offset if change > 0 else -offset), q["quoted_decimals"])
            kind, price = "LIMIT", max(q["min_price"], price)
        else:
            kind, price = "MARKET", None
        self.needs_reconcile = True
        order = self.client.submit_order(ticker, direction, qty, kind, price)
        filled = int(order.get("quantity_filled") or 0)
        if order.get("status") == "OPEN":
            order_id = order["order_id"]
            order = self.client.order(order_id)
            if order.get("status") == "OPEN":
                try:
                    self.client.cancel_order(order_id)
                except RequestRejected as error:
                    if error.status_code != 404:
                        raise
                order = self.client.order(order_id)
            filled = int(order.get("quantity_filled") or 0)
            if order.get("status") == "OPEN":
                raise RitError("Order still open after cancellation")
        self.needs_reconcile = False
        self.positions[ticker] += filled if change > 0 else -filled
        self.traded += filled if ticker != "RTM" else 0
        print(f"FILL t{self.tick} {direction} {filled}/{qty} {ticker} @ {order.get('vwap')} [{order.get('order_id')}]", flush=True)
        return filled

    def hedge(self, force=False):
        delta = self.delta()
        self.max_delta = max(self.max_delta, *map(abs, delta))
        limit = 1500 if self.tick >= self.total - 90 else 2200
        shares = -round((min(delta) + max(delta)) / 2)
        if shares and (force or max(map(abs, delta)) >= limit):
            filled = self.order("RTM", shares)
            if filled == 0:
                raise RitError("No shares filled on hedge")

    def pairs(self):
        strikes = sorted({option.strike for option, _, _ in self.options.values()})
        for strike in strikes:
            pair = (f"RTM{strike:g}C", f"RTM{strike:g}P")
            if all(k in self.options and self.options[k][1] is not None for k in pair):
                yield pair

    def trade(self, changes, entry=False):
        def projected(ticker, lot):
            next_positions = self.positions.copy()
            next_positions[ticker] += lot
            return max(map(abs, self.delta(next_positions)))

        def fill(ticker, lot):
            filled = self.order(ticker, lot, entry)
            if filled != abs(lot):
                self.hedge()
                raise RitError("Partial fill, refreshing positions")
            changes[ticker] -= lot
            if changes[ticker] == 0:
                del changes[ticker]

        while changes:
            tickers = list(changes)
            quantity = min(100, *map(abs, changes.values()))
            impact = lambda ticker: projected(ticker, int(math.copysign(quantity, changes[ticker])))
            priority = min(tickers, key=impact)
            while quantity > SIZE_INCREMENT and impact(priority) > 5800:
                quantity = max(SIZE_INCREMENT, quantity - SIZE_INCREMENT)
                priority = min(tickers, key=impact)
            visit = [priority, *(k for k in tickers if k != priority)]
            for ticker in visit:
                fill(ticker, quantity if changes[ticker] > 0 else -quantity)
                if max(map(abs, self.delta())) > 6600:
                    self.hedge(force=True)
            self.hedge(force=not changes and not entry and not any(self.positions[k] for k in self.options))

    def expected_fair(self, ticker):
        option, _, _ = self.options[ticker]
        vols = (self.bounds[0], self.sigma, self.bounds[1])
        return sum(weight * vm.bs_price(option.is_call, self.spot, option.strike, self.t, self.r, vol)
                   for weight, vol in zip((.25, .50, .25), vols))

    def costs(self, pair, direction):
        """Estimate costs for each size using the unit-straddle delta scenarios."""
        stock = self.quotes[vm.UNDERLYING]
        bid, ask = stock["bid"] or self.spot, stock["ask"] or self.spot
        share_cost = stock["trading_fee"] + max(0, ask - bid) / 2
        movement = self.spot * self.week_sigma / math.sqrt(12 * self.total)
        unit = dict.fromkeys(self.positions, 0)
        unit.update(dict.fromkeys(pair, direction))
        local = self.delta(unit)
        center = (min(local) + max(local)) / 2
        future = []
        for step, probability in SPOT_SCENARIOS:
            simulated = self.delta(unit, self.spot + step * movement)
            drift = (min(simulated) + max(simulated)) / 2 - center
            future.append((probability, drift, [d - center for d in simulated]))

        def for_size(quantity):
            hedge = -quantity * center
            if abs(hedge) > vm.RTM_LIMIT:
                return math.inf
            fines = 0.0
            drift = 0.0
            for probability, move, deltas in future:
                drift += probability * abs(move * quantity)
                fines += probability * sum(
                    weight * vm.delta_penalty_per_second(delta * quantity)
                    for weight, delta in zip(DELTA_MODEL_WEIGHTS, deltas)
                )
            current_shares = self.positions.get(vm.UNDERLYING, 0)
            execution = (abs(hedge - current_shares) + abs(hedge)) * share_cost
            return execution + self.risk_horizon * (fines + .65 * drift * share_cost)

        return for_size

    def candidate(self, pair, current=0):
        if any(not self.quotes[k]["bid"] or not self.quotes[k]["ask"] for k in pair):
            return None
        gaps = [self.sigma - self.options[k][1] for k in pair]
        direction = 1 if sum(gaps) > 0 else -1
        if min(direction * gap for gap in gaps) < .03:
            return None
        if current and self.positions[pair[0]] * direction <= 0:
            return None

        edges = []
        for ticker in pair:
            _, _, unit = self.options[ticker]
            q = self.quotes[ticker]
            bid, ask = q["bid"], q["ask"]
            price = ask if direction > 0 else bid
            edges.append(unit * direction * (self.expected_fair(ticker) - price) - q["trading_fee"])
        if min(edges) <= 0:
            return None
        aggressive = min(edges) >= 20
        offset = .02 if min(edges) >= 12 else .01
        market_order = min(edges) >= 12 and all(
            0 < self.quotes[k]["ask"] - self.quotes[k]["bid"] <= .04
            and min(self.quotes[k]["bid_size"], self.quotes[k]["ask_size"]) >= 1000
            for k in pair
        )
        max_size = self.max_per_leg if min(edges) >= 10 else 100
        costs = self.costs(pair, direction)
        baseline = costs(current)
        round_trip_edge = sum(
            edge - self.quotes[k]["trading_fee"]
            - self.options[k][2] * ((self.quotes[k]["ask"] - self.quotes[k]["bid"]) / 2 + offset)
            for k, edge in zip(pair, edges)
        )
        best = None
        best_value = 0
        for target in range(current + SIZE_INCREMENT, max_size + 1, SIZE_INCREMENT):
            expected = (target - current) * round_trip_edge - (costs(target) - baseline)
            if expected > best_value:
                best_value = expected
                best = Entry(pair, direction, target, expected, offset, market_order, aggressive)
        return best

    def repair(self, held):
        if len(held) == 2:
            first, second = held
            left, right = held.values()
            contracts = self.options[first][0], self.options[second][0]
            if (left * right > 0 and contracts[0].strike == contracts[1].strike
                    and contracts[0].is_call != contracts[1].is_call):
                if left == right:
                    return False
                excess = first if abs(left) > abs(right) else second
                self.trade({excess: -int(math.copysign(abs(left - right), held[excess]))})
                return True
        self.trade({k: -q for k, q in held.items()})
        return True

    def should_close(self, pair, signed_quantity):
        direction = 1 if signed_quantity > 0 else -1
        gaps = [self.sigma - self.options[k][1] for k in pair]
        if direction * sum(gaps) <= 0:
            return "forecast crossed market"
        size = abs(signed_quantity)
        per_pair = sum(
            self.options[k][2] * direction * (
                self.expected_fair(k) - (self.quotes[k]["bid"] + self.quotes[k]["ask"]) / 2
            ) for k in pair
        )
        costs = self.costs(pair, direction)
        expected_hold = size * per_pair - (costs(size) - costs(0))
        if self.close_reason_tick != self.tick:
            self.close_streak = self.close_streak + 1 if expected_hold <= 0 else 0
            self.close_reason_tick = self.tick
        if self.close_streak >= 3:
            return "remaining edge fell below modeled holding costs"
        return None

    def act(self):
        held = {k: self.positions[k] for k in self.options if self.positions[k]}
        if self.tick >= self.total - LIQUIDATION_TICKS or self.closing:
            self.trade({k: -q for k, q in held.items()})
            self.hedge(force=True)
            self.closing = bool(any(self.positions.values()))
            return
        if self.tick % 5 == 0 and self.last_display != self.tick:
            self.last_display = self.tick
            forecast = f"{self.sigma:.2%}" if self.sigma is not None else "unknown"
            print(f"STATE t{self.tick} RTM {self.spot:.3f} forecast {forecast} "
                  f"positions {held} hedge {self.positions.get(vm.UNDERLYING, 0)}", flush=True)
        self.hedge(force=bool(self.positions.get(vm.UNDERLYING)) and not held)
        if held and self.repair(held):
            self.last_decision_tick = self.tick
            return
        if not self.sigma:
            return
        pairs = [tuple(held)] if held else list(self.pairs())
        if held:
            pair = pairs[0]
            if all(self.options[k][1] is not None for k in pair):
                reason = self.should_close(pair, next(iter(held.values())))
                if reason:
                    print(f"EXIT t{self.tick} {reason}", flush=True)
                    self.closing = True
                    self.close_streak = 0
                    self.last_decision_tick = self.tick
                    self.trade({k: -q for k, q in held.items()})
                    self.closing = bool(any(self.positions.values()))
                    return
        if self.tick >= self.total - LAST_ENTRY_TICKS or self.last_decision_tick == self.tick:
            return
        current = abs(next(iter(held.values()), 0))
        entries = [self.candidate(pair, current) for pair in pairs
                   if all(self.options[k][1] is not None for k in pair)]
        best = max((entry for entry in entries if entry), key=lambda entry: entry.expected, default=None)
        if best is None:
            return
        self.entry_offset = best.offset
        self.cross_deep_book = best.market_order
        print(f"ENTRY t{self.tick} {'BUY' if best.direction > 0 else 'SELL'} up to {best.target} "
              f"{best.pair} projected net {best.expected:,.0f}", flush=True)
        self.last_decision_tick = self.tick
        self.close_streak = 0
        batch = min(400 if best.aggressive else 200, best.target - current)
        self.trade(dict.fromkeys(best.pair, best.direction * batch), entry=True)

    def run(self):
        start = self.client.case()
        print("WATCH", start, self.client.trader(), flush=True)
        if self.client.orders():
            raise SystemExit("Account has an outstanding order")
        seen_active = False
        deadline = time.monotonic() + self.idle_seconds
        while True:
            then = time.monotonic()
            try:
                case = self.client.case()
                tick = case["tick"]
                if "volatility" not in case["name"].lower():
                    print("STOP different case", case, flush=True)
                    return
                if seen_active and (case["status"] == "STOPPED" or tick < self.last_tick):
                    print("FINAL", case, self.client.trader(), self.client.limits(),
                          {q["ticker"]: q["position"] for q in self.client.securities() if q["position"]},
                          "option contracts filled", self.traded, "max observed delta", self.max_delta, flush=True)
                    seen_active = False
                    self.last_decision_tick = self.last_display = -1
                    self.max_delta = self.traded = 0
                    self.closing = False
                    self.close_streak = 0
                    self.close_reason_tick = None
                    deadline = time.monotonic() + self.idle_seconds
                if case["status"] != "ACTIVE":
                    if not seen_active and time.monotonic() >= deadline:
                        print("STOP no additional volatility heat started", flush=True)
                        return
                    time.sleep(.65)
                    continue
                if not seen_active:
                    print("START", case, self.client.trader(), flush=True)
                    if self.client.orders():
                        raise SystemExit("Account has an outstanding order")
                seen_active = True
                self.last_tick = tick
                if self.needs_reconcile:
                    self.reconcile()
                    time.sleep(.3)
                self.snapshot(case)
                if self.spot and self.options:
                    self.act()
            except AuthError:
                raise
            except RitError as error:
                print("RIT", error, flush=True)
            time.sleep(max(.08, .32 - (time.monotonic() - then)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=16520)
    parser.add_argument("--idle-seconds", type=int, default=180,
                        help="How long to wait for the heat to start")
    parser.add_argument("--max-per-leg", type=int, choices=(100, 200, 300, 400, 500), default=500)
    parser.add_argument("--execute", action="store_true", help="Send simulated orders")
    args = parser.parse_args()
    credentials = prompt_credentials()
    with RitClient(credentials, args.port, args.host, execute=args.execute, timeout_s=1.5) as client:
        session = Session(client, args.max_per_leg, args.idle_seconds)
        if args.execute:
            if input(f"Type EXECUTE to trade on {client.base_url}: ").strip() != "EXECUTE":
                return
            session.run()
        else:
            case = client.case()
            print(case, client.trader())
            if case["status"] == "ACTIVE":
                session.snapshot(case)
                if session.sigma:
                    for pair in session.pairs():
                        print(session.candidate(pair))


if __name__ == "__main__":
    main()
