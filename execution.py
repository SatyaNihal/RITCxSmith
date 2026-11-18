"""Order execution for the ETF bot (execution mode only): one trade plan at a time.

Each `Executor.step(snapshot)` call advances the active plan. Rules:
- Later legs are sized from what earlier legs actually filled.
- Legs on different tickers may go out back to back in one step. The same
  ticker is never written twice in one step, so position-based reconciliation
  always compares against a snapshot taken after the previous write.
- A submission that timed out is reconciled from the position (the source of
  truth) before anything is resent.
- Anything that doesn't reconcile pauses the plan: no new trades start until the
  next heat, and the reason is printed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from etf_config import ETF, FX, StrategyConfig
from rit_client import RitError, SubmissionUncertain
from snapshot import Snapshot
from strategy import Decision, Leg, unwind_step

ACTIVE, DONE, PAUSED = "ACTIVE", "DONE", "PAUSED"
FINISHED = ("TRANSACTED", "CANCELLED", "REJECTED")

LogFn = Callable[..., None]


def _sign(action: str) -> float:
    return 1.0 if action == "BUY" else -1.0


@dataclass
class TrackedOrder:
    ticker: str
    action: str
    quantity: float
    order_type: str
    price: float | None
    position_before: float
    order_id: int | None = None
    filled: float = 0.0
    vwap: float | None = None
    status: str = "PENDING"  # OPEN, TRANSACTED, CANCELLED, REJECTED, UNCERTAIN
    polls: int = 0
    cancel_requested: bool = False
    error: str = ""

    @property
    def finished(self) -> bool:
        return self.status in FINISHED

    def update_from(self, data: dict) -> None:
        if data.get("order_id") is not None:
            self.order_id = int(data["order_id"])
        self.filled = float(data.get("quantity_filled") or 0.0)
        vwap = data.get("vwap")
        self.vwap = float(vwap) if vwap is not None else self.vwap
        status = str(data.get("status") or "").upper()
        self.status = status if status else "OPEN"

    def text(self) -> str:
        ident = f"#{self.order_id}" if self.order_id is not None else "#?"
        price = f" @{self.price}" if self.price is not None else ""
        return (
            f"{ident} {self.action} {self.quantity:,.0f} {self.ticker} {self.order_type}{price}: "
            f"{self.status}, filled {self.filled:,.0f}" + (f" ({self.error})" if self.error else "")
        )


class OrderDesk:
    """Submits, polls, cancels and reconciles orders. Remembers every order it sent."""

    def __init__(self, client: Any, cfg: StrategyConfig, log: LogFn):
        self.client = client
        self.cfg = cfg
        self.log = log
        self.orders: list[TrackedOrder] = []

    def known_ids(self) -> set[int]:
        return {order.order_id for order in self.orders if order.order_id is not None}

    def submit(
        self,
        snap: Snapshot,
        ticker: str,
        action: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> TrackedOrder:
        order = TrackedOrder(ticker, action, quantity, order_type, price, snap.position(ticker))
        self.orders.append(order)
        try:
            order.update_from(self.client.submit_order(ticker, action, quantity, order_type, price))
        except SubmissionUncertain as error:
            order.status, order.error = "UNCERTAIN", str(error)
        except RitError as error:
            order.status, order.error = "REJECTED", str(error)
        self.log("order", order.text(), order=order.__dict__)
        return order

    def refresh(self, order: TrackedOrder, snap: Snapshot) -> None:
        """Advance one unfinished order using the server's view."""
        if order.status == "UNCERTAIN":
            self._reconcile_uncertain(order, snap)
            return
        if order.order_id is None:
            order.status, order.error = "REJECTED", "no order id returned"
            return
        try:
            order.update_from(self.client.order(order.order_id))
        except RitError as error:
            order.error = f"poll failed: {error}"
            return
        order.polls += 1
        if not order.finished and order.polls >= self.cfg.order_poll_steps and not order.cancel_requested:
            order.cancel_requested = True
            try:
                self.client.cancel_order(order.order_id)
            except RitError as error:
                order.error = f"cancel failed: {error}"
        if order.finished:
            self.log("order", order.text(), order=order.__dict__)

    def _reconcile_uncertain(self, order: TrackedOrder, snap: Snapshot) -> None:
        known = self.known_ids()
        strays = [o for o in snap.open_orders if o.ticker == order.ticker and o.order_id not in known]
        if strays:
            # The lost submission is resting on the book. Cancel it, then measure.
            for stray in strays:
                try:
                    self.client.cancel_order(stray.order_id)
                except RitError as error:
                    order.error = f"cancel of stray #{stray.order_id} failed: {error}"
            return
        moved = (snap.position(order.ticker) - order.position_before) * _sign(order.action)
        if moved <= 0.5:
            order.polls += 1
            if order.polls < 2:  # give a late arrival one more snapshot to show up
                return
        order.filled = max(0.0, min(order.quantity, moved))
        order.status = "TRANSACTED" if order.filled > 0 else "CANCELLED"
        order.error = f"reconciled from position: filled {order.filled:,.0f}"
        self.log("reconcile", order.text(), order=order.__dict__)


class BasketPlan:
    """ETF vs basket: equity legs (thinnest book first), then the FX conversion."""

    kind = "basket"

    def __init__(self, desk: OrderDesk, decision: Decision, cfg: StrategyConfig):
        self.desk = desk
        self.cfg = cfg
        self.name = decision.name
        self.legs: list[Leg] = [leg for leg in decision.legs if leg.ticker != FX]
        self.legs += [leg for leg in decision.legs if leg.ticker == FX]
        self.filled = {leg.ticker: 0.0 for leg in self.legs}
        self.attempts = {leg.ticker: 0 for leg in self.legs}
        self.index = 0
        self.reference: float | None = None  # what the first leg actually filled
        self.usd_flow = 0.0
        self.current: TrackedOrder | None = None
        self.state = ACTIVE
        self.reason = "started"

    def _target(self, index: int) -> float:
        leg = self.legs[index]
        if index == 0:
            return leg.quantity
        if leg.ticker == FX:
            return float(round(self.usd_flow))
        return self.reference or 0.0

    def _record(self, order: TrackedOrder, leg: Leg) -> None:
        self.filled[order.ticker] += order.filled
        if order.ticker == ETF and order.filled > 0:
            self.usd_flow += order.filled * (order.vwap or leg.est_vwap or 0.0)

    def exposure_text(self) -> str:
        return ", ".join(
            f"{leg.action} {leg.ticker} {self.filled[leg.ticker]:,.0f}/{self._target(i):,.0f}"
            for i, leg in enumerate(self.legs)
        )

    def _next_leg(self) -> tuple[Leg, float] | None:
        while self.index < len(self.legs):
            leg = self.legs[self.index]
            tolerance = 1.0 if leg.ticker == FX else 0.5
            remaining = self._target(self.index) - self.filled[leg.ticker]
            exhausted = self.attempts[leg.ticker] >= self.cfg.max_leg_attempts
            if remaining <= tolerance or (self.index == 0 and exhausted and self.filled[leg.ticker] > 0):
                if self.index == 0:
                    self.reference = self.filled[leg.ticker]
                self.index += 1
                continue
            if exhausted:
                if self.index == 0:
                    self.state, self.reason = DONE, f"first leg {leg.ticker} never filled; no exposure taken"
                else:
                    self.state = PAUSED
                    self.reason = (
                        f"{leg.action} {leg.ticker} stuck after {self.attempts[leg.ticker]} attempts. "
                        f"Exposure left: {self.exposure_text()}"
                    )
                return None
            return leg, remaining
        self.state, self.reason = DONE, f"all legs filled: {self.exposure_text()}"
        return None

    def step(self, snap: Snapshot) -> None:
        written: set[str] = set()
        for _ in range(2 * len(self.legs) + 2):
            if self.state != ACTIVE:
                return
            if self.current is not None:
                if not self.current.finished:
                    if self.current.ticker in written:
                        return  # just sent: check it with the next snapshot
                    self.desk.refresh(self.current, snap)
                    if not self.current.finished:
                        return
                self._record(self.current, self.legs[self.index])
                self.current = None
            nxt = self._next_leg()
            if nxt is None:
                return
            leg, remaining = nxt
            if leg.ticker in written:
                return
            quantity = int(round(remaining))
            if quantity <= 0:
                self.filled[leg.ticker] = self._target(self.index)
                continue
            use_limit = self.cfg.use_limit_orders and leg.ticker != FX and leg.limit_price is not None
            self.attempts[leg.ticker] += 1
            self.current = self.desk.submit(
                snap,
                leg.ticker,
                leg.action,
                quantity,
                "LIMIT" if use_limit else "MARKET",
                leg.limit_price if use_limit else None,
            )
            written.add(leg.ticker)


class TenderPlan:
    """Accept a fixed-price tender, confirm it in the position, then unwind in slices."""

    kind = "tender"

    def __init__(self, desk: OrderDesk, decision: Decision, tender: Any, snap: Snapshot, cfg: StrategyConfig):
        self.desk = desk
        self.cfg = cfg
        self.name = decision.name
        self.tender_id = tender.tender_id
        self.price = tender.price
        self.delta = float(decision.details["delta"])
        self.target = float(decision.details["target"])
        self.position_before = snap.position(ETF)
        self.phase = "ACCEPT"
        self.confirm_steps = 0
        self.last_progress_tick = snap.tick
        self.failures = 0
        self.current: TrackedOrder | None = None
        self.state = ACTIVE
        self.reason = "started"
        self.note = ""

    def step(self, snap: Snapshot) -> None:
        if self.state != ACTIVE:
            return
        if self.phase == "ACCEPT":
            try:
                self.desk.client.accept_tender(self.tender_id, self.price)
                self.note = "accepted; waiting for the position to change"
            except SubmissionUncertain:
                self.note = "accept may not have arrived; checking the position"
            except RitError as error:
                self.state, self.reason = DONE, f"RIT refused the tender: {error}"
                return
            self.phase = "CONFIRM"
            return

        if self.phase == "CONFIRM":
            moved = snap.position(ETF) - self.position_before
            if abs(moved - self.delta) < 1:
                self.phase = "UNWIND"
                self.last_progress_tick = snap.tick
                self.note = f"position {snap.position(ETF):+,.0f}; unwinding to {self.target:+,.0f}"
            elif abs(moved) < 1:
                self.confirm_steps += 1
                if self.confirm_steps >= self.cfg.tender_confirm_steps:
                    self.state = DONE
                    self.reason = "tender never showed in the position; treating it as not accepted"
                return
            else:
                self.state = PAUSED
                self.reason = f"RITC moved {moved:+,.0f} after accepting, expected {self.delta:+,.0f}"
                return

        # UNWIND
        if self.current is not None and not self.current.finished:
            self.desk.refresh(self.current, snap)
            if not self.current.finished:
                return
        if self.current is not None:
            if self.current.filled > 0:
                self.last_progress_tick = snap.tick
                self.failures = 0
            elif self.current.status == "REJECTED":
                self.failures += 1
            self.current = None
            return  # next slice uses a snapshot taken after this order finished
        if self.failures >= self.cfg.max_leg_attempts:
            self.state = PAUSED
            self.reason = f"unwind orders rejected {self.failures} times; RITC {snap.position(ETF):+,.0f}"
            return
        residual = snap.position(ETF) - self.target
        if abs(residual) < 1:
            self.state, self.reason = DONE, f"unwound to {self.target:+,.0f} RITC"
            return
        leg, why = unwind_step(snap, self.target, self.price, snap.tick - self.last_progress_tick, self.cfg)
        self.note = why
        if leg is None:
            if snap.ticks_left <= self.cfg.unwind_stop_crossing_ticks:
                self.state, self.reason = DONE, why
            return
        self.current = self.desk.submit(snap, ETF, leg.action, leg.quantity, "LIMIT", leg.limit_price)


class Executor:
    """Owns at most one plan. While a plan is active or paused, nothing new starts."""

    def __init__(self, client: Any, cfg: StrategyConfig, log: LogFn):
        self.cfg = cfg
        self.log = log
        self.desk = OrderDesk(client, cfg, log)
        self.plan: BasketPlan | TenderPlan | None = None
        self.completed: list[tuple[str, str, str]] = []

    def can_start(self) -> bool:
        return self.plan is None or self.plan.state == DONE

    @property
    def paused(self) -> bool:
        return self.plan is not None and self.plan.state == PAUSED

    def status_text(self) -> str:
        if self.plan is None:
            return "idle"
        note = getattr(self.plan, "note", "")
        return f"{self.plan.kind} {self.plan.name} {self.plan.state}: {self.plan.reason}" + (f" | {note}" if note else "")

    def start_basket(self, decision: Decision, snap: Snapshot) -> None:
        self._start(BasketPlan(self.desk, decision, self.cfg), snap)

    def start_tender(self, decision: Decision, tender: Any, snap: Snapshot) -> None:
        self._start(TenderPlan(self.desk, decision, tender, snap, self.cfg), snap)

    def _start(self, plan: BasketPlan | TenderPlan, snap: Snapshot) -> None:
        if not self.can_start():
            raise RuntimeError(f"a plan is already running: {self.status_text()}")
        self.plan = plan
        self.log("plan", f"start {plan.kind} {plan.name}")
        self.step(snap)

    def step(self, snap: Snapshot) -> None:
        plan = self.plan
        if plan is None or plan.state != ACTIVE:
            return
        plan.step(snap)
        if plan.state != ACTIVE:
            self.completed.append((plan.kind, plan.name, f"{plan.state}: {plan.reason}"))
            self.log("plan", f"{plan.kind} {plan.name} {plan.state}: {plan.reason}")

    def reset(self) -> None:
        """New heat: forget local plan state (the server has reset positions)."""
        self.plan = None

    def order_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {"orders": len(self.desk.orders), "rejected": 0, "uncertain": 0}
        for order in self.desk.orders:
            if order.status == "REJECTED":
                stats["rejected"] += 1
            if "reconciled" in order.error:
                stats["uncertain"] += 1
            key = f"filled_{order.ticker}"
            stats[key] = stats.get(key, 0.0) + order.filled
        return stats
