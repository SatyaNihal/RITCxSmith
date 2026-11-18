"""Execution tests against a fake RIT: full fills, partial fills, rejections,
uncertain submissions and the tender accept/unwind path."""

import math
import unittest
from collections import defaultdict
from dataclasses import replace

from etf_config import StrategyConfig
from execution import ACTIVE, DONE, PAUSED, Executor
from rit_client import RequestRejected, SubmissionUncertain
from strategy import RICH, Decision, Leg, evaluate_tender
from tests.etf_helpers import DEEP, book, make_snapshot, standard_books, tender

CFG = StrategyConfig()
PRICES = {"RITC": 18.70, "BULL": 10.02, "BEAR": 15.03, "USD": 1.35}


class FakeRit:
    """Fills orders instantly unless a per-ticker script says otherwise.

    Script entries: "fill", "partial:<fraction>", "reject", "rest" (LIMIT order sits
    unfilled), "uncertain_filled" (fills, but the response is lost), "uncertain_lost"
    (never arrives).
    """

    execute = True

    def __init__(self, positions=None, script=None):
        self.positions = defaultdict(float, positions or {})
        self.script = {ticker: list(steps) for ticker, steps in (script or {}).items()}
        self.orders_by_id = {}
        self.next_id = 100
        self.calls = []
        self.tender_deltas = {}
        self.tender_behavior = "fill"

    def submit_order(self, ticker, action, quantity, order_type="MARKET", price=None):
        self.calls.append(("submit", ticker, action, quantity, order_type, price))
        steps = self.script.get(ticker)
        behavior = steps.pop(0) if steps else "fill"
        if behavior == "reject":
            raise RequestRejected("POST /orders", 400, "rejected by fake")
        if behavior == "uncertain_lost":
            raise SubmissionUncertain("timed out")
        if behavior.startswith("partial:"):
            filled = math.floor(quantity * float(behavior.split(":")[1]))
        elif behavior == "rest":
            filled = 0
        else:
            filled = quantity
        sign = 1 if action == "BUY" else -1
        self.positions[ticker] += sign * filled
        order = {
            "order_id": self.next_id,
            "ticker": ticker,
            "action": action,
            "type": order_type,
            "quantity": quantity,
            "quantity_filled": filled,
            "price": price,
            "vwap": (price or PRICES[ticker]) if filled else None,
            "status": "OPEN" if behavior == "rest" else "TRANSACTED",
        }
        self.orders_by_id[self.next_id] = order
        self.next_id += 1
        if behavior == "uncertain_filled":
            raise SubmissionUncertain("response lost")
        return dict(order)

    def order(self, order_id):
        self.calls.append(("poll", order_id))
        return dict(self.orders_by_id[order_id])

    def cancel_order(self, order_id):
        self.calls.append(("cancel", order_id))
        if self.orders_by_id[order_id]["status"] == "OPEN":
            self.orders_by_id[order_id]["status"] = "CANCELLED"
        return {"success": True}

    def open_orders(self):
        return [dict(o) for o in self.orders_by_id.values() if o["status"] == "OPEN"]

    def accept_tender(self, tender_id, price=None):
        self.calls.append(("accept", tender_id, price))
        if self.tender_behavior == "reject":
            raise RequestRejected("POST /tenders", 400, "tender expired")
        if self.tender_behavior == "fill":
            self.positions["RITC"] += self.tender_deltas[tender_id]
        return {"success": True}

    def submits(self, ticker=None):
        return [c for c in self.calls if c[0] == "submit" and (ticker is None or c[1] == ticker)]


def snapshot_of(fake, books=None, tick=10, tenders=()):
    return make_snapshot(
        books=books or standard_books(),
        positions=dict(fake.positions),
        orders=fake.open_orders(),
        tick=tick,
        tenders=tenders,
    )


def basket_decision(quantity=1000):
    legs = (
        Leg("RITC", "SELL", quantity, 18.70, 18.70),
        Leg("BULL", "BUY", quantity, 10.02, 10.02),
        Leg("BEAR", "BUY", quantity, 15.03, 15.03),
        Leg("USD", "SELL", round(quantity * 18.70), 1.35, 1.35),
    )
    return Decision("basket", RICH, True, quantity, 10.0, "test", legs)


def run(executor, fake, steps=12, **snap_kwargs):
    for _ in range(steps):
        if executor.plan.state != ACTIVE:
            break
        executor.step(snapshot_of(fake, **snap_kwargs))


class BasketExecutionTests(unittest.TestCase):
    def setUp(self):
        self.log = []

    def executor(self, fake, cfg=CFG):
        return Executor(fake, cfg, lambda kind, message, **data: self.log.append((kind, message)))

    def test_full_fill_sends_all_legs_and_converts_fx(self):
        fake = FakeRit()
        executor = self.executor(fake)
        executor.start_basket(basket_decision(), snapshot_of(fake))
        run(executor, fake)
        self.assertEqual(executor.plan.state, DONE, executor.status_text())
        self.assertEqual(dict(fake.positions), {"RITC": -1000, "BULL": 1000, "BEAR": 1000, "USD": -18700})
        self.assertEqual([c[1] for c in fake.submits()], ["RITC", "BULL", "BEAR", "USD"])

    def test_partial_first_leg_sizes_the_other_legs(self):
        fake = FakeRit(script={"RITC": ["partial:0.4", "reject", "reject"]})
        executor = self.executor(fake)
        executor.start_basket(basket_decision(), snapshot_of(fake))
        run(executor, fake)
        self.assertEqual(executor.plan.state, DONE, executor.status_text())
        self.assertEqual(fake.positions["RITC"], -400)
        self.assertEqual(fake.positions["BULL"], 400)
        self.assertEqual(fake.positions["BEAR"], 400)
        self.assertEqual(fake.positions["USD"], -round(400 * 18.70))

    def test_failed_leg_pauses_with_visible_exposure(self):
        fake = FakeRit(script={"BULL": ["reject", "reject", "reject"]})
        executor = self.executor(fake)
        executor.start_basket(basket_decision(), snapshot_of(fake))
        run(executor, fake)
        self.assertEqual(executor.plan.state, PAUSED)
        self.assertIn("BULL stuck after 3 attempts", executor.plan.reason)
        self.assertIn("SELL RITC 1,000/1,000", executor.plan.reason)
        self.assertTrue(executor.paused)
        self.assertFalse(executor.can_start())  # nothing new compounds the exposure
        self.assertEqual(fake.positions["BEAR"], 0)

    def test_first_leg_that_never_fills_ends_without_exposure(self):
        fake = FakeRit(script={"RITC": ["reject", "reject", "reject"]})
        executor = self.executor(fake)
        executor.start_basket(basket_decision(), snapshot_of(fake))
        run(executor, fake)
        self.assertEqual(executor.plan.state, DONE)
        self.assertIn("never filled", executor.plan.reason)
        self.assertEqual(fake.submits("BULL"), [])

    def test_uncertain_submission_that_filled_is_not_resent(self):
        fake = FakeRit(script={"RITC": ["uncertain_filled"]})
        executor = self.executor(fake)
        executor.start_basket(basket_decision(), snapshot_of(fake))
        run(executor, fake)
        self.assertEqual(executor.plan.state, DONE, executor.status_text())
        self.assertEqual(len(fake.submits("RITC")), 1)
        self.assertEqual(fake.positions["RITC"], -1000)
        self.assertEqual(fake.positions["BULL"], 1000)

    def test_uncertain_submission_that_was_lost_is_resent_once(self):
        fake = FakeRit(script={"RITC": ["uncertain_lost"]})
        executor = self.executor(fake)
        executor.start_basket(basket_decision(), snapshot_of(fake))
        run(executor, fake)
        self.assertEqual(executor.plan.state, DONE, executor.status_text())
        self.assertEqual(len(fake.submits("RITC")), 2)
        self.assertEqual(fake.positions["RITC"], -1000)

    def test_resting_limit_order_is_cancelled_and_counted(self):
        fake = FakeRit(script={"RITC": ["rest", "reject", "reject"]})
        executor = self.executor(fake, replace(CFG, use_limit_orders=True))
        executor.start_basket(basket_decision(), snapshot_of(fake))
        run(executor, fake, steps=20)
        self.assertIn(("cancel", 100), fake.calls)
        self.assertEqual(executor.plan.state, DONE)
        self.assertIn("never filled", executor.plan.reason)


class TenderExecutionTests(unittest.TestCase):
    BOOKS = standard_books(RITC=book([(25.05, DEEP)], [(25.07, DEEP)]))

    def setUp(self):
        self.fake = FakeRit()
        self.fake.tender_deltas[7] = 1000
        raw_tender = tender(7, "BUY", 1000, 24.90)
        self.snap = snapshot_of(self.fake, books=self.BOOKS, tenders=[raw_tender])
        cfg = replace(CFG, min_tender_profit_cad=0, min_tender_edge_per_unit_cad=0)
        self.decision = evaluate_tender(self.snap.tenders[0], self.snap, cfg)
        self.assertTrue(self.decision.go, self.decision.reason)
        self.executor = Executor(self.fake, cfg, lambda *a, **k: None)

    def test_accept_confirm_and_unwind(self):
        self.executor.start_tender(self.decision, self.snap.tenders[0], self.snap)
        run(self.executor, self.fake, books=self.BOOKS)
        self.assertEqual(self.executor.plan.state, DONE, self.executor.status_text())
        self.assertEqual(self.fake.positions["RITC"], 0)
        self.assertEqual(self.fake.submits(), [("submit", "RITC", "SELL", 1000, "LIMIT", 25.05)])
        self.assertIn(("accept", 7, 24.90), self.fake.calls)

    def test_refused_tender_ends_without_orders(self):
        self.fake.tender_behavior = "reject"
        self.executor.start_tender(self.decision, self.snap.tenders[0], self.snap)
        self.assertEqual(self.executor.plan.state, DONE)
        self.assertIn("refused", self.executor.plan.reason)
        self.assertEqual(self.fake.submits(), [])

    def test_tender_that_never_shows_is_dropped(self):
        self.fake.tender_behavior = "ignore"
        self.executor.start_tender(self.decision, self.snap.tenders[0], self.snap)
        run(self.executor, self.fake, books=self.BOOKS)
        self.assertEqual(self.executor.plan.state, DONE)
        self.assertIn("never showed", self.executor.plan.reason)
        self.assertEqual(self.fake.submits(), [])


if __name__ == "__main__":
    unittest.main()
