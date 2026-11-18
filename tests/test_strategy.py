"""Deterministic tests for ETF pricing, sizing, limits and tender decisions."""

import unittest
from dataclasses import replace

from etf_config import StrategyConfig
from snapshot import build_snapshot, parse_book
from strategy import (
    CHEAP,
    RICH,
    basket_opportunity,
    estimate_basket,
    evaluate_tender,
    exposure,
    exposure_mismatch,
    hedge_target,
    sweep_book,
    tender_position_delta,
    unwind_step,
)
from tests.etf_helpers import (
    DEEP,
    book,
    guide_basket_raw,
    guide_losing_tender_raw,
    make_raw,
    make_snapshot,
    quote,
    slide10_tender_raw,
    standard_books,
    tender,
)

CFG = StrategyConfig()
NO_MIN = replace(CFG, min_tender_profit_cad=0.0, min_tender_edge_per_unit_cad=0.0)


def practice_units(raw):
    """Report RITC the way the practice server does: LIMIT-STOCK units 0.5."""
    for row in raw["securities"]:
        if row["ticker"] == "RITC":
            row["limits"] = [{"name": "LIMIT-STOCK", "units": 0.5}]
    return raw


class SweepBookTests(unittest.TestCase):
    def test_single_level(self):
        fill = sweep_book([(25.00, 200)], 200)
        self.assertTrue(fill.complete)
        self.assertAlmostEqual(fill.cost, 5_000.0)
        self.assertAlmostEqual(fill.vwap, 25.00)

    def test_guide_levels(self):
        fill = sweep_book([(25.00, 200), (25.16, 400)], 600)
        self.assertTrue(fill.complete)
        self.assertAlmostEqual(fill.cost, 15_064.0)
        self.assertEqual(fill.worst_price, 25.16)

    def test_short_book_is_incomplete(self):
        fill = sweep_book([(25.00, 200)], 600)
        self.assertFalse(fill.complete)
        self.assertEqual(fill.filled, 200)

    def test_zero_quantity(self):
        fill = sweep_book([], 0)
        self.assertTrue(fill.complete)
        self.assertEqual(fill.cost, 0)
        self.assertIsNone(fill.vwap)

    def test_limit_price_stops_at_worse_levels(self):
        fill = sweep_book([(25.00, 200), (25.16, 400)], 600, limit_price=25.10, buying=True)
        self.assertEqual(fill.filled, 200)
        fill = sweep_book([(25.05, 300), (24.90, 500)], 800, limit_price=25.00, buying=False)
        self.assertEqual(fill.filled, 300)


class ParsingTests(unittest.TestCase):
    def test_book_accepts_spec_keys_and_partial_fills(self):
        parsed = parse_book(
            "RITC",
            {
                "bid": [
                    {"price": 24.99, "quantity": 500, "quantity_filled": 200},
                    {"price": 25.00, "quantity": 100, "quantity_filled": 0},
                    {"price": 24.99, "quantity": 50, "quantity_filled": 0},
                ],
                "ask": [{"price": 25.02, "quantity": 100, "quantity_filled": 100}, {"price": 25.03, "quantity": 10}],
            },
        )
        self.assertEqual(parsed.bids, ((25.00, 100.0), (24.99, 350.0)))
        self.assertEqual(parsed.asks, ((25.03, 10.0),))  # fully filled order dropped

    def test_missing_book_is_reported(self):
        books = standard_books()
        del books["USD"]
        snap = make_snapshot(books=books)
        self.assertIn("no book for USD", snap.missing())
        self.assertFalse(snap.is_complete())

    def test_limit_weights_come_from_securities(self):
        snap = make_snapshot()
        self.assertEqual(snap.limit_weights()["LIMIT-STOCK"], {"BULL": 1.0, "BEAR": 1.0, "RITC": 2.0})

    def test_half_unit_reported_by_practice_counts_double(self):
        weights = build_snapshot(practice_units(make_raw())).limit_weights()["LIMIT-STOCK"]
        self.assertEqual(weights, {"BULL": 1.0, "BEAR": 1.0, "RITC": 2.0})

    def test_side_that_hit_the_request_limit_is_flagged(self):
        raw = make_raw(books=standard_books(RITC=book([(25.00, 10), (24.90, 10)], [(25.10, 10)])))
        raw["book_depth"] = 2
        ritc = build_snapshot(raw).books["RITC"]
        self.assertTrue(ritc.truncated("SELL"))
        self.assertFalse(ritc.truncated("BUY"))


class ExposureTests(unittest.TestCase):
    def test_case_example(self):
        self.assertEqual(exposure({"RITC": 500, "BULL": -500, "BEAR": -500}), (2_000, 0))

    def test_etf_leg_alone(self):
        self.assertEqual(exposure({"RITC": 500}), (1_000, 1_000))

    def test_self_check_against_server(self):
        raw = make_raw(positions={"RITC": 500, "BULL": -500})
        self.assertIsNone(exposure_mismatch(build_snapshot(raw)))
        raw["limits"][0]["gross"] = 999
        self.assertIn("RIT reports", exposure_mismatch(build_snapshot(raw)))


class BasketTests(unittest.TestCase):
    def test_guide_gross_edge(self):
        snap = build_snapshot(guide_basket_raw(fees={"RITC": 0, "BULL": 0, "BEAR": 0}))
        estimate = estimate_basket(snap, RICH, 100, replace(CFG, basket_buffer_per_unit_cad=0))
        self.assertAlmostEqual(estimate.gross_cad / 100, 0.195)
        self.assertAlmostEqual(estimate.net_cad / 100, 0.195)

    def test_guide_net_edge_with_case_fees(self):
        snap = build_snapshot(guide_basket_raw())
        estimate = estimate_basket(snap, RICH, 100, CFG)
        ritc_fee_cad = 0.02 * 1.3502  # RITC fee is in USD; paying USD costs the ask
        expected = 0.195 - ritc_fee_cad - 0.02 - 0.02 - CFG.basket_buffer_per_unit_cad
        self.assertAlmostEqual(estimate.net_cad / 100, expected)

    def test_sizes_to_profitable_depth(self):
        decision = basket_opportunity(build_snapshot(guide_basket_raw()), RICH, CFG)
        self.assertTrue(decision.go, decision.reason)
        self.assertEqual(decision.quantity, 300)  # the 18.60 level isn't worth it
        legs = {leg.ticker: (leg.action, leg.quantity) for leg in decision.legs}
        self.assertEqual(legs["RITC"], ("SELL", 300))
        self.assertEqual(legs["BULL"], ("BUY", 300))
        self.assertEqual(legs["BEAR"], ("BUY", 300))
        self.assertEqual(legs["USD"], ("SELL", round(300 * 18.70)))
        self.assertEqual(decision.legs[0].ticker, "RITC")  # thinnest book first
        self.assertEqual(decision.legs[-1].ticker, "USD")

    def test_fair_prices_skip_both_directions(self):
        snap = make_snapshot()
        for direction in (RICH, CHEAP):
            decision = basket_opportunity(snap, direction, CFG)
            self.assertFalse(decision.go)
            self.assertTrue(decision.reason.startswith("top edge"), decision.reason)

    def test_limits_cap_quantity_including_partial_legs(self):
        raw = guide_basket_raw()
        raw["limits"][0]["gross_limit"] = 1_000  # 90% headroom -> 900
        decision = basket_opportunity(build_snapshot(raw), RICH, CFG)
        self.assertTrue(decision.go, decision.reason)
        self.assertEqual(decision.quantity, 200)  # 300 would reach gross 1,200
        self.assertIn("capped", decision.reason)

    def test_no_limits_reported_blocks_trading(self):
        raw = guide_basket_raw()
        raw["limits"] = []
        decision = basket_opportunity(build_snapshot(raw), RICH, CFG)
        self.assertFalse(decision.go)
        self.assertIn("no limits reported", decision.reason)


class TenderTests(unittest.TestCase):
    def evaluate(self, raw, cfg=CFG):
        snap = build_snapshot(raw)
        return evaluate_tender(snap.tenders[0], snap, cfg)

    def test_guide_losing_tender_is_skipped(self):
        decision = self.evaluate(guide_losing_tender_raw(), NO_MIN)
        self.assertFalse(decision.go)
        # 600 x 25.10 - (200 x 25.00 + 400 x 25.16) = -4 USD before the 600 x 0.02 fee
        self.assertAlmostEqual(decision.details["incremental_quote"], -4.0 - 12.0)
        self.assertIn("buy back 600 at VWAP 25.1067", decision.reason)

    def test_slide10_deep_book_accepts(self):
        decision = self.evaluate(slide10_tender_raw([(25.08, 200_000)]))
        self.assertTrue(decision.go, decision.reason)
        self.assertAlmostEqual(decision.details["incremental_quote"], (0.07 - 0.02) * 98_000)
        self.assertEqual(decision.legs[0].action, "BUY")
        self.assertEqual(decision.legs[0].quantity, 98_000)

    def test_slide10_shallow_book_skips(self):
        decision = self.evaluate(slide10_tender_raw([(25.08, 20_000), (25.20, 100_000)]))
        self.assertFalse(decision.go)
        self.assertLess(decision.details["incremental_quote"], 0)

    def test_insufficient_depth_skips(self):
        decision = self.evaluate(slide10_tender_raw([(25.08, 20_000)]))
        self.assertFalse(decision.go)
        self.assertIn("covers only 20,000 of 98,000", decision.reason)

    def test_expired_tender_skips(self):
        decision = self.evaluate(make_raw(tenders=[tender(expires=9)], tick=10))
        self.assertFalse(decision.go)
        self.assertIn("expires at tick 9", decision.reason)

    def test_limit_breach_skips(self):
        raw = slide10_tender_raw([(25.08, 200_000)])
        raw["limits"][0]["gross_limit"] = 100_000
        decision = self.evaluate(raw)
        self.assertFalse(decision.go)
        self.assertIn("gross 196,000 > 100,000", decision.reason)

    def test_limit_breach_skips_with_practice_units(self):
        raw = practice_units(slide10_tender_raw([(25.08, 200_000)]))
        raw["limits"][0]["gross_limit"] = 100_000
        decision = self.evaluate(raw)
        self.assertFalse(decision.go)
        self.assertIn("gross 196,000 > 100,000", decision.reason)  # x2, not x0.5

    def test_practice_sized_tender_fits_the_full_net_limit(self):
        raw = practice_units(slide10_tender_raw([(25.08, 200_000)], gross_limit=300_000, net_limit=200_000))
        decision = self.evaluate(raw)  # 98,000 x2 = 196,000 <= 200,000
        self.assertTrue(decision.go, decision.reason)

    def test_cut_off_book_is_named_in_the_reason(self):
        raw = slide10_tender_raw([(25.08, 20_000), (25.10, 20_000)])
        raw["book_depth"] = 2
        decision = self.evaluate(raw)
        self.assertFalse(decision.go)
        self.assertIn("raise book_depth", decision.reason)

    def test_tender_that_offsets_a_position_is_its_own_exit(self):
        books = standard_books(RITC=book([(25.00, DEEP)], [(25.12, DEEP)]))
        flat = self.evaluate(make_raw(books=books, tenders=[tender()]), NO_MIN)
        self.assertFalse(flat.go)  # from flat, buying back at 25.12 loses
        long = self.evaluate(make_raw(books=books, positions={"RITC": 600}, tenders=[tender()]), NO_MIN)
        self.assertTrue(long.go, long.reason)
        # sells the existing 600 at 25.10 instead of 25.00, and saves that exit fee
        self.assertAlmostEqual(long.details["incremental_quote"], 600 * 0.10 + 12.0)
        self.assertIn("offsets the current position", long.reason)

    def test_buy_tender_direction(self):
        books = standard_books(RITC=book([(25.00, DEEP)], [(25.02, DEEP)]))
        decision = self.evaluate(make_raw(books=books, tenders=[tender(action="BUY", quantity=1000, price=24.90)]), NO_MIN)
        self.assertTrue(decision.go, decision.reason)
        self.assertEqual(decision.details["delta"], 1000)
        self.assertAlmostEqual(decision.details["incremental_quote"], 1000 * (25.00 - 24.90 - 0.02))
        self.assertEqual(decision.legs[0].action, "SELL")

    def test_action_mapping_can_be_flipped(self):
        snap = make_snapshot(tenders=[tender(action="SELL", quantity=600)])
        self.assertEqual(tender_position_delta(snap.tenders[0], CFG), -600)
        flipped = replace(CFG, tender_action_is_trader_side=False)
        self.assertEqual(tender_position_delta(snap.tenders[0], flipped), 600)

    def test_variable_price_tender_skips(self):
        decision = self.evaluate(make_raw(tenders=[tender(fixed=False)]))
        self.assertFalse(decision.go)
        self.assertIn("variable-price", decision.reason)

    def test_hedge_target(self):
        self.assertEqual(hedge_target({"BULL": 1000, "BEAR": 1000}), -1000)
        self.assertEqual(hedge_target({"BULL": -500, "BEAR": -800}), 500)
        self.assertEqual(hedge_target({"BULL": 1000}), 0)


class UnwindTests(unittest.TestCase):
    def snap(self, bids, tick=100):
        books = standard_books(RITC=book(bids, [(25.20, DEEP)]))
        return make_snapshot(books=books, positions={"RITC": 1000}, tick=tick)

    def test_sells_only_down_to_break_even(self):
        leg, why = unwind_step(self.snap([(25.05, 300), (24.90, 5000)]), 0, 25.00, 0, CFG)
        self.assertEqual((leg.action, leg.quantity, leg.limit_price), ("SELL", 300, 25.05))
        self.assertIn("limit 25.02", why)

    def test_limit_relaxes_with_patience(self):
        cfg = replace(CFG, unwind_max_loss_per_unit=0.15)
        leg, _ = unwind_step(self.snap([(25.05, 300), (24.90, 5000)]), 0, 25.00, cfg.unwind_patience_ticks, cfg)
        self.assertEqual(leg.quantity, 1000)
        self.assertEqual(leg.limit_price, 24.90)

    def test_waits_when_bids_are_too_low(self):
        leg, why = unwind_step(self.snap([(24.80, 5000)]), 0, 25.00, 0, CFG)
        self.assertIsNone(leg)
        self.assertIn("waiting", why)

    def test_leaves_the_rest_to_close_out_near_the_end(self):
        leg, why = unwind_step(self.snap([(25.05, 5000)], tick=296), 0, 25.00, 0, CFG)
        self.assertIsNone(leg)
        self.assertIn("close-out", why)


if __name__ == "__main__":
    unittest.main()
