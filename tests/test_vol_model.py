"""Offline tests for the Volatility case math and the read-only dashboard."""

import math
import unittest
import warnings
from datetime import datetime

import vol_model as vm
import vol_tool

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from py_vollib.black_scholes import black_scholes as ref_price
        from py_vollib.black_scholes.greeks.analytical import delta as ref_delta
except ImportError:  # pragma: no cover - py_vollib is in requirements.txt
    ref_price = ref_delta = None


CASES = [
    # (S, K, years, r, sigma)
    (50.0, 48.0, 300 / 3600, 0.0, 0.20),
    (50.0, 50.0, 150 / 3600, 0.0, 0.30),
    (50.0, 52.0, 37 / 3600, 0.02, 0.25),
    (49.3, 51.0, 10 / 3600, 0.0, 0.45),
]


class BlackScholesTests(unittest.TestCase):
    @unittest.skipIf(ref_price is None, "py_vollib not installed")
    def test_price_and_delta_match_py_vollib(self):
        for s, k, t, r, sigma in CASES:
            for is_call, flag in ((True, "c"), (False, "p")):
                with self.subTest(s=s, k=k, t=t, r=r, sigma=sigma, flag=flag):
                    self.assertAlmostEqual(vm.bs_price(is_call, s, k, t, r, sigma), ref_price(flag, s, k, t, r, sigma), places=9)
                    self.assertAlmostEqual(vm.bs_delta(is_call, s, k, t, r, sigma), ref_delta(flag, s, k, t, r, sigma), places=9)

    def test_put_call_parity(self):
        for s, k, t, r, sigma in CASES:
            with self.subTest(k=k, sigma=sigma):
                call = vm.bs_price(True, s, k, t, r, sigma)
                put = vm.bs_price(False, s, k, t, r, sigma)
                self.assertAlmostEqual(call - put, s - k * math.exp(-r * t), places=10)

    def test_implied_vol_round_trip(self):
        for s, k, t, r, _ in CASES:
            for sigma in (0.10, 0.25, 0.60):
                for is_call in (True, False):
                    with self.subTest(k=k, sigma=sigma, is_call=is_call):
                        price = vm.bs_price(is_call, s, k, t, r, sigma)
                        self.assertAlmostEqual(vm.implied_vol(price, is_call, s, k, t, r), sigma, places=5)

    def test_implied_vol_outside_bounds_is_none(self):
        t = 150 / 3600
        self.assertIsNone(vm.implied_vol(1.99, True, 50.0, 48.0, t))  # below intrinsic 2.00
        self.assertIsNone(vm.implied_vol(50.0, True, 50.0, 48.0, t))  # at the call's upper bound S
        self.assertIsNone(vm.implied_vol(1.0, True, 50.0, 50.0, vm.years_to_expiry(300)))  # expired

    def test_years_to_expiry(self):
        self.assertAlmostEqual(vm.years_to_expiry(0), 300 / 3600)
        self.assertAlmostEqual(vm.years_to_expiry(150), 150 / 3600)
        self.assertEqual(vm.years_to_expiry(300), vm.MIN_YEARS)
        self.assertEqual(vm.years_to_expiry(305), vm.MIN_YEARS)

    def test_expiry_falls_back_to_intrinsic(self):
        self.assertAlmostEqual(vm.bs_price(True, 51.0, 50.0, 0.0, 0.0, 0.25), 1.0)
        self.assertAlmostEqual(vm.bs_price(False, 51.0, 50.0, 0.0, 0.0, 0.25), 0.0)
        self.assertEqual(vm.bs_delta(True, 51.0, 50.0, 0.0, 0.0, 0.25), 1.0)
        self.assertEqual(vm.bs_delta(False, 51.0, 50.0, 0.0, 0.0, 0.25), 0.0)


class TickerTests(unittest.TestCase):
    def test_parse_option_tickers(self):
        call = vm.parse_option_ticker("RTM48C")
        self.assertEqual((call.underlying, call.strike, call.is_call, call.kind), ("RTM", 48.0, True, "C"))
        put = vm.parse_option_ticker("RTM52P")
        self.assertEqual((put.strike, put.is_call, put.kind), (52.0, False, "P"))
        self.assertIsNone(vm.parse_option_ticker("RTM"))
        self.assertIsNone(vm.parse_option_ticker(""))


class NewsTests(unittest.TestCase):
    def test_packet_sample_sentences(self):
        current = vm.parse_vol_news("The realized volatility of RTM for this week will be 20%")
        self.assertEqual((current.kind, current.low, current.high), ("current", 0.20, 0.20))
        estimate = vm.parse_vol_news("The realized volatility of RTM for next week will be between 27-30%")
        self.assertEqual(estimate.kind, "next")
        self.assertAlmostEqual(estimate.low, 0.27)
        self.assertAlmostEqual(estimate.high, 0.30)
        self.assertAlmostEqual(estimate.value, 0.285)

    def test_range_variants(self):
        for text in (
            "realized volatility for next week will be between 27% - 30%",
            "realized volatility for next week will be between 27 ~ 30%",
            "realized volatility for next week will be between 27–30%",
            "realized volatility for next week will be between 27 to 30 %",
            "Realized Volatility of RTM for NEXT  week will be between 30-27%",
        ):
            with self.subTest(text=text):
                news = vm.parse_vol_news(text)
                self.assertEqual(news.kind, "next")
                self.assertAlmostEqual(news.low, 0.27)
                self.assertAlmostEqual(news.high, 0.30)

    def test_decimal_and_irrelevant_text(self):
        self.assertAlmostEqual(vm.parse_vol_news("Volatility for this week will be 22.5%").value, 0.225)
        self.assertIsNone(vm.parse_vol_news("RTM rallies 3% on earnings"))
        self.assertIsNone(vm.parse_vol_news("The realized volatility will be 20%"))  # no week stated
        self.assertIsNone(vm.parse_vol_news(""))


class VolScheduleTests(unittest.TestCase):
    def test_effective_vol_blend_by_hand(self):
        # At tick 37 the remaining steps are 38..300: 38 in week 1 at 20%,
        # then 75 each in weeks 2-4 at the 28.5% midpoint (weeks 3-4 carried).
        expected = math.sqrt((0.20**2 * 38 + 0.285**2 * 225) / 263)
        self.assertAlmostEqual(vm.effective_vol(37, {1: 0.20, 2: 0.285}), expected, places=12)
        self.assertAlmostEqual(expected, 0.274351, places=5)
        self.assertEqual(vm.remaining_ticks_by_week(37), {1: 38, 2: 75, 3: 75, 4: 75})

    def test_effective_vol_edge_cases(self):
        self.assertIsNone(vm.effective_vol(10, {}))
        self.assertAlmostEqual(vm.effective_vol(0, {1: 0.20}), 0.20)
        self.assertAlmostEqual(vm.effective_vol(299, {1: 0.2, 4: 0.26}), 0.26)
        self.assertAlmostEqual(vm.effective_vol(300, {1: 0.2, 4: 0.26}), 0.26)

    def test_schedule_from_news(self):
        schedule = vm.VolSchedule()
        self.assertEqual(schedule.apply(vm.parse_vol_news("volatility for this week will be 20%"), 1), 1)
        self.assertEqual(schedule.apply(vm.parse_vol_news("volatility for next week will be between 27-30%"), 37), 2)
        self.assertAlmostEqual(schedule.week_vols()[2], 0.285)
        # A boundary-tick announcement counts toward the week that is starting.
        self.assertEqual(schedule.apply(vm.parse_vol_news("volatility for this week will be 29%"), 75), 2)
        self.assertEqual(schedule.week_vols(), {1: 0.20, 2: 0.29})
        self.assertIn("wk2 29.0%", schedule.describe(80))
        self.assertIn("carried", schedule.describe(80))


class SignalAndRiskTests(unittest.TestCase):
    def setUp(self):
        self.s, self.tick, self.sigma = 50.0, 150, 0.25
        t = vm.years_to_expiry(self.tick)
        self.fair_call = vm.bs_price(True, self.s, 50.0, t, 0.0, self.sigma)
        self.fair_put = vm.bs_price(False, self.s, 50.0, t, 0.0, self.sigma)

    def test_edges_include_the_two_dollar_fee(self):
        quotes = [
            ("RTM", 49.99, 50.01),  # underlying is skipped
            ("RTM50C", self.fair_call - 0.12, self.fair_call - 0.10),  # ask 10c under fair
            ("RTM50P", self.fair_put + 0.01, self.fair_put + 0.03),  # bid only 1c over fair
        ]
        signals = {sig.ticker: sig for sig in vm.option_signals(quotes, self.s, self.tick, self.sigma)}
        self.assertEqual(set(signals), {"RTM50C", "RTM50P"})
        call, put = signals["RTM50C"], signals["RTM50P"]
        self.assertAlmostEqual(call.buy_edge, 10.0 - 2.0, places=9)
        self.assertAlmostEqual(call.sell_edge, -12.0 - 2.0, places=9)
        self.assertEqual(call.signal, "BUY")
        self.assertAlmostEqual(put.sell_edge, 1.0 - 2.0, places=9)  # 1c x 100 doesn't cover the $2 fee
        self.assertEqual(put.signal, "")
        self.assertAlmostEqual(call.market_iv, vm.implied_vol(self.fair_call - 0.11, True, 50.0, 50.0, vm.years_to_expiry(150)))

    def test_min_edge_and_sell_signal(self):
        rich = [("RTM50C", self.fair_call + 0.10, self.fair_call + 0.12)]
        self.assertEqual(vm.option_signals(rich, self.s, self.tick, self.sigma)[0].signal, "SELL")
        cheap = [("RTM50C", self.fair_call - 0.12, self.fair_call - 0.10)]
        self.assertEqual(vm.option_signals(cheap, self.s, self.tick, self.sigma, min_edge=10.0)[0].signal, "")

    def test_ten_calls_at_half_delta(self):
        delta = vm.portfolio_delta({"RTM50C": 10}, {"RTM50C": 0.50})
        self.assertEqual(delta, 500.0)
        self.assertEqual(vm.hedge_shares(delta), -500)
        self.assertEqual(vm.portfolio_delta({"RTM50C": 10, "RTM": -500}, {"RTM50C": 0.50}), 0.0)

    def test_delta_bands_and_penalty(self):
        self.assertEqual(vm.delta_status(4_999), "OK")
        self.assertEqual(vm.delta_status(-5_001), "WARN")
        self.assertEqual(vm.delta_status(7_001), "ALERT")
        self.assertAlmostEqual(vm.delta_penalty_per_second(-8_000), 100.0)
        self.assertEqual(vm.delta_penalty_per_second(6_000), 0.0)

    def test_split_order_and_limits(self):
        self.assertEqual(vm.split_order(-12_345), [10_000, 2_345])
        self.assertEqual(vm.split_order(0), [])
        usage = vm.limit_usage({"RTM": -1_200, "RTM50C": 300, "RTM48P": -100})
        self.assertEqual((usage.options_gross, usage.options_net, usage.rtm), (400, 200, -1_200))
        self.assertEqual(usage.headroom(), {"options_gross": 2_100, "options_net": 800, "rtm": 48_800})

    def test_interest_rate_units(self):
        self.assertEqual(vm.normalize_rate(None), 0.0)
        self.assertEqual(vm.normalize_rate(0), 0.0)
        self.assertAlmostEqual(vm.normalize_rate(0.02), 0.02)
        self.assertAlmostEqual(vm.normalize_rate(2), 0.02)
        secs = [{"ticker": "RTM50C", "interest_rate": 5}, {"ticker": "RTM", "interest_rate": 1}]
        self.assertAlmostEqual(vm.rate_from_securities(secs), 0.01)
        self.assertEqual(vm.rate_from_securities([{"ticker": "RTM"}]), 0.0)


class FakeReadOnlyClient:
    """Implements only the reads vol_tool needs; any write would raise AttributeError."""

    def __init__(self, case, securities, news, limits):
        self._case, self._securities, self._news, self._limits = case, securities, news, limits
        self.calls = []

    def case(self):
        self.calls.append("case")
        return self._case

    def securities(self):
        self.calls.append("securities")
        return self._securities

    def news(self, after=None, limit=None):
        self.calls.append("news")
        return [n for n in self._news if after is None or n["news_id"] > after]

    def limits(self):
        self.calls.append("limits")
        return self._limits


class DashboardTests(unittest.TestCase):
    NEWS = [
        {"news_id": 1, "tick": 1, "headline": "Volatility update",
         "body": "The realized volatility of RTM for this week will be 20%"},
        {"news_id": 2, "tick": 37, "headline": "Volatility forecast",
         "body": "The realized volatility of RTM for next week will be between 27-30%"},
    ]

    def make(self, status="ACTIVE", call_position=200):
        tick = 40
        sigma = vm.effective_vol(tick, {1: 0.20, 2: 0.285})
        fair = vm.bs_price(True, 50.01, 50.0, vm.years_to_expiry(tick), 0.0, sigma)
        securities = [
            {"ticker": "RTM", "bid": 50.00, "ask": 50.02, "position": 0, "size": 1, "interest_rate": 0},
            {"ticker": "RTM50C", "bid": round(fair - 0.22, 2), "ask": round(fair - 0.20, 2),
             "position": call_position, "size": 100},
            {"ticker": "RTM50P", "bid": 1.00, "ask": 1.02, "position": 0, "size": 100},
        ]
        case = {"name": "Volatility Trading", "period": 1, "tick": tick, "ticks_per_period": 300, "status": status}
        limits = [{"name": "OPTIONS", "gross": 200, "net": 200, "gross_limit": 2500, "net_limit": 1000}]
        return FakeReadOnlyClient(case, securities, self.NEWS, limits)

    def test_active_screen_flags_signal_alert_and_hedge(self):
        client = self.make()
        screen = vol_tool.Dashboard(client).refresh(now=datetime(2026, 11, 21, 12, 0, 0))
        self.assertIn("tick 40/300", screen)
        self.assertIn("Forecast vol", screen)
        call_row = next(line for line in screen.splitlines() if line.startswith("RTM50C"))
        self.assertIn("BUY", call_row)
        # 200 calls x 100 x ~0.5 delta is about +10,000 shares: beyond the 7,000 CRO limit.
        self.assertIn("ALERT", screen)
        self.assertIn("Hedge to zero: SELL", screen)
        self.assertIn("in 2 orders", screen)
        self.assertIn("Limits (server): OPTIONS", screen)
        self.assertTrue(set(client.calls) <= {"case", "securities", "news", "limits"})

    def test_idle_when_not_active(self):
        client = self.make(status="STOPPED")
        screen = vol_tool.Dashboard(client).refresh()
        self.assertIn("Idle: case status is STOPPED", screen)
        self.assertNotIn("securities", client.calls)

    def test_news_is_read_once(self):
        client = self.make(call_position=0)
        dashboard = vol_tool.Dashboard(client)
        dashboard.refresh()
        dashboard.refresh()
        self.assertEqual(dashboard.last_news_id, 2)
        vols = dashboard.schedule.week_vols()
        self.assertEqual(set(vols), {1, 2})
        self.assertAlmostEqual(vols[1], 0.20)
        self.assertAlmostEqual(vols[2], 0.285)
        screen = dashboard.refresh()
        self.assertIn("Delta OK", screen)
        self.assertIn("Hedge: none needed.", screen)


if __name__ == "__main__":
    unittest.main()
