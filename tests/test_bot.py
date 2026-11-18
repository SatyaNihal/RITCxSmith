"""Bot loop tests with a read-only fake client: idling, decisions, heat changes."""

import unittest

from bot import Bot
from etf_config import StrategyConfig
from tests.etf_helpers import make_raw, standard_books, book, DEEP, tender


class FakeReadClient:
    """Serves one synthetic raw snapshot. Any write would raise AttributeError."""

    execute = False
    base_url = "http://fake/v1"

    def __init__(self, raw):
        self.raw = raw
        self.calls = []
        self.stats = {"requests": 0}

    def case(self):
        self.calls.append("case")
        return dict(self.raw["case"])

    def book(self, ticker, limit=None):
        self.calls.append(f"book {ticker}")
        return self.raw["books"][ticker]

    def securities(self):
        return self.raw["securities"]

    def limits(self):
        return self.raw["limits"]

    def tenders(self):
        return self.raw["tenders"]

    def orders(self, status="OPEN"):
        return self.raw["orders"]


class Recorder:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


class BotTests(unittest.TestCase):
    def make_bot(self, raw):
        lines = []
        log = Recorder()
        bot = Bot(FakeReadClient(raw), StrategyConfig(), log, out=lines.append)
        return bot, lines, log

    def test_idles_when_case_is_not_active(self):
        bot, lines, _ = self.make_bot(make_raw(status="STOPPED"))
        self.assertTrue(bot.step())
        self.assertNotIn("book RITC", bot.client.calls)
        self.assertTrue(any("idle" in line for line in lines))

    def test_dry_run_prints_both_directions_and_logs_them(self):
        bot, lines, log = self.make_bot(make_raw())
        self.assertFalse(bot.step())
        self.assertIsNone(bot.executor)
        self.assertTrue(any("ETF_RICH" in line for line in lines))
        self.assertTrue(any("ETF_CHEAP" in line for line in lines))
        names = [r.get("name") for r in log.records if r.get("event") == "decision"]
        self.assertEqual(names, ["ETF_RICH", "ETF_CHEAP"])

    def test_dry_run_reports_a_tender_it_would_accept_once(self):
        books = standard_books(RITC=book([(25.00, DEEP)], [(25.02, DEEP)]))
        raw = make_raw(books=books, tenders=[tender(3, "BUY", 40_000, 24.80)])
        bot, lines, _ = self.make_bot(raw)
        bot.step()
        bot.step()
        self.assertEqual(sum("would accept tender 3" in line for line in lines), 1)
        self.assertEqual(bot.stats["tenders_seen"], 1)

    def test_new_heat_clears_tender_memory(self):
        raw = make_raw(tick=200, tenders=[tender(3, expires=250, tick=200)])
        bot, lines, _ = self.make_bot(raw)
        bot.step()
        self.assertIn(3, bot.tender_verbs)
        raw["case"]["tick"] = 2
        raw["tenders"] = []
        bot.step()
        self.assertTrue(any("New heat" in line for line in lines))
        self.assertNotIn(3, bot.tender_verbs)

    def test_incomplete_snapshot_skips_decisions(self):
        books = standard_books(USD=book([], [(1.40, DEEP)]))
        bot, lines, log = self.make_bot(make_raw(books=books))
        bot.step()
        self.assertTrue(any("incomplete snapshot" in line for line in lines))
        self.assertFalse(any(r.get("event") == "decision" for r in log.records))


if __name__ == "__main__":
    unittest.main()
