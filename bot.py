"""ETF arbitrage bot for the RITCxSmith Algorithmic ETF Arbitrage case.

    .venv/bin/python bot.py                  # dry run (default): reads only, prints decisions
    .venv/bin/python bot.py --once           # one snapshot and its decisions, then exit
    .venv/bin/python bot.py --record logs    # also save raw snapshots for replay.py
    .venv/bin/python bot.py --execute        # sends simulated orders (asks you to type EXECUTE)
    .venv/bin/python bot.py --port 16xxx     # the competition-day port

Stop with Ctrl+C. The bot stops sending orders, cancels open orders (execute
mode), then prints open orders, positions and a run summary. Press Ctrl+C a
second time only if that procedure hangs.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from etf_config import DEFAULT_CONFIG, TICKERS, StrategyConfig
from execution import Executor
from rit_client import (
    DEFAULT_HOST,
    PRACTICE_PORTS,
    AuthError,
    RitClient,
    RitError,
    prompt_credentials,
)
from snapshot import Snapshot, build_snapshot, collect_raw, format_market_line, limit_weight
from strategy import CHEAP, RICH, basket_opportunity, evaluate_tender, exposure_mismatch

IDLE_PRINT_EVERY_S = 5.0
IDLE_SLEEP_S = 1.0


class JsonlLog:
    """Append-only JSON lines file. Holds market data and decisions, never credentials."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class Bot:
    def __init__(
        self,
        client: Any,
        cfg: StrategyConfig,
        decisions: JsonlLog | None,
        snapshots: JsonlLog | None = None,
        out: Any = print,
    ):
        self.client = client
        self.cfg = cfg
        self.decisions = decisions
        self.snapshots = snapshots
        self.out = out
        self.executor = Executor(client, cfg, self.event) if client.execute else None
        self.stop_requested = False
        self.heat: int | None = None
        self.last_tick: int | None = None
        self.tender_verbs: dict[int, str] = {}
        self.acted_tenders: set[int] = set()
        self.last_idle_print = 0.0
        self.last_mismatch: str | None = None
        self.last_status_print = ""
        self.stats: dict[str, float] = {
            "loops": 0,
            "snapshots": 0,
            "skipped_incomplete": 0,
            "skipped_stale": 0,
            "api_errors": 0,
            "basket_signals_rich": 0,
            "basket_signals_cheap": 0,
            "tenders_seen": 0,
            "tenders_accept_signals": 0,
            "tenders_started": 0,
            "peak_gross": 0.0,
            "peak_abs_net": 0.0,
        }

    # -- logging ------------------------------------------------------------

    def event(self, kind: str, message: str, **data: Any) -> None:
        self.out(f"  [{kind}] {message}")
        self._record({"event": kind, "message": message, **data})

    def _record(self, record: dict) -> None:
        if self.decisions is not None:
            self.decisions.write({"ts": time.time(), "heat": self.heat, "tick": self.last_tick, **record})

    # -- startup -------------------------------------------------------------

    def preflight(self) -> bool:
        case = self.client.case()
        securities = self.client.securities()
        limits = self.client.limits()
        tickers = {row.get("ticker") for row in securities}
        self.out(
            f"Case: {case.get('name')} | status {case.get('status')} | period {case.get('period')} "
            f"| tick {case.get('tick')}/{case.get('ticks_per_period')} | endpoint {self.client.base_url}"
        )
        missing = [ticker for ticker in TICKERS if ticker not in tickers]
        if missing:
            self.out(f"Missing tickers {missing}: this is not the ETF case (wrong port?). Stopping.")
            return False
        self.out("Instrument settings reported by /securities:")
        for row in securities:
            if row.get("ticker") not in TICKERS:
                continue
            units = ", ".join(
                f"{item.get('name')}={item.get('units')} (counted x{limit_weight(float(item['units'])):g})"
                if item.get("units")
                else f"{item.get('name')}={item.get('units')}"
                for item in row.get("limits") or []
            )
            self.out(
                f"  {row.get('ticker'):<5} currency {row.get('currency')}  fee {row.get('trading_fee')}  "
                f"rebate {row.get('limit_order_rebate')}  max size {row.get('max_trade_size')}  "
                f"decimals {row.get('quoted_decimals')}  limit units [{units}]"
            )
        for item in limits:
            self.out(
                f"  limit {item.get('name')}: gross {item.get('gross')}/{item.get('gross_limit')} "
                f"net {item.get('net')}/{item.get('net_limit')} fines {item.get('gross_fine')}/{item.get('net_fine')}"
            )
        mode = "EXECUTION (simulated orders)" if self.executor else "DRY RUN (reads only)"
        self.out(f"Mode: {mode}")
        self._record({"event": "preflight", "case": case, "securities": securities, "limits": limits})
        return True

    # -- main loop -------------------------------------------------------------

    def run(self, once: bool = False, max_loops: int | None = None) -> None:
        while not self.stop_requested:
            started = time.time()
            self.stats["loops"] += 1
            try:
                idle = self.step(force=once)
            except AuthError:
                raise
            except RitError as error:
                self.stats["api_errors"] += 1
                self.out(f"API error: {error}. No new trades this loop.")
                idle = False
            if once or (max_loops is not None and self.stats["loops"] >= max_loops):
                break
            pause = IDLE_SLEEP_S if idle else self.cfg.loop_sleep_s
            time.sleep(max(0.0, pause - (time.time() - started)))

    def step(self, force: bool = False) -> bool:
        """One loop. Returns True when the case is idle (not ACTIVE)."""
        case = self.client.case()
        period = int(case.get("period") or 0)
        tick = int(case.get("tick") or 0)
        self._detect_new_heat(period, tick)
        status = str(case.get("status"))
        if status != "ACTIVE" and not force:
            now = time.time()
            if now - self.last_idle_print >= IDLE_PRINT_EVERY_S:
                self.out(f"idle: {case.get('name')} is {status} (period {period}, tick {tick}); no orders")
                self.last_idle_print = now
            return True

        raw = collect_raw(self.client, self.cfg.book_depth, case=case)
        if self.snapshots is not None:
            self.snapshots.write(raw)
        snap = build_snapshot(raw)
        self.stats["snapshots"] += 1
        self._track_peaks(snap)
        self.out(format_market_line(snap))
        self._check_exposure(snap)

        problems = snap.missing()
        if problems:
            self.stats["skipped_incomplete"] += 1
            self.out(f"  skip: incomplete snapshot ({'; '.join(problems)})")
            return False
        if snap.collection_span_s > self.cfg.max_snapshot_age_s:
            self.stats["skipped_stale"] += 1
            self.out(f"  skip: snapshot took {snap.collection_span_s:.2f}s to collect (max {self.cfg.max_snapshot_age_s}s)")
            return False
        if status != "ACTIVE":
            self.out(f"  note: case is {status}; decisions below are informational")

        if self.executor is not None:
            self.executor.step(snap)
            text = self.executor.status_text()
            if text != "idle" and text != self.last_status_print:
                self.out(f"  plan: {text}")
                self.last_status_print = text

        self._handle_tenders(snap, live=status == "ACTIVE")
        self._handle_basket(snap, live=status == "ACTIVE")
        return False

    # -- decisions ---------------------------------------------------------------

    def _handle_tenders(self, snap: Snapshot, live: bool) -> None:
        for tender in snap.tenders:
            decision = evaluate_tender(tender, snap, self.cfg)
            first_time = tender.tender_id not in self.tender_verbs
            if first_time:
                self.stats["tenders_seen"] += 1
                self.out(
                    f"  tender {tender.tender_id}: {tender.action} {tender.quantity:,.0f} {tender.ticker} "
                    f"@ {tender.price} (fixed={tender.is_fixed_bid}, expires tick {tender.expires}) "
                    f"\"{tender.caption}\""
                )
            if first_time or self.tender_verbs[tender.tender_id] != decision.verb:
                self.out(f"  {decision.summary()}")
                self._record({"event": "decision", **decision.to_dict(), "tender": tender.__dict__})
            self.tender_verbs[tender.tender_id] = decision.verb
            if not decision.go or tender.tender_id in self.acted_tenders or not live:
                continue
            self.stats["tenders_accept_signals"] += 1
            if self.executor is None:
                self.out(f"  [dry run] would accept tender {tender.tender_id}")
                self.acted_tenders.add(tender.tender_id)
            elif self.executor.can_start():
                self.acted_tenders.add(tender.tender_id)
                self.stats["tenders_started"] += 1
                self.executor.start_tender(decision, tender, snap)

    def _handle_basket(self, snap: Snapshot, live: bool) -> None:
        for direction in (RICH, CHEAP):
            decision = basket_opportunity(snap, direction, self.cfg)
            self.out(f"  {decision.summary()}")
            self._record({"event": "decision", "period": snap.period, "tick": snap.tick, **decision.to_dict()})
            if not decision.go or not live:
                continue
            self.stats["basket_signals_rich" if direction == RICH else "basket_signals_cheap"] += 1
            if self.executor is not None and self.executor.can_start():
                self.executor.start_basket(decision, snap)

    # -- bookkeeping -----------------------------------------------------------------

    def _detect_new_heat(self, period: int, tick: int) -> None:
        new_heat = self.heat is not None and (
            period != self.heat or (self.last_tick is not None and tick + 5 < self.last_tick)
        )
        if new_heat:
            self.out(f"=== New heat detected (period {period}, tick {tick}): clearing local state ===")
            self.tender_verbs.clear()
            self.acted_tenders.clear()
            self.last_mismatch = None
            if self.executor is not None:
                if not self.executor.can_start():
                    self.out(f"  previous plan dropped: {self.executor.status_text()}")
                self.executor.reset()
            self._print_server_state()
        self.heat = period
        self.last_tick = tick

    def _track_peaks(self, snap: Snapshot) -> None:
        for limit in snap.limits:
            self.stats["peak_gross"] = max(self.stats["peak_gross"], limit.gross)
            self.stats["peak_abs_net"] = max(self.stats["peak_abs_net"], abs(limit.net))

    def _check_exposure(self, snap: Snapshot) -> None:
        mismatch = exposure_mismatch(snap)
        if mismatch and mismatch != self.last_mismatch:
            self.out(f"  WARNING exposure check: {mismatch}")
            self._record({"event": "exposure_mismatch", "message": mismatch})
        self.last_mismatch = mismatch

    def _print_server_state(self) -> None:
        try:
            open_orders = self.client.orders("OPEN")
            securities = self.client.securities()
        except RitError as error:
            self.out(f"  could not read server state: {error}")
            return
        positions = ", ".join(
            f"{row.get('ticker')} {row.get('position'):+,.0f}" for row in securities if row.get("ticker") in TICKERS
        )
        self.out(f"  server positions: {positions}")
        self.out(f"  server open orders: {len(open_orders)}")
        for order in open_orders:
            self.out(
                f"    #{order.get('order_id')} {order.get('action')} {order.get('quantity')} {order.get('ticker')} "
                f"{order.get('type')} @{order.get('price')} filled {order.get('quantity_filled')}"
            )

    # -- stop procedure ------------------------------------------------------------------

    def shutdown(self) -> None:
        self.out("Stopping: no new orders will be sent.")
        if self.executor is not None:
            self.out(f"  plan at stop: {self.executor.status_text()}")
            try:
                result = self.client.cancel_all()
                self.out(f"  cancelled open orders: {result.get('cancelled_order_ids', result)}")
            except RitError as error:
                self.out(f"  cancel_all failed: {error}. Check open orders in RIT now.")
        self._print_server_state()
        self.out("Run summary:")
        for key, value in self.stats.items():
            self.out(f"  {key}: {value:,.0f}")
        if self.executor is not None:
            for key, value in self.executor.order_stats().items():
                self.out(f"  {key}: {value:,.0f}")
            for kind, name, result in self.executor.completed:
                self.out(f"  plan {kind} {name}: {result}")
        for key, value in self.client.stats.items():
            self.out(f"  http_{key}: {value:,}")
        self._record({"event": "summary", "stats": self.stats, "http": self.client.stats})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RITC ETF arbitrage bot (dry run unless --execute).")
    parser.add_argument("--execute", action="store_true", help="send simulated orders (asks for confirmation)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=PRACTICE_PORTS["etf"])
    parser.add_argument("--record", metavar="DIR", help="save raw snapshots as JSON lines for replay.py")
    parser.add_argument("--log-dir", default="logs", help="where decision logs go (default: logs)")
    parser.add_argument("--once", action="store_true", help="take one snapshot, print decisions, exit")
    parser.add_argument("--max-loops", type=int, help="stop after this many loops")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and args.once:
        print("--once is read-only; drop --execute.")
        return 2
    print(f"RIT endpoint: http://{args.host}:{args.port}/v1")
    credentials = prompt_credentials()
    client = RitClient(credentials, args.port, args.host, execute=args.execute)
    if args.execute:
        answer = input(f"EXECUTION MODE on {client.base_url}: orders will be sent. Type EXECUTE to continue: ")
        if answer.strip() != "EXECUTE":
            print("Not confirmed. Exiting without sending anything.")
            return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    decisions = JsonlLog(Path(args.log_dir) / f"decisions-{stamp}.jsonl")
    snapshots = JsonlLog(Path(args.record) / f"snapshots-{stamp}.jsonl") if args.record else None
    bot = Bot(client, DEFAULT_CONFIG, decisions, snapshots)

    def request_stop(signum: int, frame: Any) -> None:
        bot.stop_requested = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # a second Ctrl+C interrupts immediately

    signal.signal(signal.SIGINT, request_stop)
    code = 0
    try:
        if not bot.preflight():
            return 1
        bot.run(once=args.once, max_loops=args.max_loops)
    except AuthError as error:
        print(error)
        code = 1
    finally:
        try:
            if not args.once:
                bot.shutdown()
        except RitError as error:
            print(f"Stop procedure hit an API error: {error}. Check RIT for open orders and positions.")
        decisions.close()
        if snapshots is not None:
            snapshots.close()
        client.close()
        print(f"Decision log: {decisions.path}" + (f" | snapshots: {snapshots.path}" if snapshots else ""))
    return code


if __name__ == "__main__":
    sys.exit(main())
