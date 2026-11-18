"""Replay recorded or fixture snapshots through the strategy. Offline: no network.

    .venv/bin/python replay.py tests/fixtures/guide_losing_tender.json
    .venv/bin/python replay.py logs/snapshots-20261118-170000.jsonl --verbose

A .json file holds one raw snapshot (or a list of them); a .jsonl file holds one
per line, as written by `bot.py --record`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from etf_config import DEFAULT_CONFIG
from snapshot import build_snapshot, format_market_line
from strategy import CHEAP, RICH, basket_opportunity, evaluate_tender


def load_raw(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def replay(paths: list[Path], verbose: bool = False, out=print) -> dict[str, int]:
    counts = {"snapshots": 0, "incomplete": 0, "basket_trades": 0, "tender_accepts": 0, "tender_skips": 0}
    for path in paths:
        for raw in load_raw(path):
            snap = build_snapshot(raw)
            counts["snapshots"] += 1
            out(format_market_line(snap))
            problems = snap.missing()
            if problems:
                counts["incomplete"] += 1
                out(f"  skip: incomplete snapshot ({'; '.join(problems)})")
                continue
            decisions = [evaluate_tender(tender, snap, DEFAULT_CONFIG) for tender in snap.tenders]
            decisions += [basket_opportunity(snap, direction, DEFAULT_CONFIG) for direction in (RICH, CHEAP)]
            for decision in decisions:
                out(f"  {decision.summary()}")
                if verbose:
                    out("    " + json.dumps(decision.details, default=str))
                if decision.kind == "basket" and decision.go:
                    counts["basket_trades"] += 1
                elif decision.kind == "tender":
                    counts["tender_accepts" if decision.go else "tender_skips"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--verbose", action="store_true", help="also print each decision's numbers")
    args = parser.parse_args(argv)
    counts = replay(args.paths, args.verbose)
    print("Summary: " + ", ".join(f"{key} {value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
