# ETF Arbitrage Trading Bot

A Python trading bot for the Algorithmic ETF Arbitrage case of **RITCxSmith 2026**, a student trading competition run on the Rotman Interactive Trader (RIT) market simulator. It prices an ETF against its underlying stocks across two currencies. It only accepts large private tender offers when it can actually exit them at a profit. It places and reconciles simulated orders within the case's position limits.

All prices, positions and P&L here are from a **simulated market**.

## Competition result

Team 19 placed first in a Rotman Algorithmic ETF Arbitrage heat at RITCxSmith 2026 with **CAD 109.8K in simulated profit and zero fines**.

[![Watch Team 19's ETF heat video](assets/team-19-etf-heat-preview.jpg)](assets/team-19-etf-heat-win.mp4?raw=true)

[View the post on X](https://x.com/KodukulaNihal/status/2124216934434537907)

## The problem

- **One ETF, two stocks, two currencies.** One unit of `RITC` holds one share of `BULL` and one share of `BEAR`. The stocks trade in CAD, the ETF in USD, and `USD` quotes CAD per US dollar. In equilibrium `RITC (USD) × USD/CAD = BULL + BEAR`, and the bot trades the gap when it is wider than fees.
- **The best price is not the price you get.** Order books are finite. Selling 1,000 units walks down the bids, so every estimate here sweeps the book level by level instead of using the top quote.
- **Tender offers are the trap.** An institution offers to buy or sell 80,000–100,000 ETF units at an attractive price, valid for about five seconds. Accepting is easy. The profit depends on getting back out, and a tender priced above the best ask can still lose money once you price buying all of it back through the book. The official starter code accepts every tender. This bot values the full exit first.
- **Offsetting tenders.** During an unwind, the bot can accept a qualifying opposite offer that reduces its unhedged ETF position. A larger reversal can also qualify when the excess shares independently clear the exit costs or the stronger fair-value hurdle. The bot reconciles the current slice first and works the confirmed remainder back to the basket hedge target. When a price within the spread meets its unwind limit, it can rest a smaller order there.
- **Position limits.** The ETF counts twice toward a gross/net limit, so the bot checks the limits after each leg of a trade could fill on its own, not just after the whole trade.

## Example (synthetic replay)

`replay.py` runs saved market snapshots through the strategy offline. These are the bundled worked examples in `tests/fixtures/`:

```text
SKIP   tender 1   sell 600 @ 25.10, buy back 600 at VWAP 25.1067 (worst 25.16): est -34 CAD ...
ACCEPT tender 7   sell 98,000 @ 25.15, buy back 98,000 at VWAP 25.0800 (worst 25.08): est +4,900 CAD ...
SKIP   tender 7   sell 98,000 @ 25.15, buy back 98,000 at VWAP 25.1755 (worst 25.20): est -8,204 CAD ...
TRADE  ETF_RICH   300 units est +32.40 CAD (+0.1080/unit) after fees 20.10 and buffer 6.00; VWAP RITC 18.7000 ...
```

- **Tender 1:** the institution pays 25.10 and the best ask is 25.00, so it looks like +USD 60. But only 200 units are offered at 25.00 and the next 400 cost 25.16, so buying back all 600 loses USD 4 before fees. The bot skips it.
- **Tender 7:** with a deep book, the bot accepts because an immediate exit pays. With a shallow book, an immediate exit would lose about CAD 8,200, but the tender is still 0.15 above the stocks' fair value, so it accepts on fair value and unwinds over time with limit orders. In heat 1 on competition day, that kind of paced unwind beat the immediate-exit estimate by about 3×.
- **Basket trade:** sized to 300 units, because the next price level no longer covers the fees.

On the competition's practice server, running read-only, the bot skipped a live 96,000-unit tender. Exiting through the book would have cost about CAD 35,000 more than the tender paid.

## What's here

| File | Role |
| --- | --- |
| `bot.py` | Main loop and command line. Dry run by default; `--execute` sends simulated orders after a typed confirmation |
| `strategy.py` | Pure pricing and risk functions: order book sweeps, FX conversion, basket edge and sizing, tender valuation, limit checks, unwind pacing |
| `execution.py` | One trade plan at a time. Later legs are sized from actual fills; timed-out submissions are reconciled from positions before resending; anything that doesn't reconcile pauses new trading |
| `snapshot.py` | Collects and parses one consistent market snapshot (books, positions, limits, tenders, open orders) |
| `rit_client.py` | RIT DMA REST client: timeouts, HTTP 429 back-off, and a hard block on writes in dry-run mode |
| `etf_config.py` | Thresholds, sizes and safety margins |
| `replay.py` | Offline replay of recorded or synthetic snapshots |
| `vol_model.py`, `vol_tool.py` | Read-only dashboard for the competition's Volatility case: Black–Scholes fair values, implied volatility, portfolio delta and hedge size |
| `tests/` | 85 deterministic tests with synthetic data, including a fake RIT server for partial fills, rejections and lost submissions |

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python replay.py tests/fixtures/*.json
```

Connecting to a live case needs RIT competition credentials, which the bot asks for at runtime and never stores:

```bash
.venv/bin/python bot.py --port <case port>            # dry run: reads only, prints every decision
.venv/bin/python bot.py --port <case port> --execute  # simulated orders
```

Press Ctrl+C to stop. The bot stops sending orders, cancels open orders, then prints positions and a run summary.

## Status

The bot traded in execution mode on the RITCxSmith competition simulator on November 21, 2026. Offline tests exercise the strategy and order handling with synthetic market data.

## Credits

The case, the RIT simulator and the official starter algorithm belong to the Rotman Finance Research and Trading Lab, University of Toronto. This repository doesn't include their case materials or code. The starter was used as a reference for the API's authentication and rate-limit handling; the pricing, tender valuation, risk checks and execution logic are original.
