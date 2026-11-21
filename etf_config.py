"""Settings for the ETF arbitrage bot.

Edit these between heats. Nothing here is secret: credentials are prompted at
runtime by rit_client.prompt_credentials().
"""

from __future__ import annotations

from dataclasses import dataclass

ETF = "RITC"
STOCKS = ("BULL", "BEAR")
FX = "USD"  # quoted as CAD per 1 USD
BASE = "CAD"
TICKERS = ("CAD", "USD", "BULL", "BEAR", "RITC")
BOOK_TICKERS = ("RITC", "BULL", "BEAR", "USD")

# Case-packet values, used only when /securities does not report a setting.
FALLBACK_FEE_PER_SHARE = {"RITC": 0.02, "BULL": 0.02, "BEAR": 0.02, "USD": 0.0, "CAD": 0.0}
FALLBACK_REBATE_PER_SHARE = {"RITC": 0.01, "BULL": 0.01, "BEAR": 0.01, "USD": 0.0, "CAD": 0.0}
FALLBACK_MAX_TRADE_SIZE = {"RITC": 10_000, "BULL": 10_000, "BEAR": 10_000, "USD": 2_500_000, "CAD": 2_500_000}
FALLBACK_CURRENCY = {"RITC": "USD", "BULL": "CAD", "BEAR": "CAD", "USD": "CAD", "CAD": "CAD"}
# Stocks count once toward the limits and the ETF counts twice (case packet pp. 3-4).
# The bot prefers the per-security `limits[].units` that /securities reports, read
# through snapshot.limit_weight() (practice reports RITC as 0.5, counted as x2).
FALLBACK_LIMIT_WEIGHTS = {"RITC": 2.0, "BULL": 1.0, "BEAR": 1.0}
DEFAULT_TICKS_PER_PERIOD = 300


@dataclass(frozen=True)
class StrategyConfig:
    # --- Tender offers -------------------------------------------------
    min_tender_profit_cad: float = 500.0        # estimated net profit needed to accept
    min_tender_edge_per_unit_cad: float = 0.02  # ... and per ETF unit
    tender_buffer_per_unit_cad: float = 0.02    # allowance for prices moving while exiting
    min_ticks_to_tender_expiry: int = 1
    min_heat_ticks_left_for_tender: int = 5
    # Fair-value path: also accept a tender whose price beats the stocks' fair value,
    # (BULL + BEAR) / USD, by at least this much per unit (USD, on your side), even if
    # the book can't absorb it all right now. The paced limit-order unwind works it off
    # as the book refills. Competition day, heat 1: tender 1018's slow unwind earned
    # +0.158/unit; skipped tenders 18 and 518 (+0.14 vs fair) would have earned about
    # +0.04 and +0.10/unit over 20 ticks, while 1518 (+0.10) and 2018 (-0.08) would have lost.
    # None turns the path off (immediate full exit required).
    tender_fair_edge_min: float | None = 0.12
    tender_fair_min_ticks_left: int = 40        # time to unwind ~90k units in slices
    reversal_fair_edge_min: float = 0.18        # USD/unit for a larger position opened by an opposite tender
    # VERIFY in practice: True means a tender with action BUY makes *your* RITC
    # position go up. Compare with the RIT tender window before trading.
    tender_action_is_trader_side: bool = True

    # --- ETF vs basket arbitrage ---------------------------------------
    basket_min_edge_per_unit_cad: float = 0.02  # marginal edge after fees and buffer
    basket_buffer_per_unit_cad: float = 0.02
    basket_max_qty: int = 3_000                 # capped further by executable depth and limits
    basket_lot: int = 100
    convert_fx: bool = True                     # convert the ETF leg's USD to CAD

    # --- Risk ----------------------------------------------------------
    limit_headroom: float = 0.90                # basket trades use at most 90% of each limit
    # Tenders arrive sized just under the limits (practice: 96,000 RITC x2 = 192,000 of a
    # 200,000 net limit). Accepting is one fill and the unwind only shrinks it, and RIT
    # enforces the hard limit, so tenders may use the full limit.
    tender_limit_headroom: float = 1.00

    # --- Data ----------------------------------------------------------
    max_snapshot_age_s: float = 2.0
    book_depth: int = 200                       # orders requested per book side (100 was hit on most practice snapshots)

    # --- Execution -----------------------------------------------------
    use_limit_orders: bool = True               # price-limit the first submission on each equity leg
    max_leg_attempts: int = 3
    order_poll_steps: int = 3                   # loops to wait for a fill before cancelling
    tender_confirm_steps: int = 6               # loops to wait for an accepted tender to show
    unwind_slice_max: int = 10_000
    unwind_passive_max: int = 5_000             # resting slice when the exit price lies within the spread
    unwind_stop_crossing_ticks: int = 5         # leave the rest to the end-of-heat close-out
    unwind_max_loss_per_unit: float = 0.05      # in the ETF's quote currency (USD)
    unwind_patience_ticks: int = 20             # ticks to relax the floor to the max loss

    # --- Loop ----------------------------------------------------------
    loop_sleep_s: float = 0.5


DEFAULT_CONFIG = StrategyConfig()
