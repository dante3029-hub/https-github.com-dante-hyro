# HyroTrader 200k Eval Bot — v1 (dry-run/replay only, NOT live-ready)

## What this is

The executable bot that wires together the three previously-delivered pieces
into one runnable system:

- `signal_engine/` (Phase 2) — the 6 sleeve signals (Main, Short, Flow, DELTA,
  RELVOL, BOS), provably identical to the backtested reference implementations.
- `portfolio_layer/` (Phase 3) — sleeve blending, A/B combination, position
  sizing, and the `-$3,000` kill switch / ORIG_THROTTLE drawdown-band risk
  overlay, plus HyroTrader compliance tracking (max-DD floor, daily loss limit).
- `bot/` (Phase 4, this delivery) — the orchestration, state persistence,
  cadence gating, exchange client, and CLI that turn the above into an
  actual running process.

## What it can do today

Run in **dry-run/replay mode**: pull the latest available historical data,
compute sleeve signals, size the portfolio, apply risk/compliance checks, and
log what it *would* trade — without ever placing an order. This has been
smoke-tested end-to-end including a process-restart recovery test. See
`BUILD_PLAN.md` Phase 4 for the exact numbers from that test run.

```bash
python3 bot/main.py --dry-run --cycles 1
```

## What it CANNOT do today — do not point this at a real account

**There is no live market-data ingestion pipeline.** The only implemented
`DataFeed` is `ReplayDataFeed`, which reads static CSVs already confirmed
stale (universe A frozen at 2026-07-25, ~15 days old at build time; universe B
at 2026-08-05, ~4 days old). `LiveDataFeed.get_snapshot()` is a stub that
raises `NotImplementedError` on purpose. Passing `--live` also independently
fails at bot construction time if no exchange client is configured. Two
separate hard-stops, by design — this is not an oversight, it is the correct
state for a bot that has not been given a real order-book / cross-venue-basis
/ open-interest / taker-flow ingestion pipeline (see the Data Acquisition
Spec deliverable for what that pipeline would need to look like).

**Before this can trade real money:**
1. Build the live data ingestion pipeline (Phase 4 remainder / Phase 6 in `BUILD_PLAN.md`).
2. Testnet paper-trade burn-in with live signals (Phase 6).
3. Live-vs-backtest fee/slippage drift check.
4. Resolve the still-unconfirmed Phase 1→2 drawdown-floor carry-over rule
   (`portfolio_layer/compliance.py::DrawdownFloorMode` — currently defaults
   to RESET, unverified with the firm).
5. Resolve the still-open Strategy B reconstruction gap flagged in
   `BUILD_PLAN.md` Phase 3 (corr=0.9862 vs the real cached ground truth,
   not an exact match) before trusting live DELTA/RELVOL/BOS sizing at full
   precision.

## Portability caveat (read before running elsewhere)

This code was built and only ever tested inside this session's sandbox at
the absolute path `/home/user/workspace`. All modules now read a
`HYRO_WORKSPACE` environment variable (falling back to that same default
path) instead of hardcoding it, so it CAN be relocated — but you must also
bring the full historical dataset it depends on:

- `run/` — Universe A source data (currently 4 symlinked subdirectories
  pointing at this sandbox's `ex/taker_data`, `ex/oi_data`,
  `ex/coins_history` uploads — not included in this zip; several hundred MB).
- `clean_panel/` (or `clean_panel_full/clean_panel`) — Universe B source data
  (~150MB).

This zip contains **code only** — `bot/`, `signal_engine/`,
`portfolio_layer/`, `existing_botcode/` (the original exchange client the
new one extends), `option1_reference.py`, `reference_impl.py`,
`sleeve_S_reconstructed.py`, and the build plan / audit docs. To actually run
`--dry-run`, set `HYRO_WORKSPACE` to wherever you place the code AND copy the
`run/` and `clean_panel/` data directories alongside it, or run it in this
same sandbox where the data already exists.

## File map

See `BUILD_PLAN.md` Phase 4 for the full module-by-module description,
the two real bugs found and fixed during the smoke test (with exact root
causes), and the exact numeric smoke-test output (gross notional, sleeve
multipliers, per-coin dollar targets, restart-recovery verification).
