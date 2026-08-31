# PRE-FLIGHT — items 1, 2, 3

**57/57 tests pass** (46 prior + 11 new kill-switch burn-in), clean extract,
with and without a market-data panel.

---

## ITEM 2 — MULTI-DAY KILL-SWITCH BURN-IN ✅ DONE (and it found a bug)

`portfolio_layer/test_killswitch_burnin.py` — 11 tests driving the **real**
`RiskState` through realistic paths, not a single synthetic equity jump.

| test | what it would catch |
|---|---|
| gradual intraday bleed | trips at exactly −$3,000, not early/late |
| no trip at −$2,999.99 | flattening without cause |
| stays blocked after recovery | **re-entering the same bad day** — the failure that costs accounts |
| re-arms next session only | a per-session block silently becoming permanent |
| 5 consecutive losing days | fires all 5 — a re-arm failure looks fine on day 1 |
| throttle bands / peak ratchet | peak equity ratcheting DOWN |
| restart mid-drawdown | restart clearing the flag and re-entering |
| daily limit trips independently | ← **FOUND A BUG** |
| gap straight through threshold | single-mark gap still trips, doesn't double-fire |

### BUG FOUND AND FIXED — firm daily-limit breach was never recorded

`on_intraday_pnl_update()` returned early on the kill-switch branch:

```python
if ... <= -KILL_SWITCH_DOLLARS:
    self.kill_switch_tripped_today = True
    return True                      # <-- daily-limit check never reached
if ... <= -DAILY_LOSS_LIMIT:
    self.daily_loss_limit_tripped_today = True
```

A single mark gapping straight past −$10,000 set the kill-switch flag and left
`daily_loss_limit_tripped_today` **False**. Compounded by `orchestrator.py:170`
using `elif`, so only the −$3,000 kill switch was logged.

- **Trading was SAFE** — `target_exposure_multiplier()` returns 0.0 on either flag.
- **Reporting was WRONG** — a breach of the firm's hard $10,000 limit was never
  recorded or alerted. You would have heard about it from HyroTrader, not your logs.

**Fixed:** both thresholds evaluated before returning; orchestrator `elif` → `if`.

---

## ITEM 1 — DRY-RUN RECONCILIATION ⚠️ SCRIPT WRITTEN, YOU MUST RUN IT

`dry_run_reconcile.py`. **I cannot run this — no exchange access.** It places no
orders; a read-only key is sufficient.

```bash
export BYBIT_API_KEY=...        # READ-ONLY preferred
export BYBIT_API_SECRET=...
python3 dry_run_reconcile.py
```

Checks, in order:
1. live equity + open positions from the venue
2. **exchange truth vs persisted `BotState`** — flags `QTY MISMATCH`,
   `UNKNOWN TO BOT`, `BOT PHANTOM`
3. risk state this cycle (exposure multiplier, thresholds, caps)
4. **per-leg 3% compliance on positions you actually hold** — stop-out loss and
   gap loss vs the $6,000 limit

Exit codes: `0` clean · `1` **DRIFT — do not go live** · `2` setup error.

**Why this and not more unit tests:** position drift is the one live failure mode
no test can catch. Partial fills, rejected orders, tick/lot rounding and manual
intervention all cause it, and every rebalance compounds it because targets are
diffed against a wrong base. **Run it once per cycle for a full session.**

---

## ITEM 3 — SURVIVABLE STARTING SIZE ✅ COMPUTED

Sizing criterion: **the size at which a kill-switch failure is still survivable**,
because the switch is load-bearing and has never run through a real drawdown.

| vol/day | pass, switch ON | pass, switch **OFF** | daily-limit breach OFF | verdict |
|---|---|---|---|---|
| **$3,000** | 88% | **81%** | 1.4% | **survivable** |
| $4,000 | 88% | 69% | 21.2% | marginal |
| $5,000 | 86% | **50%** | 35.3% | **not survivable** |

**Start at $3,000/day.** It is the only level where a kill-switch failure still
passes 81%. At $5,000 a failure is a coin flip, and **35% of failures come from a
single day** rather than a slow bleed.

Per-leg risk at that size:

```
max leg notional   $10,000        stop-out at 40%   $4,000   PASS (limit $6,000)
60% gap            $6,000  PASS   aggregate cap    $30,000   = 15% of account
max legs at full size  8
```

**Step up only after** the kill switch has fired correctly on real equity marks
in live conditions.

---

## STILL OPEN — not blocking, but unresolved

1. **`reference_impl.run_sleeve` erases the trigger-day loss.** Deterministic, so
   2.493526 reproduces exactly *and* is inflated. Measured +0.92 Sharpe at
   stop=0.24, +0.28 at 0.60 on Universe B. Reproducible ≠ correct.
2. **No walk-forward on the 6-sleeve config** — your own standing rule.
3. **3.75% median intrabar slippage** unmodelled in a daily-close backtest.
4. **Sharpe 2.49 sits at/above the ~2.4 noise ceiling** for ~80 independent
   weekly bets.

To close 1 and 2 I need `analysis/combined_uncapped_024.py` and either the `run/`
panel or the six saved sleeve return series (`main_hist`, `short_hist`,
`flow_hist`, `delta_hist`, `relvol_hist`, `bos_hist`). Neither is in the zip.
