# Burn-in, Live Execution & Data-Feed Status

**Date:** 2026-08-09 · **Scope:** the four things asked for — run the testnet burn-in, resolve the Strategy B reconstruction gap, make the bot able to place live trades, audit/debug everything.

**Bottom line: three of the four are done and verified. The fourth — a real testnet burn-in — could not be run, and no amount of debugging on my side changes that.** Details below, stated plainly.

---

## Verdict

| Deliverable | Status |
|---|---|
| Strategy B reconstruction gap | **RESOLVED** — root cause identified, quantified, and it does *not* change position sizing |
| Live execution path | **BUILT and TESTED** — 48/48 execution tests, 16/16 burn-in checks |
| Full audit / debug | **DONE** — 7 bugs found, all fixed, all regression-tested |
| Testnet burn-in against the real venue | **NOT RUN — BLOCKED.** Offline equivalent runs 16/16. See "What is still blocked" |

The bot is runnable end-to-end today against a mock venue. It is **not** ready to be pointed at real money, and the remaining gaps are listed explicitly at the bottom rather than buried.

---

## Test results — exact numbers

| Suite | Result | What it covers |
|---|---|---|
| `bot/test_safety.py` | **18 / 18 PASS** | kill switch, cadence carry-forward, session reset, compliance bust, UTC rollover, dry-run isolation |
| `bot/test_execution.py` | **48 / 48 PASS** | dollar→contract conversion, quantization, diffing, no-trade band, ordering, stops, failure isolation, convergence, dust |
| `bot/test_ingest.py` | **22 / 22 PASS** | Binance/Bybit response parsing, incremental CSV append, de-duplication |
| `bot/burn_in.py --mode offline --cycles 12` | **16 / 16 PASS** | 12 sequential 4h cycles, restart recovery, kill switch, compliance bust |
| `bot/burn_in.py --cycles 16 --hours-step 24 --start-date 2026-06-01` | **17 / 17 PASS** | 16 daily cycles **inside the data range**, so rebalances actually fire |
| `bot/smoke_test_bot.py` | **PASS** | original wiring smoke test, no regression |
| `ast.parse` across 41 modules | **0 failures** | no syntax corruption |

### A burn-in that looked clean but tested nothing

The first 12- and 14-cycle runs both returned 16/16 with the order sequence `[34, 0, 0, 0, ...]` — build the book once, then zero orders forever. That looks like perfect convergence. It is not.

The panel's last bar is **2026-08-05**, and the run started 2026-08-09. Every cycle was past the end of the data, so every cycle saw an **identical snapshot**. Targets never moved, so no rebalance ever fired. The run exercised zero turnover while reporting a clean sweep.

Re-run starting **2026-06-01**, inside the data range, 16 daily cycles:

```
order sequence: [31, 0, 0, 12, 0, 0, 14, 30, 0, 13, 0, 0, 12, 0, 30, 14]
rebalances fired on cycles: 3, 6, 7, 9, 12, 14, 15
```

**17 / 17 checks passed**, this time with genuine turnover: 31 orders to build, quiet cycles placing exactly zero, and seven real rebalances of 12–30 orders each. Max gross notional $121,288, zero planning errors. The kill switch then flattened all 31 live positions to an empty book.

A `at least one REBALANCE fired` check was added to the harness so this failure mode cannot recur silently.

---

## Bugs found and fixed

The first five were found before this segment; #6 and #7 are new.

1. **`rebalanced_sleeves` KeyError** when no sleeve was due — crashed the cycle. Fixed with unconditional init.
2. **Docstring corruption** from an earlier refactor inserted `import os` inside a docstring in `signal_engine/data_loader.py`, producing a `NameError`. Fixed; all 41 modules now `ast.parse`-verified.
3. **Crash on upgrade** — naive vs timezone-aware `datetime` comparison in `_due()` raised `TypeError`. Fixed via a single shared `bot/clock.py`.
4. **Timezone rollover inconsistency** — three call sites used `utcnow()`, `date.today()`, and `utcnow()` inconsistently. On a non-UTC VPS the trading day would roll 10–11 hours early, permitting a **second full -$10,000 daily loss inside one exchange day**. Fixed via `bot/clock.py`.
5. **Kill switch defeated by cadence carry-forward** (most severe of the first five). Proven pre-fix: switch tripped at -$3,100, `account_multiplier = 0.0`, `sized.gross_notional = 0.0` — and combined gross still came back **$45,460.50** instead of $0.00, because un-due sleeves carried forward their prior dollar targets. Fixed and regression-tested.
6. **Live execution was a stub.** `report["execution"]` was the literal string `"orders would be placed here..."`. No order was ever placed. Worse, the tracked quantity field held **dollars, not contracts** — at BTC $61,000 that is a 61,000× over-order had it ever been wired up. Now fully implemented in `bot/execution.py`.
7. **NEW — flatten left residual exposure on every position.** Found by burn-in phase 3. Close orders were sized by round-tripping contracts → USD → contracts, then floor-quantized to the instrument step. The float round-trip loses a hair and the floor then drops a **full step**. Observed: a 1,422.1-contract ICPUSDT position produced a **1,422.0** close order, leaving 0.1 contracts live. This fired on *every* leg — the bot reported `forced_flat=True` and gross target $0.00 while still carrying open positions on all 34 legs. **A kill switch that does not actually go flat is not a kill switch.** Fixed: a full close now uses the exchange-reported position size verbatim. Regression test `test_close_leaves_no_dust` added.

---

## Strategy B reconstruction gap — resolved

Original discrepancy: reconstruction correlated **0.98617** (R² 0.97253) with the cached series, and showed Sharpe **2.5895** vs the cached **2.0888** — a **24.0% overstatement**.

Ruled out by direct test, not assumption:

- **Date misalignment** — swept 317 offsets; best is 303, which is the offset already in use.
- **Data snapshot** — all **27/27** universe coins byte-identical between the two panels.
- **Universe** — `select_universe()` returns 27 coins / 884 days, matching the docstring.
- **Non-determinism** — re-ran the pipeline; bit-identical, max diff **0.000e+00**, corr **1.0000000000**.
- **Normalization window** — variant scored 0.98427, worse.
- **Vol rescale** — implied 0.0097669 vs configured `TARGET_DAILY_VOL = 0.00963175`, matching to 1.4%.

**Root cause: the cached series was built without the per-leg stop-loss.** `STOP_FRAC` sweep against the cached series:

| STOP_FRAC | corr |
|---|---|
| 0.40 | 0.97373 |
| 0.50 | 0.97921 |
| **0.60 (current config)** | **0.98617** |
| 0.70 | 0.98867 |
| 0.80 | 0.99349 |
| 1.00 | 0.99535 |
| **no stop** | **0.99558** |

Best fit — no stop, N=6, hold=7 — gives **corr 0.99558, R² 0.99118, Sharpe 2.3230**, cutting the Sharpe gap from 24.0% to **11.2%**.

Corroborating evidence: the cumulative gap is **+0.130531** and is a pure mean difference (standard deviations identical at 8.769068e-03). The top 5 days (0.9% of the sample) account for **50.5%** of the gap, top 15 (2.6%) for **91.4%**, top 20 (3.5%) for **96.2%**. Of the 17 days with |diff| > 0.003, the reconstruction outperforms on **17 of 17** — one-directional, exactly what a missing stop-loss looks like. Those days carry **1.83×** the average absolute return. Replacing the top-15 days with cached values drops reconstruction Sharpe from 2.5895 to **2.1335**.

The residual is a cost assumption. A fee sweep leaves correlation essentially invariant (0.99559 → 0.99555) while moving Sharpe a lot:

| FEE | Sharpe |
|---|---|
| 0.0 | 2.7235 |
| 0.00055 | 2.4643 |
| **0.00085 (current)** | **2.3230** |
| 0.0012 | 2.1580 |
| 0.002 | 1.7809 |

The remaining R² gap of 0.0088 is most likely the BOS parameters, which were held fixed and never swept.

### Decision: keep the cached series as the sizing basis

Tail comparison over the identical 568-day window (units of daily vol):

| Series | Sharpe | Worst day | 1st pct | Max DD |
|---|---|---|---|---|
| **Cached `combo_b_correct`** (current sizing basis) | 2.0888 | -4.31 | -1.76 | 8.65 |
| Reconstruction, no stop | 2.3230 | -3.25 | -1.48 | 7.14 |
| Reconstruction, 60% stop (what the bot actually trades) | 2.5895 | -3.04 | -1.57 | 7.23 |

The cached series is the most conservative on **both** Sharpe and tail. Switching the sizing basis to the reconstruction would size the book **up** — the wrong direction for a drawdown-limited evaluation. No change made.

**Caveat that should not be lost:** all three Sharpes sit near the noise ceiling of roughly 2.4 implied by ~568 days / ~80 independent weekly bets. These are not distinguishable from each other with statistical confidence, and none of them should be read as expected forward performance.

---

## Live data feed — the plan changed, for a concrete reason

The original plan was a `LiveDataFeed` reading Bybit's v5 public REST API. **That plan is not viable, and it would have failed silently rather than loudly.**

The signals were fit on a panel whose kline files use the **Binance** schema:

```
,open_time,open,high,low,close,volume,quote_volume,trades,taker_buy_base
```

`reference_impl.build_matrices()` reads column 9, `taker_buy_base`, and the entire DELTA sleeve — the strongest single sleeve in the book — is computed from it.

Bybit's v5 kline endpoint returns **seven fields**: `startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover`. Confirmed against [Bybit's own API documentation](https://bybit-exchange.github.io/docs/v5/market/kline): no taker buy base volume, no trade count.

So Bybit's public kline endpoint **cannot** feed the DELTA sleeve. Substituting it would not degrade the signal — it would compute a **different signal** from the one that was backtested, and nothing in the code would complain. `bot/ingest.py` therefore does not offer that option at all. Correct provider assignment is:

| Data | Provider | Why |
|---|---|---|
| klines + `taker_buy_base` | Binance USDT-M (`fapi`) | the schema the panel was actually built from |
| open interest | Bybit v5 `/v5/market/open-interest` | execution venue's own OI |
| taker delta series | Coinalyze | `run/taker/*.csv` came from there; no Bybit equivalent |

**Unquantified risk, flagged not solved:** signals are computed from Binance/Coinalyze data while orders execute on Bybit. This is a common and defensible setup, but it is a genuine basis/divergence exposure and it has never been measured in any phase of this project, including this one.

---

## What is still blocked, and why

### 1. A real testnet burn-in cannot run from here

Verified directly with `curl`:

- `https://api.bybit.com/v5/market/time` → **HTTP 403**, body: `The Amazon CloudFront distribution is configured to block access from your country`
- Same 403 on `api-testnet.bybit.com` and `api-demo.bybit.com`
- Same 403 on unauthenticated public endpoints
- `api.binance.com` → **451**
- `api.github.com` → **200** from the same host

That last line is the important one: this is a targeted per-destination geo block, not an outage, not a proxy failure, and **not a credentials problem**. No API key changes it.

**This also affects you directly, not just this sandbox.** Bybit blocks US IP addresses, and you are in Hawaii. You will hit the same wall from home. A live or testnet run needs a VPS in a permitted jurisdiction — and you should confirm with HyroTrader that their rules permit it before you set one up.

I did not fake a burn-in result. `bot/burn_in.py --mode demo` deliberately **aborts** if credentials are absent rather than falling back to the mock while claiming to have hit the venue.

### 2. What the offline burn-in does and does not prove

**Proves:** cycle sequencing, cadence, state persistence and restart recovery, kill-switch and compliance flattening, dollar→contract conversion, quantization against instrument filters, position diffing, order convergence, reduce-before-increase ordering, stop attachment, per-symbol failure isolation.

**Does NOT prove:** connectivity, authentication, request signing, rate limits, real fills, slippage, partial fills, exchange-side rejections, funding, or liquidation behaviour.

Only a run against the real venue proves the second list. Treat the offline 16/16 as necessary, not sufficient.

### 3. The mock's instrument filters are approximated

Real `qtyStep` / `minOrderQty` values come from `/v5/market/instruments-info`, which is geo-blocked. `bot/mock_exchange.py` approximates them by price tier. This reproduces the *shape* of the constraint so the quantization and min-qty rejection paths get exercised, but it is **not** the real filter table. Any live run must read the real filters first.

### 4. The HTTP layer in `bot/ingest.py` has never executed

Parsers are tested (22/22), including a check that a parsed Binance row exactly equals the real first row of `clean_panel/hist/BTC_1h.csv`. The network code itself has never made a successful request.

### 5. Still unconfirmed with HyroTrader

- `DrawdownFloorMode.RESET` vs `CARRY_OVER`
- `DAILY_RESET_HOUR_UTC = 0` — the assumed daily reset boundary

Both materially change risk behaviour. They are assumptions in the code right now, marked as such.

### 6. Untouched from earlier phases

Order-book depth and cross-venue basis were specified in the "Data Acquisition Spec" document and never implemented. The liquidity screen still runs off static CSVs.

---

## How to run

```bash
cd /home/user/workspace

python3 bot/test_safety.py        # 18/18
python3 bot/test_execution.py     # 48/48
python3 bot/test_ingest.py        # 22/22
python3 bot/smoke_test_bot.py

# offline burn-in — no credentials, no network
python3 bot/burn_in.py --mode offline --cycles 12

# burn-in that actually exercises rebalancing (start date must be inside the panel)
python3 bot/burn_in.py --mode offline --cycles 16 --hours-step 24 --start-date 2026-06-01

# real venue — needs credentials AND a non-geo-blocked host
BYBIT_API_KEY=... BYBIT_API_SECRET=... python3 bot/burn_in.py --mode demo --cycles 12
```

Three execution modes: `dry_run=True` (plans only, never touches the client), `dry_run=False, paper_trade=True` (full pipeline, sends nothing — the burn-in mode), and `dry_run=False, paper_trade=False` (live).

---

## Recommended next step

Stand up a VPS in a jurisdiction Bybit permits, confirm that arrangement is acceptable to HyroTrader, then run `--mode demo` for a minimum of one full week including at least two rebalance boundaries. Nothing in the untested list above can be closed any other way.

**No claim is made here about profitability.** Every number in this document is in-sample or synthetic, and the Sharpe figures sit at the noise ceiling for the available sample.
