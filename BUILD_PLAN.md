# Build plan — research book to live bot

Account at **$214,000**, floor $180,000 (static), buffer **$34,000**, target
$220,000 = **+$6,000 away**. At $3,000/day that is ~10 trading days at
expected value. There is no case for sizing up.

---

## 0. DO TODAY — before any building

**Turn off BOS on the live bot.** It runs at 4h, which tested 0.33 against a
control of 0.74 -- the one setting where random direction beats it. The taker
data fix means it can now actually fire, so it will start trading its worst
configuration on the next cycle.

```
grep -n "bos" ~/bot_hyrotrader_v1/bot/config.py
```

Also outstanding: regenerate the 4 Discord webhooks and the GitHub token, all
exposed in chat.

---

## 1. WHAT EXISTS vs WHAT IS NEEDED

| sleeve | in research | in live bot | work |
|---|---|---|---|
| delta | yes (60 coins, hold 14) | yes (24 coins, hold 7) | change universe + hold |
| relvol | yes (24 coins, N=8) | yes | verify params |
| bos | **dropped** | yes, at 4h | **remove** |
| skew | yes (60d, n=6) | no | **build** |
| cascade | yes (z<=-2.5, hold 3d) | no | **build** |
| sr | yes (6h, +OI, 3ATR) | scanner only | **build** |
| fvg | yes (12h, +OI, 1ATR) | no | **build** |
| pattern | yes (6h, bull+OI) | scanner only | **build** |
| srflip | yes (6h retest) | no | **build** |

Six sleeves to build, two to adjust, one to remove.

---

## 2. STAGE 1 — signal generation

Each sleeve must produce, on its own schedule, a target position per coin.

**Cross-sectional (delta, relvol, skew, cascade)** -- daily bars, rebalance on
a fixed cadence, rank and take top/bottom N. `book.py` is the reference.

**Event (sr, fvg, pattern, srflip)** -- native timeframes 6h / 12h / 6h / 6h.
`scanner.py` already generates sr and pattern signals live, so that code is
reusable. fvg and srflip need building; both are short functions in
`hourly_sim.py`.

**Reuse, do not rewrite:**
- `markittick_detector.py` -- verified 24/24 against the Pine source
- `sr2.py` -- the levels function, after four corrections
- `fvg.py`, `hourly_sim.py::srflip_hourly`

**Data the live bot needs that it does not have:**
- matched-venue taker delta (`fetch_taker.py`, already on the server)
- OI at 6h and 12h for the sr / fvg / pattern filters

---

## 3. STAGE 2 — portfolio layer

`hourly_sim.py::simulate` is the SPECIFICATION. The live version must produce
the same positions from the same inputs -- that is a testable property, not a
design note.

1. collect target weights from all 8 sleeves
2. equal-RISK scale (each sleeve to the same trailing vol, not equal notional)
3. **net per coin** -- saves 44.7% of gross exposure
4. per-coin cap 0.15 on the NET
5. gross cap 1.0
6. cost the net change once, not per sleeve

**The concurrency bug found in the audit must not be reproduced:** a fixed
per-position size does NOT cap how many positions open at once. FVG ran at 12.5
concurrent when its sizing assumed 6 -- 2.1x levered. Every event sleeve needs
an explicit `max_gross`.

---

## 4. STAGE 3 — paper trade, one month

Run live, log what it would do, execute nothing. Compare realised signals
against the simulator. This is the only test not downstream of the same data.

**What would make me not fund it:**
- signals firing at materially different times than the simulator says
- FVG's 297x/year turnover proving unexecutable
- realised slippage above the modelled 3bp

---

## 5. SIZING

| vol/day | floor risk (SR 4.19) | floor risk (SR 2.1) | P(-$10k day) |
|---|---|---|---|
| $2,000 | 0.04% | 1.16% | 0.0% |
| **$3,000** | **0.41%** | **3.11%** | **0.4%** |
| $4,000 | 1.17% | 4.88% | 6.3% |
| $5,000 | 2.14% | 6.28% | **23.4%** |

From $214k the floor is barely a risk -- **the $10k daily limit is what binds**.
Past $4,000/day it dominates.

**Start at $2,000. Move to $3,000 once a month of live data tracks.**

---

## 6. OPEN ITEMS

1. **maker fill rate** -- never measured, outstanding since the patch plan
2. **FVG turnover** 297x/year = 25% annual fee drag
3. **survivorship** -- the panel holds coins that exist TODAY
4. **funded-account rules** -- confirm the floor does not reset or ratchet
   after passing. If it resets to the new balance, the growing-buffer logic
   disappears and sizing needs redoing.
