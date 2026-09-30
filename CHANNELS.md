# Three Discord channels, three questions

| channel | env var | answers | cadence |
|---|---|---|---|
| **alerts** | `DISCORD_WEBHOOK_BOT` | is something broken? | only when it is |
| **cycles** | `DISCORD_WEBHOOK_REPORT` | what did it just do? | every 4h |
| **money** | `DISCORD_WEBHOOK_PNL` | is it actually working? | daily / weekly |

Separate on purpose. Mixed together you would read none of them: routine
reports train you to scroll past alerts, and alerts make routine reports feel
urgent.

**alerts is silent when healthy.** Every failure on this project was silent --
BOS produced zero weights for 300+ cycles while the log said success. So the
rule is inverted: anything appearing in that channel is worth reading.

```bash
export DISCORD_WEBHOOK_BOT='...'      # alerts  (alerts.py)
export DISCORD_WEBHOOK_REPORT='...'   # cycles  (report.py)
export DISCORD_WEBHOOK_PNL='...'      # money   (pnl_report.py)
```

---

## The money channel

```
**P&L · last 14 days** · 28 Sep
equity **$214,480** · +10,705 · fees -799
**$5,520** to target · **$34,480** above floor

**who paid**
`fvg     ` **   +4,820** ·  59 trades · 53% win
`oirank  ` **   +2,640** ·  20 trades · 65% win
`sr      ` **   +1,910** ·   6 trades · 67% win
`delta   ` **   +1,205** ·   2 trades · 100% win
`skew    ` **     +780**
`relvol  ` **     +410** ·   4 trades · 50% win
`cascade ` **     -240** ·   1 trades · 0% win
`pattern ` **     -290** ·   3 trades · 33% win
`srflip  ` **     -520** ·   4 trades · 25% win

**trading at the wrong rate** — usually a data problem, not a market one
· ⚠ pattern 6/mo vs ~19 expected

**by coin**
· best  ZEC +3,120 · SOL +1,980 · 1000PEPE +1,640
· worst LINK -410 · ADA -680 · DOT -1,240

**risk events**
· kill switch fired **2x** (expect ~16% of days at full size)
· no cycles halted
```

### The trade-rate check is the point

`⚠ pattern 6/mo vs ~19 expected` is the line this channel exists for. A sleeve
trading a third as often as the backtest says is almost always a DATA problem
upstream, not a market one -- and it is invisible in a P&L number.

fvg should fire ~126 times a month across the book. A quiet fvg means something
broke, not that markets were calm.

### Sleeves are judged against the Sharpe they were VALIDATED at

Not against zero. cascade has a NEGATIVE bull-regime Sharpe by design -- it
earns its place by being uncorrelated and firing when the price sleeves hurt.
A losing fortnight for cascade is expected; a losing fortnight for fvg is not.

### One-coin concentration warning

If any single coin exceeds 40% of absolute P&L the report says so. ZEC was
17.3% of gross profit in the research and that was already worth flagging.

---

## The cycle channel

```
**HyroTrader** · 28 Sep 11:56 UTC
equity **$214,480** · today **+580** · sizing $7,000/day

**what happened**
· `fvg    ` SHORT **SOL** @ 142.5023 · stop 148.2011 (4.0%) — gap down, OI expanding
· `fvg    ` LONG  **1000PEPE** @ 0.000012100 · stop 0.000011700 (3.3%) — gap up, OI expanding
· `sr     ` LONG  **LINK** @ 18.4020 · stop 17.1140 (7.0%) — resistance break @ 18.31, OI expanding
· `srflip ` closed **ZEC** @ 612.4000 — stop hit · -3.2%
· `pattern` closed **ADA** @ 0.8140 — 15 bars elapsed · +4.8%
· `oirank ` rebalanced · 5 long / 5 short · 7 position(s) changed

**the book**
`fvg     `  6 pos · gross 0.42 · due in 12h
`delta   ` 10 pos · gross 0.31 · due in 13d
...
**total** 60 positions · gross 1.93

**risk**
· floor $180,000 — **$34,480** of room (19.2%)
· daily limit $-10,000 — not approached
```

On a bad day the risk block becomes:

```
· daily limit $-10,000 — **44%** used
· kill switch $-5,000 — $645 away
```

### A formatting bug the preview caught

1000PEPE at $0.0000121 rendered as `0.0000` under a fixed 4dp format -- a
position with no usable price. `_px()` now scales precision to magnitude, since
crypto prices span eight orders of magnitude.
