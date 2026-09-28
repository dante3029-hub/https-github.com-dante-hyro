# report.py — the bot reports back like a colleague

`alerts.py` shouts when something breaks. This is the other half: after every
cycle it says what it did, why, and what it is watching.

**Separate channel from alerts.** Mixing routine reports into the alert channel
trains you to scroll past both.

```bash
export DISCORD_WEBHOOK_REPORT='https://discord.com/api/webhooks/...'
```

## What a normal cycle looks like

```
**HyroTrader** · 28 Sep 11:48 UTC
equity **$203,775** · today **+420** · sizing $3,000/day

**what happened**
· `fvg    ` SHORT **SOL** @ 142.5023 · stop 148.2011 (4.0%) — gap down, OI expanding
· `sr     ` LONG  **LINK** @ 18.4020 · stop 17.1140 (7.0%) — resistance break, OI expanding
· `srflip ` closed **ZEC** @ 612.4000 — stop hit · -3.2%
· `pattern` closed **ADA** @ 0.8140 — 15 bars elapsed · +4.8%
· `oirank ` rebalanced · 5 long / 5 short · 7 position(s) changed

**the book**
`fvg     `  6 pos · gross 0.42 · due in 12h
`delta   ` 10 pos · gross 0.31 · due in 13d
`oirank  ` 10 pos · gross 0.30 · due in 3d
...
**total** 60 positions · gross 1.93

**risk**
· floor $180,000 — **$23,775** of room (13.2%)
· daily limit $-10,000 — not approached

**needs attention**
· ⚠ pattern flat 9 cycles (expected roughly every 4)

**watching**
· oirank using researched vols — 12 of 60 days of live history
```

1,159 characters. Discord caps at 2,000, so long sections truncate with a count
rather than being dropped silently.

## On a bad day

```
**risk**
· floor $180,000 — **$19,420** of room (10.8%)
· daily limit $-10,000 — **44%** used
· kill switch $-5,000 — **$645 away**
```

and once it fires:

```
· ⚠ kill switch $-5,000 — **TRIPPED**, flat for the rest of the session
```

and when the health gate stops the cycle:

```
🔴 **CYCLE HALTED** — 2 CRITICAL: data STALE — SOL_1h.csv newest bar 431h old
**No orders were placed.**
```

## Three design choices

**Numbers without reasons are noise.** "opened SOL short" tells you nothing you
can act on. "SHORT SOL @ 142.50, stop 4.0% away, gap down with OI expanding"
tells you whether the bot is doing what you think it does. Every entry carries
its trigger and its stop distance.

**Risk is stated in the firm's own terms, every cycle**, whether or not
anything happened. Not "drawdown 2.1%" but "$23,775 above the $180,000 floor"
and "44% of the daily limit used". Those two numbers are what end the
evaluation.

**This posts every cycle by design.** It is a status report, not an exception --
which is exactly why it must not share a channel with alerts.

## Wiring

```python
from report import from_cycle
rep = from_cycle(report, self.state, config, findings)
rep.post()
```

`from_cycle` is deliberately tolerant of missing keys: **a reporting layer must
never be the thing that breaks a trading cycle.**
