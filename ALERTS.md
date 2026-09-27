# alerts.py — the bot tells you, instead of you checking

## Set one webhook for the bot

Not a shared channel with the scanner. You want alerts to stand out, not
scroll past between pattern grids.

```bash
echo "export DISCORD_WEBHOOK_BOT='https://discord.com/api/webhooks/...'" >> ~/.bashrc
source ~/.bashrc
```

## The rule: SILENCE MEANS HEALTHY

Every failure on this project was silent. BOS produced zero weights for 300+
cycles while the log said `rebalanced=['short','bos']`. `run/` went 408 hours
stale while cycles kept "succeeding". Nobody was watching, because nothing
ever asked them to.

So this does the inverse. **If the bot is alive and nothing changed, it says
nothing.** Anything appearing in the channel is worth reading.

| level | when | repeat |
|---|---|---|
| heartbeat | a position opened or closed | only on change |
| alert | needs attention, trading continues | at most every 12h |
| critical | **trading has STOPPED** | every 4h until resolved, @-mentions |

Plus a `resolved` message when something that WAS critical clears -- otherwise
you never learn whether a problem fixed itself.

## THE WATCHDOG — the one nobody thinks of

**A bot that has stopped sends no alerts. That looks exactly like a healthy
quiet period.**

`watchdog()` must run from a SEPARATE cron entry so the alarm does not depend
on the thing it is watching:

```cron
# the bot
*/5 * * * * cd ~ && python3 doctor.py >/tmp/doc.txt 2>&1 && python3 bot/run_cycle.py

# the watchdog -- independent, shouts if the bot has not run in 2 hours
7 * * * * cd ~ && python3 -c "import alerts,json,os; \
  st=json.load(open('bot_state.json')) if os.path.exists('bot_state.json') else {}; \
  alerts.Alerter(st).watchdog(st.get('last_cycle_ts'))"
```

## Verified

| behaviour | result |
|---|---|
| repeat suppression, 4h window | just sent -> silent; 5h ago -> sends |
| watchdog, 30min since last cycle | silent (correct) |
| watchdog, 6h since last cycle | CRITICAL |
| cycle with nothing opened/closed | **silent** |
| cycle with a position opened | posts |

De-duplication state lives in the bot's state file, so it survives restarts --
otherwise a crash-loop would spam the channel every cycle.

## The 403 that costs an evening

Discord rejects urllib's default User-Agent with a 403. curl works, the script
does not, and the error says nothing useful. `_post()` sets a real User-Agent.

## Wiring

```python
from alerts import Alerter
a = Alerter(self.state, mention="<@YOUR_DISCORD_ID>")

# after the health check
halt = a.from_findings(findings)
if halt:
    return report          # no orders

# at the end of the cycle
a.cycle(equity=eq, daily_pnl=pnl, opened=opened, closed=closed,
        gross=gross, vol_target=config.daily_vol_target(eq))
self.state["last_cycle_ts"] = time.time()
```

## Running live at $300/day

This is the right first step and better than a longer paper period. At $300/day
a total failure costs a few thousand over a month against a $34,000 buffer --
and it exercises everything a diagnostic structurally cannot: order rejections,
rate limits, quantization against real instrument filters, and the kill switch
firing on real intraday equity.

What to watch in the first week:

1. **Does it trade at all?** Expect roughly 4 opens/day across 9 sleeves at 24
   coins. Zero for a day is the BOS failure repeating.
2. **Kill-switch frequency.** At $300/day it should essentially never fire.
   If it does, the sizing maths is wrong somewhere.
3. **Do live signals match `replay.py`?** Run the replay over the same window
   and diff the trade lists. They should be near-identical.
4. **Rejections and quantization errors.** These only appear live.
