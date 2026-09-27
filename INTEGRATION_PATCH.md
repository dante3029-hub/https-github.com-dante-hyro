# Integration patch — wiring the nine-sleeve book into the live bot

Five files change. Nothing here is invention: the architecture already
supports this, and the new sleeves slot into interfaces that exist.

**Apply in this order.** Each step leaves the bot runnable, so a failure is
isolated to the step that caused it.

---

## Step 0 — the one thing to do before anything else

**Turn BOS off.** It runs at 4h, which tested 0.33 against a random-direction
control of 0.74 -- the one setting where random beats it. The taker-data fix
means it can now actually fire, so it will start trading its worst
configuration on the next cycle.

This is the only item on the list that is costing money right now.

---

## Step 1 — `portfolio_layer/portfolio.py`

```python
SLEEVE_NAMES = (
    "delta", "relvol", "skew", "oirank",          # cross-sectional
    "cascade", "sr", "srflip", "pattern", "fvg",  # event-driven
)
```

`size_portfolio()` needs no change -- it is already generic over
`{sleeve: {coin: weight}}`.

---

## Step 2 — `bot/config.py`

```python
CADENCE_HOURS = {
    "delta":   336,   # was 168. IC strengthens to 30d; hold=7 was a grid-search artifact
    "relvol":  168,
    "skew":   1080,   # 45d hold
    "oirank":   72,   # 3d hold
    "cascade":  24,
    "sr":        6,
    "srflip":    6,
    "pattern":   6,
    "fvg":      12,
}
```

Plus the sizing block from `bot/config_v2.py`:

```python
TARGET_DAILY_VOL_DOLLARS   = 7_000.0
DEGEAR_TRIGGER_EQUITY      = 210_000.0
DEGEAR_DAILY_VOL_DOLLARS   = 3_000.0
KILL_SWITCH_DOLLARS        = -5_000.0   # was -3_000 (0.6 sigma at $7k/day,
                                        # would fire every ~4 days)
```

---

## Step 3 — `bot/data_feed.py`

`MarketSnapshot` gains the new weights, and `get_snapshot` takes the new
tracker states instead of short/bos:

```python
@dataclass
class MarketSnapshot:
    as_of_date: dt.date
    signal_snapshot: object
    sleeve_histories: dict
    event_weights: dict      # {sleeve: {coin: weight}} for sr/srflip/pattern/fvg
    xs_weights: dict         # {sleeve: {coin: weight}} for skew/oirank/cascade
    data_source: str
```

```python
def get_snapshot(self, as_of_date, tracker_states: dict) -> MarketSnapshot:
    ...
```

`tracker_states` is `{sleeve: state_dict}` -- one per event sleeve. The
trackers mutate those dicts IN PLACE, exactly as ShortSleeveTracker does, so
nothing is reassigned.

Inside, build the event weights with the tracker:

```python
from bot.price_event_tracker import PriceEventTracker

event_weights = {}
for s in ("sr", "srflip", "pattern", "fvg"):
    tr = PriceEventTracker(s, tracker_states[s],
                           bars_fn=lambda c, _s=s: self._bars(c, PARAMS[_s]['tf']),
                           oi_fn=self._oi_expanding if PARAMS[s]['needs_oi'] else None)
    event_weights[s] = tr.update(self.coins_for(s))
```

and the cross-sectional ones:

```python
from signal_engine.sleeve_skew    import latest_target_weights as skew_w
from signal_engine.sleeve_oirank  import latest_target_weights as oi_w
from signal_engine.sleeve_cascade import latest_target_weights as cas_w

xs_weights = {
    "skew":    dict(zip(coins, skew_w(R_daily)[0])),
    "oirank":  dict(zip(coins, oi_w(OI_daily)[0])),
    "cascade": dict(zip(coins, cas_w(R_daily)[0])),
}
```

### The OI gate MUST stay at order time

`PriceEventTracker` queries OI when it OPENS a slot, not when the signal is
detected. Do not "optimise" that by passing OI into `latest_signal()`.

On SOL 6h there are 60 raw sr breaks. Gating at the SIGNAL bar keeps 31;
gating at the ENTRY bar keeps 23; the two sets overlap by **12 -- 29%**. Both
are causal. They are different strategies.

---

## Step 4 — `bot/orchestrator.py`

```python
snap = self.data_feed.get_snapshot(as_of_date, self.state.tracker_states)

raw_weights_this_cycle = {
    "delta":  snap.signal_snapshot.delta_weights,
    "relvol": snap.signal_snapshot.relvol_weights,
    **snap.xs_weights,
    **snap.event_weights,
}
```

The cadence-gating loop below it is unchanged -- it already iterates
`SLEEVE_NAMES`.

### Protective stops

Replace the short/bos stop calls with:

```python
stops = {}
for s in ("sr", "srflip", "pattern", "fvg"):
    tr = PriceEventTracker(s, self.state.tracker_states[s], ...)
    stops.update({to_bybit_symbol(c): px
                  for c, px in tr.protective_stops().items()})
sync_protective_stops(client, stops, dry_run=self.dry_run)
```

Stops live on the exchange, not in the bot. If the bot dies, the stop still
works.

### The health gate

At the end of the cycle, before any order goes out:

```python
from sleeve_health import run_health_check, format_report, CRITICAL, book_expectations

findings = run_health_check(
    book_expectations(TAKER_DIR, OI_DIR),
    weights_by_sleeve=sized_input_weights,
    flat_cycles=self.state.flat_cycles,
    hours_since_rebalance={s: _hours_since(self.state.last_rebalance.get(s), now)
                           for s in SLEEVE_NAMES},
    sanity_files={"delta": f"{TAKER_DIR}/SOL_1h.csv"},
)
logger.info(format_report(findings))
if any(f.severity == CRITICAL for f in findings) and config.HALT_ON_CRITICAL:
    report["halted"] = True
    return report          # <-- NO ORDERS
```

**This is the point of the whole exercise.** BOS produced zero weights for
300+ cycles while the log said `rebalanced=['short','bos']`. `run/` went 408
hours stale while cycles kept "succeeding". The taker column had
taker_buy > total volume on 60% of bars. All three were silent. A sleeve that
cannot be verified does not trade.

---

## Step 5 — `bot/burn_in.py`

Change the two `get_snapshot` call sites to pass `tracker_states`, and the
mock feed's signature to match. Nothing else -- the four phases are
sleeve-agnostic.

---

## Then run, in this order

```bash
# 1. offline: no credentials, no network. Proves cycle sequencing, cadence,
#    state persistence, restart recovery, kill switch, compliance flattening,
#    dollar->contract conversion, quantization, position diffing.
python3 bot/burn_in.py --mode offline --cycles 12

# 2. demo: real Bybit testnet. Proves connectivity, auth, signing, rate
#    limits, real fills, exchange-side rejections. MUST run on your server --
#    Bybit returns HTTP 403 from the research sandbox (CloudFront geo-block).
BYBIT_API_KEY=... BYBIT_API_SECRET=... python3 bot/burn_in.py --mode demo --cycles 12

# 3. paper month: dry_run=False, paper_trade=True against live data. Reads the
#    real exchange, builds fully-quantized orders, sends nothing. Compare the
#    logged orders against replay.py output.

# 4. fund it -- at $3,000/day, not $7,000. Move up once a month of live data
#    tracks the simulator.
```

---

## What each stage can and cannot prove

| stage | proves | does NOT prove |
|---|---|---|
| replay.py | signal timing, slot lifecycle, restart | anything about the exchange |
| burn-in offline | cycle logic, kill switch, quantization | connectivity, fills, rejections |
| burn-in demo | auth, signing, rate limits, real fills | behaviour under real P&L pressure |
| paper month | live signal timing vs simulator | slippage at size, funding |
| funded | everything | -- |

Skipping a stage does not save time. It moves the discovery of a bug from a
test to a live position.
