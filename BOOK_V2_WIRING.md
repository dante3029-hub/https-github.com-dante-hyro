# bot/book_v2.py — the live entry point for the nine-sleeve book

One call returns every sleeve's weights:

```python
from bot.book_v2 import compute_all_weights, protective_stops
weights = compute_all_weights(self.state.tracker_states)   # {sleeve: {coin: weight}}
stops   = protective_stops(self.state.tracker_states)      # {coin: stop price}
```

`data_feed.py` changes by about five lines instead of being rewritten. delta
and relvol keep coming from the existing engine; this supplies the other seven.

## Verified on live data

| sleeve | positions | gross |
|---|---|---|
| skew | 12 | 1.000 |
| oirank | 10 | 1.000 |
| cascade | 0 | 0.000 (normal -- holds ~3.3% of the time) |
| fvg | 3 | 0.500 |
| sr / srflip / pattern | 0 | 0.000 (quiet bar) |

- tracker state JSON-serialisable (768 bytes)
- protective stops exposed for `sync_protective_stops()`
- a second cycle from the same state is idempotent: cross-sectional identical,
  event slots unchanged

## A robustness bug found while wiring it

**oirank silently returned FLAT when OI lagged price by even one bar.**

OI and price are topped up by separate fetchers. If OI is behind, the union
index leaves the last row all-NaN, `oi_scores` finds nothing, `rank_to_weights`
fails, and the sleeve reports zero positions -- indistinguishable from a quiet
market.

That is the same shape as every other silent failure here: BOS's impossible
condition, the spliced taker column, the OI column misalignment.

**Fix, two parts:**

1. `ffill(limit=3)` on the OI reindex, so a lag of a few hours is tolerated.
   The LIMIT matters -- without it the sleeve would quietly trade on
   arbitrarily stale positioning if the OI fetcher died.
2. An explicit `logger.error` when fewer than 12 of 24 coins have OI on the
   latest bar, so a starved feed says so instead of looking like a quiet
   market.

Confirmed both ways: with 10-day-old OI the sleeve correctly refuses to trade
and logs; with current OI it ranks 5 long / 5 short at gross 1.000.

## Four things that must not be "tidied up"

1. **The OI gate fires at ORDER time, inside the tracker** -- not at detection.
   60 raw sr breaks on SOL 6h: gating at the signal bar keeps 31, at the entry
   bar keeps 23, overlap 12 (29%). Both causal, different strategies.
2. **Universes are PINNED, never globbed.** Globbing once switched the book
   from 24 coins to 60 silently and skew went +0.88 -> -0.16.
3. **delta runs WIDE, everything else CORE24.** delta 1.11 on 24 vs 1.45 on 60;
   relvol and skew both degrade on the wide set.
4. **cascade returning all-zero is NORMAL.** `sleeve_health` knows a flat
   cascade is expected and a flat delta is not.
