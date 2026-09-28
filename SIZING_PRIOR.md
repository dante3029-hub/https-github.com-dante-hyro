# The cold-start sizing problem

## What the burn-in found

```
portfolio_layer.multipliers_v2 WARNING no usable history for
  cascade, fvg, oirank, pattern, skew, sr, srflip -- defaulting them to 1.0
```

A live bot starts with NO return history, so equal-risk cannot scale anything
and every sleeve defaults to 1.0.

**That is not a small problem.** The sleeves' daily vols span 6.5x:

| sleeve | daily vol | multiplier |
|---|---|---|
| fvg | 0.02930 | **0.351** |
| pattern | 0.01595 | 0.645 |
| sr | 0.01509 | 0.682 |
| cascade | 0.01269 | 0.811 |
| srflip | 0.01029 | 1.000 |
| delta | 0.01001 | 1.028 |
| skew | 0.00913 | 1.127 |
| relvol | 0.00745 | 1.381 |
| oirank | 0.00449 | **2.289** |

At a flat 1.0 each, fvg takes ~6.5x the risk oirank does. The book being run
would not be the book that was validated.

## The fix: use the vols the research already measured

`SLEEVE_VOL_DAILY` holds the full-sample daily vol of each sleeve, measured by
`hourly_sim` -- the same run that produced every number in the book (4.13
Sharpe, 3.7% maxDD).

Using them as the cold-start prior is **more** faithful than a trailing 60-day
window, not less: the research scaled on full-sample vol, so this reproduces
the validated sizing exactly.

**Verified on a cold start: every sleeve lands at 11.1% risk share.** That is
1/9 -- perfect equal-risk from the first cycle.

## It is a PRIOR, not a fixture

Once a sleeve has `MIN_OBS_FOR_LIVE = 60` days of live history, its own
trailing vol takes over completely.

**Deliberately not blended.** A blend would smear the moment live vol diverges
from research vol -- which is exactly what the paper month exists to detect. A
hard switch at 60 days means the divergence shows up as a step you can see in
the logs, not a drift you cannot.

## Why this class of bug keeps appearing

Six silent failures found on this project, all the same shape -- code runs,
logs say success, a sleeve quietly does nothing or the wrong thing:

1. BOS produced zero weights for 300+ cycles (impossible entry condition)
2. `run/` went 408 hours stale while cycles kept "succeeding"
3. taker_buy exceeded total volume on 60% of bars
4. OI files had a column misalignment after an append
5. oirank went flat whenever OI lagged price by one bar
6. **every new sleeve sized at 1.0 because it had no history yet**

Only the last two were caught before reaching live, and both by the burn-in and
health checks built today. That is the argument for the gate.
