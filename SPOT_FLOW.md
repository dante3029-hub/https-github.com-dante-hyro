# Spot flow — candidate 10th sleeve

## Why this one is different

Seven candidate 10th sleeves were tested and rejected (EMA retest, calendar
pivots ×4, BTC pairs, funding momentum, funding fade). Every one of them
correlated 0.4–0.7 with `sr`. The reason was always the same: they all re-read
**price**. A different formula on the same tape is not a different signal.

The nine live sleeves read exactly three inputs:

| input | sleeves |
|---|---|
| perp price | sr, srflip, pattern, fvg, cascade |
| perp taker delta | delta, relvol |
| perp open interest | oirank |
| perp returns distribution | skew |

All perp. **Spot flow is a tape the book has never read.**

## The hypothesis

Spot and perp taker flow answer different questions:

- **Perp delta** = who is pressing leveraged risk right now.
- **Spot delta** = who is paying cash to actually own the coin.

The informative state is when they **disagree**:

```
spot delta > 0, perp delta < 0   →  cash absorbing leverage.
                                    Someone is buying the coin while the
                                    derivatives crowd is positioned against
                                    them. Constructive.

spot delta < 0, perp delta > 0   →  holders distributing into leveraged
                                    chasers. Supply moving to the weakest
                                    hands. Destructive.
```

Agreement (both signs the same) carries much less information — it is just
directional consensus, which `delta` already ranks.

## Why it should decorrelate

`delta` ranks perp delta. A spot sleeve ranks `spot_delta − perp_delta`. Those
are mechanically different series, and the second is *orthogonal by
construction* to the first whenever the two venues move together — which is
most of the time. The signal only fires on the residual.

This is the first candidate with a structural reason to decorrelate rather
than a hope that it will.

## Data

`fetch_spot.py` pulls Binance **spot** 1h klines, same fields as the futures
endpoint `fetch_taker.py` uses:

```
field 5  volume              (base units)
field 9  taker buy base volume
delta = 2*taker_buy - volume
```

Both numbers come from one response, so they cannot drift apart — the
single-venue guarantee that fixed the `clean_panel` splice bug (where
`taker_buy > volume` on 60% of bars because Binance taker volume had been
spliced against Bybit price volume).

Output: `~/spot_data/<COIN>_spot_1h.csv`, identical columns to
`~/taker_data/<COIN>_1h.csv`, filed under the **perp** coin name so the two
align 1:1 by filename.

## The symbol trap — read this before using the data

Perps use 1000× contracts for small-unit coins (`1000PEPEUSDT`,
`1000SHIBUSDT`, `1000BONKUSDT`, `1000FLOKIUSDT`, `1000RATSUSDT`). Spot has no
such pair — it is `PEPEUSDT`, `SHIBUSDT`, etc.

So for those coins the spot **price is 1000× smaller** and the spot base
**volume is 1000× larger** than the perp series.

- **Never** compare spot vs perp price levels or raw volumes for these coins.
- **Do** compare scale-free quantities: `delta/volume`, `sign(delta)`,
  z-scores of `delta/volume`. Unit-invariant, which is what a flow-divergence
  signal should be built on anyway.

`--verify` does not catch a scale mistake. The unit discipline has to live in
the research code.

## Coins with no Binance spot pair

Perp-only, excluded automatically: `FARTCOIN ASTER XPL HYPE AIXBT MOODENG
PNUT POPCAT PUMP KAITO GRASS VIRTUAL PENGU TRUMP`. The script also checks
`exchangeInfo` live, so anything newly listed or delisted is handled without
a code change.

This matters for universe construction: a spot sleeve can only trade coins
that have a spot market, which is a *smaller* universe than CORE24. If the
sleeve only works on the handful of coins that remain, that is a red flag, not
a result.

## Commands

```bash
cd ~/bot_hyrotrader_v1
git pull
python3 -u fetch_spot.py --check              # coverage plan, no fetching
python3 -u fetch_spot.py --run --limit 3      # 3 coins first
python3 -u fetch_spot.py --verify             # audit those 3
python3 -u fetch_spot.py --run                # the rest
python3 -u fetch_spot.py --verify             # audit everything
```

Resumable — a coin already in `~/spot_data` is skipped. Delete its file to
refetch. `--verify` exits 0 clean / 1 warnings / 2 do-not-use, same convention
as `doctor.py`.

## Status

Data layer written and unit-tested offline (paths, symbol mapping, verify
logic including the `taker_buy > volume` and overlap checks). **Not yet
fetched. Hypothesis not yet tested.** No sleeve is wired, and none will be
until the divergence signal clears the same bar every other sleeve had to:

1. positive Sharpe out of sample, not just full sample
2. both halves positive
3. correlation < 0.3 against all nine existing sleeves
4. survives the multiple-testing threshold (E[max Sharpe] ≈ 1.45 from 300
   noise trials — a 1.3 Sharpe candidate is noise)
5. works on more than a handful of coins
