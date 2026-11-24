#!/usr/bin/env python3
"""
fvg_overlap.py -- is the 4h FVG first-revisit trade new, or a copy of what the
book already holds?

CANDIDATES.md states the structural problem plainly: "Five of the eight
sleeves are long-only price structure on 6-12h bars... They correlate 0.4-0.6
with each other by construction. Any new sleeve of that shape duplicates
rather than adds." Breakout (corr 0.56 to sr) and EMA retest (0.55) were both
rejected for exactly this. The FVG revisit is price structure on 4h bars, so
it is the obvious suspect.

Every sleeve here is pushed through the SAME hourly engine
(sleeves_book.positions_to_hourly + hourly returns + fees), so the
correlations compare like with like. Existing sleeves use their book configs
(BOOK_VERIFIED.md); the revisit uses the config that passed the holdout.

Keep: only if correlation to every existing price-structure sleeve is low
(< 0.3, the level OI rank sits at against them) AND it raises the combined
Sharpe of the group it would join.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sleeves_book as sb          # noqa: E402
import fvg_revisit as FR           # noqa: E402

sb.TAKER = os.path.join(HERE, "taker_data")
sb.OIDIR = os.path.join(HERE, "oi_data")
FR.ROUND_TRIP = sb.FEE * 2         # same cost model as the book sleeves


def revisit_weights(coins, hidx, cols):
    """The holdout-confirmed config, converted to hourly positions."""
    import event_audit as EA
    d, _ = FR.collect(coins, 4, EA.load_bars, "revisit1", "gap", "trail")
    trades = []
    for row in d.itertuples(index=False):
        t0 = row.ts
        t1 = t0 + pd.Timedelta(hours=4 * (int(row.bars) + 1))
        trades.append((row.coin, t0, t1, int(row.side)))
    W = sb.positions_to_hourly(trades, hidx, cols, per_position=1.0 / 6,
                               max_gross=1.0)
    return W, len(trades)


def hourly_pnl(W, RH):
    g = (W.shift(1) * RH).sum(axis=1)
    c = W.diff().abs().sum(axis=1) * sb.FEE
    return (g - c).fillna(0.0)


def sharpe_daily(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0


def main() -> int:
    coins = [c for c in sb.CORE24
             if os.path.exists(os.path.join(sb.TAKER, f"{c}_1h.csv"))]
    hidx, PX, RH = sb.hourly_panel(coins)
    cols = list(PX.columns)
    print(f"  universe: {len(cols)} coins (book CORE24), "
          f"{hidx[0].date()} -> {hidx[-1].date()}\n")

    W = {}
    W["fvg_12h"], n1 = sb.fvg_hourly(coins, hidx, cols, tf=12, stop_atr=1.5,
                                     hold=10)
    W["sr_6h"], n2 = sb.sr_hourly(coins, hidx, cols, tf=6, stop_atr=3.0,
                                  hold=15)
    W["pattern_6h"], n3 = sb.pattern_hourly(coins, hidx, cols, tf=6,
                                            stop_atr=2.0, hold=15)
    W["revisit_4h"], n4 = revisit_weights(coins, hidx, cols)
    for k, n in zip(W, (n1, n2, n3, n4)):
        print(f"  {k:<12} {n:>6} trades")

    daily = pd.DataFrame({k: hourly_pnl(w, RH).resample("D").sum()
                          for k, w in W.items()})
    daily = daily.loc[daily.index >= daily.index[0] + pd.Timedelta(days=60)]

    print("\n  standalone on the same engine (daily Sharpe, ann. return):")
    for k in daily:
        print(f"    {k:<12} {sharpe_daily(daily[k]):>6.2f}  "
              f"{daily[k].mean()*365*100:>7.1f}%")

    print("\n  daily-return correlation:")
    print(daily.corr().round(2).to_string())

    base = ["fvg_12h", "sr_6h", "pattern_6h"]
    eq3 = daily[base].mean(axis=1)
    eq4 = daily[base + ["revisit_4h"]].mean(axis=1)
    print(f"\n  price-structure group, equal weight:")
    print(f"    existing 3     Sharpe {sharpe_daily(eq3):.2f}")
    print(f"    + revisit      Sharpe {sharpe_daily(eq4):.2f}")
    worst = daily.corr().loc["revisit_4h", base].max()
    keep = worst < 0.3 and sharpe_daily(eq4) > sharpe_daily(eq3)
    print(f"\n  max corr to existing: {worst:.2f}  ->  "
          f"{'ADD: orthogonal and accretive' if keep else 'DO NOT ADD'}")
    daily.to_csv(os.path.join(HERE, "fvg_overlap_daily.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
