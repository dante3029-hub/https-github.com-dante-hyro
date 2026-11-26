#!/usr/bin/env python3
"""regime_check.py -- how each crypto candidate does in bull runs, steep declines
and chop. Descriptive only: the regime label uses ONLY past data (equal-weight
basket's 60-day return measured to t-1), and nothing is tuned on it.
  bull  basket up more than 20% over the last 60 days
  bear  basket down more than 20%
  chop  in between
"""
import zlib, numpy as np, pandas as pd
import oos_book as OB, published as PB
P = OB.build_panels(); PX, R, FN, dates = P["PX"], P["R"], P["FN"], P["dates"]; C = PX.shape[1]
lp = np.log(PX)
mom = OB.run_sleeve(lambda t: lp[t-1]-lp[t-22] if t >= 23 else np.full(C, np.nan), PX, R, FN, 6, 7, 7, None)
liq = np.nanmedian(P["QV"], axis=0); top30 = np.zeros(C, bool); top30[np.argsort(np.nan_to_num(liq, nan=-1))[-30:]] = True
trend = PB.trend_book(PX, R, FN, top30)
fn = pd.DataFrame(FN).rolling(7, min_periods=5).mean().values
carry = OB.run_sleeve(lambda t: -fn[t-1] if t >= 8 else np.full(C, np.nan), PX, R, FN, 5, 7, 7, None, np.asarray(P["has_fn"], bool))
oi = OB.run_sleeve(OB.sig_oi(P["OI"]), PX, R, FN, 5, 3, 3, OB.STOP_PLATEAU, P["has_oi"])
basket = pd.Series(np.nanmean(np.nan_to_num(R, nan=np.nan), axis=1), index=dates).fillna(0)
past = (1 + basket).rolling(60).apply(np.prod, raw=True).shift(1) - 1
reg = np.where(past > 0.2, "bull", np.where(past < -0.2, "bear", "chop"))
S = pd.DataFrame({"momentum": mom, "trend": trend, "carry": carry, "oi_rank": oi}, index=dates)
S = S[dates >= dates[OB.WARMUP]]; reg = pd.Series(reg, index=dates)[S.index]
print(f"\n  days: " + "  ".join(f"{k} {int((reg == k).sum())}" for k in ("bull", "chop", "bear")))
print(f"  basket return/yr in each: " + "  ".join(f"{k} {basket[S.index][reg == k].mean()*365*100:+.0f}%" for k in ("bull", "chop", "bear")))
print(f"\n  {'strategy':<10}" + "".join(f"{k + ' %/yr':>13}" for k in ("bull", "chop", "bear")))
for c in S:
    print(f"  {c:<10}" + "".join(f"{S[c][reg == k].mean()*365*100:>12.0f}%" for k in ("bull", "chop", "bear")))
print("\n  daily correlations:"); print(S.corr().round(2).to_string())
S.to_csv("results/crypto_candidates_daily.csv")
