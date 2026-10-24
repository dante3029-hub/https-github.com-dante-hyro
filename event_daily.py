#!/usr/bin/env python3
"""
event_daily.py -- pattern and sr as daily series, so the survivors book
is complete.

WHY. survivors.py measures the book on days-to-target = DD_unit/mu_daily
and reports 268 days for the verified cross-sectional sleeves alone,
184 blended with the 12h short book. Both numbers are missing pattern and
sr -- two of the four sleeves that survived their audits -- because those
are EVENT sleeves: they emit a list of trades per coin, not a weight
vector per rebalance, and the days metric needs a daily series.

short_risk already does that conversion for the short book. This applies
the same conversion to the event sleeves, through the SAME function
(short_risk.legs_to_daily), because that arithmetic has already been
silently wrong once in this project -- the sign of a short leg's daily
path -- and a second copy is how it goes wrong again somewhere nobody
checks.

FIDELITY. simulate() below must match event_audit.simulate() exactly, or
the daily series describes a different strategy than the one that passed
the audit. The only addition is that this version returns the EXIT BAR
and the stop distance as well as the return, which the daily mapping
needs and the audit did not. The exit rule is reproduced line for line:

    ATR through the PRIOR bar; entry at the open of bar eb; stop at
    stop_atr * ATR; a time exit at the close of bar eb+hold; the stop
    checked from the ENTRY bar onward, because a stop can be hit the
    moment you are in.

LIVE PARAMETERS, from event_audit.PARAMS which reads them out of
signal_engine/sleeve_price_events.py:

    sr        tf=6   stop 3.0 ATR   hold 15   which='break'
    pattern   tf=6   stop 2.0 ATR   hold 15   detector FastPattern,
                                              bull_only, indexed by the
                                              DETECTION bar

pattern is NOT the sr break detector. Running sleeve_sr for it once gave
pattern and sr identical trade counts (2,290) and meant pattern was never
actually tested. And FastPattern indexes by the detection bar -- entering
at the historical break bar it also reports is lookahead and once
inflated a backtest from 1.04 to 6.06. real_entries() in event_audit
handles both points and is reused here rather than reimplemented.

USAGE
    python3 -u event_daily.py                 # both sleeves, reports each
    python3 -u event_daily.py --sleeve sr
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import event_audit as EA
import short_risk as SR


def log(m=""):
    print(m, flush=True)


def simulate_with_exit(o, h, l, c, atr, eb, stop_atr, hold, long=True):
    """event_audit.simulate(), plus the exit bar and the stop fraction.

    Kept deliberately line-for-line with the audited version. If these
    two ever diverge, the daily series stops describing the sleeve that
    passed the test and no amount of downstream care recovers that.
    """
    n = len(o)
    if eb >= n - 1:
        return None
    a = atr[eb - 1] if eb > 0 else np.nan       # ATR through the PRIOR bar
    if not np.isfinite(a) or a <= 0:
        return None
    ent = o[eb]
    if not np.isfinite(ent) or ent <= 0:
        return None
    stp = ent - stop_atr * a if long else ent + stop_atr * a
    ex = min(eb + hold, n - 1)
    px = c[ex]
    for j in range(eb, min(eb + 1 + hold, n)):
        if (long and l[j] <= stp) or ((not long) and h[j] >= stp):
            px, ex = stp, j
            break
    ret = (px / ent - 1.0) if long else (ent / px - 1.0)
    return ret, ex, abs(stop_atr * a / ent)


def sleeve_series(sleeve, coins=None, fee=None, quiet=False):
    """Daily PnL series for one event sleeve at 1.0x equity per leg."""
    p = EA.PARAMS[sleeve]
    fee = SR.V.FEE if fee is None else fee
    if coins is None:
        coins = SR.universe()
    legs, daily_ret, skipped = [], {}, 0
    for coin in coins:
        df = EA.load_bars(coin, p["tf"])
        if df is None or len(df) < 300:
            skipped += 1
            continue
        ebs, err = EA.real_entries(coin, sleeve, df)
        if ebs is None:
            skipped += 1
            continue
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        atr = EA.atr(h, l, c) if hasattr(EA, "atr") else None
        if atr is None:
            import v29_test as V
            atr = V.atr(h, l, c, 14)
        # daily returns for the mapping, from the 1h file not the resample
        raw = SR.V._load(f"{SR.V.PERP}/{coin}_1h.csv")
        if raw is None:
            skipped += 1
            continue
        daily_ret[coin] = raw["close"].resample("1D").last().dropna(
        ).pct_change()
        for eb in ebs:
            r = simulate_with_exit(o, h, l, c, atr, eb,
                                   p["stop_atr"], p["hold"], long=True)
            if r is None:
                continue
            ret, ex, slf = r
            legs.append((coin, df.index[eb], df.index[ex], 1,
                         ret - 2 * fee, slf))
    if not legs:
        return None, 0
    pnl, conc = SR.legs_to_daily(legs, daily_ret)
    if not quiet:
        cv = conc[conc > 0].values
        log(f"    {sleeve:<9}{len(legs):>7} legs  "
            f"{len(coins)-skipped}/{len(coins)} coins  "
            f"median {int(np.median(cv)) if len(cv) else 0} concurrent")
    return pnl, len(legs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleeve", default=None,
                    choices=sorted(EA.PARAMS), help="default: pattern and sr")
    a = ap.parse_args()
    want = [a.sleeve] if a.sleeve else ["pattern", "sr"]

    import survivors as SV
    log("  event sleeves as daily series")
    log("  params read from event_audit.PARAMS (= the live settings)\n")
    for s in want:
        p = EA.PARAMS[s]
        log(f"    {s}: tf={p['tf']}h stop={p['stop_atr']}ATR "
            f"hold={p['hold']} which={p.get('which')}")
    log("")
    out = {}
    for s in want:
        pnl, n = sleeve_series(s)
        if pnl is not None:
            out[s] = pnl
    if not out:
        log("  no sleeve produced legs")
        return 2
    log("")
    log(f"    {'sleeve':<10}{'mu/day':>10}{'DD unit':>10}{'days':>8}"
        f"{'months':>8}")
    for s, v in out.items():
        d, dd, mu, _ = SV.days_of(v)
        log(f"    {s:<10}{mu*100:>9.3f}%{dd*100:>9.0f}%"
            f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8}"
            f"{(f'{d/30.4:.1f}' if np.isfinite(d) else '-'):>8}")
    if len(out) > 1:
        ks = list(out)
        j = out[ks[0]].index
        for k in ks[1:]:
            j = j.intersection(out[k].index)
        blend = pd.Series(np.mean(
            [(out[k].loc[j] / out[k].loc[j].std()).values for k in ks],
            axis=0), index=j)
        d, dd, mu, _ = SV.days_of(blend)
        log(f"    {'both':<10}{mu*100:>9.3f}%{dd*100:>9.0f}%"
            f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8}"
            f"{(f'{d/30.4:.1f}' if np.isfinite(d) else '-'):>8}")
        log("")
        log(f"    rho(pattern, sr) = "
            f"{float(np.corrcoef(out[ks[0]].loc[j], out[ks[1]].loc[j])[0,1]):+.3f}")
    log("")
    log("  These feed survivors.py, which can now build all four verified")
    log("  sleeves instead of two. Run: python3 -u survivors.py --events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
