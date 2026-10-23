#!/usr/bin/env python3
"""
tf_sweep.py -- is V29 broken, or just run at the wrong speed?

THE OBSERVATION, from the account owner looking at the trade count rather
than the Sharpe: "9,100 trades -- the market doesn't give 9,000 perfect
opportunities to make money."

He is right, and it is the sharpest point made about V29. ~700 signals per
coin over 3.7 years is a trade every 1.9 days. Trends do not start every two
days. The 5-bar confirmation on 1-HOUR bars makes almost any wiggle look
like a trend beginning.

WHY THAT IS FATAL RATHER THAN MERELY INEFFICIENT: the fee is 0.170% per
round trip REGARDLESS of how big the move is. Trading tiny 1h wiggles at a
fixed cost is structurally losing no matter how good the signal is. V29's
gross edge is +0.033% per trade against a 0.170% cost -- but that is 3.3bp
of a move that is itself small.

On a 12h bar the ATR is roughly sqrt(12) ~ 3.5x the 1h ATR, so the move
captured per trade is several times larger while the fee does not move at
all. If the gross edge scales with the size of the move, it should scale
with the timeframe -- and the fee ratio improves as 1/sqrt(tf).

That is a testable, mechanical prediction, not a hope.

PRE-REGISTERED: timeframes 1, 2, 4, 6, 12, 24 hours. Everything else
identical -- same Supertrend parameters, same EMA/ADX filters, same stop
and TP multiples, same BTC regime overlay, same coins. ONE variable.

WHAT WOULD COUNT AS SUCCESS: gross per trade clearing 0.170% (tradeable as
it stands) or at least 0.040% (tradeable with maker fills). And the trade
count has to fall roughly as 1/tf -- if it does not, the resampling is not
doing what it should.

WHAT WOULD NOT: a single timeframe spiking while its neighbours do not.
That is the fitted-parameter shape this project has seen three times
(oirank's 7-day lookback at 1.41 against ~0.95 either side). A real effect
is monotone or a plateau.

USAGE
    python3 -u tf_sweep.py
    python3 -u tf_sweep.py --all-coins
"""
from __future__ import annotations

import sys

import numpy as np

import v29_test as V

# Fine grid around the 1h->12h rise. A plateau and a spike are
# indistinguishable at 2x spacing, and the first sweep produced exactly
# that ambiguity: 0.063 -> 0.159 -> 0.707 -> -1.812.
TFS = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24)


def log(m=""):
    print(m, flush=True)


def main() -> int:
    bar_mkt = 2 * V.FEE * 100
    bar_mak = 2 * 0.0002 * 100
    log("  V29 at six timeframes. Everything else identical.")
    log(f"  Fee is {bar_mkt:.3f}% per round trip at ANY timeframe --")
    log(f"  that is the whole point.\n")
    log(f"  {'tf':>5}{'trades':>9}{'net/trade':>12}{'gross/trade':>13}"
        f"{'total':>10}{'buy&hold':>10}{'beat':>7}{'clears':>10}")

    rows = []
    for tf in TFS:
        V.TF = tf
        V._CACHE.clear()
        argv = sys.argv
        sys.argv = [a for a in argv if a not in ("--tf",)] + ["--no-null"]
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                V.main()
            txt = buf.getvalue()
        finally:
            sys.argv = argv
        n = tot = 0
        bh = float("nan")
        nets = []
        for ln in txt.splitlines():
            p = ln.split()
            if len(p) >= 7 and p[1].isdigit() and p[2].endswith("%"):
                try:
                    cnt = int(p[1]); mean = float(p[2].rstrip("%"))
                except ValueError:
                    continue
                n += cnt
                nets += [mean] * cnt
            if ln.strip().startswith("buy & hold"):
                try:
                    bh = float(ln.split()[-1].rstrip("%"))
                except ValueError:
                    pass
            if ln.strip().startswith("V29 total"):
                try:
                    tot = float(ln.split()[-1].rstrip("%"))
                except ValueError:
                    pass
        if not nets:
            log(f"  {tf:>4}h   no trades")
            continue
        net = float(np.mean(nets))
        gross = net + bar_mkt
        clears = ("MARKET" if gross > bar_mkt else
                  "maker" if gross > bar_mak else "no")
        rows.append((tf, n, net, gross, tot, bh))
        beat = "YES" if (np.isfinite(bh) and tot > bh) else "no"
        log(f"  {tf:>4}h{n:>9,}{net:>11.3f}%{gross:>12.3f}%"
            f"{tot:>9.1f}%{bh:>10.1f}%{beat:>7}{clears:>10}")

    if len(rows) < 3:
        return 0
    log("")
    log("  " + "=" * 62)
    g = np.array([r[3] for r in rows]); t = np.array([r[0] for r in rows])
    c = np.array([r[1] for r in rows], dtype=float)
    log(f"  trade count falls {c[0]/c[-1]:.0f}x from 1h to 24h "
        f"(1/tf would be {t[-1]/t[0]:.0f}x)")
    mono = all(g[i] <= g[i + 1] + 0.02 for i in range(len(g) - 1))
    best = int(np.argmax(g))
    log(f"  gross per trade: " + " -> ".join(f"{x:+.3f}" for x in g))
    if mono:
        log(f"  MONOTONE with timeframe -- the mechanical prediction holds.")
    else:
        nb = [g[i] for i in (best - 1, best + 1) if 0 <= i < len(g)]
        if nb and g[best] - max(nb) > 0.05:
            log(f"  SPIKE at {t[best]}h, {g[best]-max(nb):+.3f}% above its")
            log(f"  neighbours. That is the fitted-parameter shape, not an")
            log(f"  effect. Do not take the maximum.")
        else:
            log(f"  non-monotone but no spike -- treat as a plateau.")
    spike = False
    nb = [g[i] for i in (best - 1, best + 1) if 0 <= i < len(g)]
    if nb and (g[best] - max(nb)) > 0.05:
        spike = True

    # A spike DISQUALIFIES the maximum. Previously this printed the spike
    # warning and then "V29 was not broken" immediately after, which is two
    # opposite conclusions in consecutive lines.
    if spike:
        log(f"  The best timeframe is DISQUALIFIED as a spike. Judge the")
        log(f"  curve by its NEIGHBOURS, not its peak: the second-best")
        log(f"  cluster is what you could actually trade.")
        order = np.argsort(g)[::-1]
        second = [i for i in order if i != best]
        if second:
            i2 = second[0]
            log(f"  Next best: {t[i2]}h at {g[i2]:+.3f}% gross "
                f"({'clears' if g[i2] > bar_mkt else 'does NOT clear'} "
                f"{bar_mkt:.3f}%).")
    elif g.max() > bar_mkt:
        log(f"  {t[best]}h clears the MARKET-order bar ({bar_mkt:.3f}%) and")
        log(f"  is NOT a spike -- its neighbours support it.")
        log(f"  V29 may have been run at the wrong speed.")
    elif g.max() > bar_mak:
        log(f"  {t[best]}h clears the MAKER bar ({bar_mak:.3f}%) but not")
        log(f"  market. Tradeable only with passive fills.")
    else:
        log(f"  No timeframe clears even {bar_mak:.3f}%. The speed was not")
        log(f"  the problem either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
