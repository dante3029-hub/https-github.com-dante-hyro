#!/usr/bin/env python3
"""
oos_diag.py -- two follow-ups to oos_book.py.

1. WHY DOES RANDOM EARN 1.3-2.1 SHARPE IN 2023?
   A random dollar-neutral long/short book should earn roughly nothing minus
   fees. Hypothesis: the per-leg stop is a hidden directional bet. A short leg
   that is pumped 60% against gets flattened and sits at zero until the next
   rebalance, while long legs are never stopped by pumps -- so the book drifts
   NET LONG, and 2023 was a +102% market. Test: run the random book with and
   without the stop, and measure its average net exposure.

2. IS THE EDGE OVER RANDOM SIGNIFICANT, PERIOD BY PERIOD?
   oos_book's z compares a sleeve's Sharpe to the spread of random Sharpes.
   The cleaner question is whether the DAILY EXCESS (real minus the mean of
   the random books) is distinguishable from zero. That excess series strips
   out whatever structural drift both share. t-stat on monthly sums.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

import oos_book as OB

SEEDS = 12


def run_with_exposure(signal_fn, P, n, hold, phases, stop, cols=None):
    """Same engine as OB.run_sleeve, but also records net exposure."""
    PX, R, FN = P["PX"], P["R"], P["FN"]
    T, C = PX.shape
    mask = np.ones(C, bool) if cols is None else cols
    net = np.zeros(T)
    acc = np.zeros(T)
    for ph in range(phases):
        w_prev = np.zeros(C)
        entry = np.full(C, np.nan)
        out = np.zeros(T)
        for t in range(OB.WARMUP, T):
            if stop is not None:
                for j in np.flatnonzero(w_prev):
                    if np.isnan(entry[j]) or np.isnan(PX[t, j]):
                        continue
                    if (PX[t, j] / entry[j] - 1) * np.sign(w_prev[j]) < -stop:
                        w_prev[j] = 0.0
                        entry[j] = np.nan
            r_t = np.nan_to_num(R[t])
            carry = np.sum(w_prev * r_t) - np.sum(w_prev * FN[t])
            if (t - ph) % hold != 0:
                out[t] = carry
                net[t] += w_prev.sum() / phases
                continue
            s = np.asarray(signal_fn(t), float).copy()
            s[~mask] = np.nan
            s[~np.isfinite(PX[t])] = np.nan
            valid = np.isfinite(s)
            if valid.sum() < 2 * n + 2:
                out[t] = carry
                net[t] += w_prev.sum() / phases
                continue
            order = np.argsort(np.where(valid, s, -1e9))
            w = np.zeros(C)
            w[order[-n:]] = 0.5 / n
            w[order[:n]] = -0.5 / n
            out[t] = (np.sum(w * r_t) - np.abs(w - w_prev).sum() * OB.FEE
                      - np.sum(w * FN[t]))
            w_prev = w
            entry = np.where(w != 0, PX[t], np.nan)
            net[t] += w.sum() / phases
        acc += out
    return acc / phases, net


def main() -> int:
    P = OB.build_panels()
    dates = P["dates"]
    per = OB.periods(dates)
    C = len(P["coins"])

    print("\n  1. Random book: does the stop create a net-long drift?\n")
    print(f"  {'stop':<8}{'period':<7}{'Sharpe':>8}{'avg net exposure':>18}")
    for stop in (None, OB.STOP_PLATEAU):
        sh = {k: [] for k in per}
        ex = {k: [] for k in per}
        for s in range(SEEDS):
            r, net = run_with_exposure(OB.sig_random(C, zlib.crc32(f"d{s}".encode())),
                                       P, 5, 7, 7, stop)
            for k, m in per.items():
                sh[k].append(OB.sharpe(r[m]))
                ex[k].append(net[m].mean())
        for k in per:
            lab = "none" if stop is None else f"{stop:.0%}"
            print(f"  {lab:<8}{k:<7}{np.mean(sh[k]):>8.2f}"
                  f"{np.mean(ex[k])*100:>16.1f}%")
        print()

    print("  2. Daily excess over the random mean, t on monthly sums\n")
    print(f"  {'sleeve':<13}{'period':<7}{'excess ann%':>12}{'t_mon':>8}"
          f"{'mon+':>8}{'excess Sharpe':>15}")
    for name, spec in OB.specs(P).items():
        sig, n, hold, phases, cols = spec
        real = OB.run_sleeve(sig, P["PX"], P["R"], P["FN"], n, hold, phases,
                             OB.STOP_PLATEAU, cols)
        ctrl = np.mean([OB.run_sleeve(
            OB.sig_random(C, zlib.crc32(f"{name}c{s}".encode())),
            P["PX"], P["R"], P["FN"], n, hold, phases, OB.STOP_PLATEAU, cols)
            for s in range(SEEDS)], axis=0)
        exc = real - ctrl
        for k, m in per.items():
            tm, nm, mp = OB.monthly_t(dates[m], exc[m])
            print(f"  {name:<13}{k:<7}{exc[m].mean()*365*100:>11.1f}%"
                  f"{tm:>8.2f}{mp:>5}/{nm:<2}{OB.sharpe(exc[m]):>14.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
