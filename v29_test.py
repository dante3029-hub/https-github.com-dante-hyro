#!/usr/bin/env python3
"""
v29_test.py -- did V29 have an edge, or did it harvest a bull market?

WHY THIS EXISTS. V29 passed two prop evaluations and made ~$20k twice. It was
switched off on 2026-08-19 with the cron comment "(no edge)" -- and that
verdict is not written down anywhere in this repo. No numbers, no method, no
analysis. The single most consequential decision in this project rests on an
assertion nobody can check, while the nine sleeves that replaced it were held
to leave-one-out tests, lookahead poisoning, and 2000-draw nulls.

This applies the same standard to V29.

THE RULES, read out of existing_botcode/botcode/{config,strategy}.py:
  entry   Supertrend direction, per-coin (period, multiplier) from
          COIN_CONFIGS; direction held for EXACTLY 5-6 bars; EMA(8) above
          EMA(21) for longs and below for shorts; ADX(14) > 18. 1h bars.
  exit    stop at 1.5 ATR; TP1 at 1.5 ATR taking 75% off; the runner trails
          the Supertrend line.
  book    max 2 concurrent positions, 5-bar cooldown per symbol after an exit.
  LONG AND SHORT -- it is not a long-only beta harvester.

THE TWO QUESTIONS, and they are different:

  1. Does it beat BUY-AND-HOLD on the same coins over the same window?
     Crypto rose over most of this sample. A directional strategy will show a
     profit and that profit is not evidence of skill. Only the gap over
     holding is.

  2. Does it beat RANDOM TIMING? Same coins, same trade count, same stop, same
     TP, same trailing exit -- entries at random bars. This isolates the entry
     RULE from the exposure, exactly as event_audit does for fvg and pattern.

A strategy can fail (1) and still pass an evaluation, because an evaluation
only asks for 10% without a 5% daily or 10% total drawdown. Beta delivers that
in a bull market. It stops delivering it in the first sustained downtrend,
which is the risk that matters for a FUNDED account rather than an eval.
So this reports both, and the drawdown profile alongside.

DECLARED SIMPLIFICATIONS -- stated rather than buried, because an unflattering
backtest of someone's working bot has to be auditable:
  - TP2 (7 ATR, 50% of the remainder) is omitted; the runner trails the
    Supertrend line, which is what actually closed most trades.
  - Position sizing is equal-risk per trade rather than V29's
    min(equity*r%, balance*3%, $1700) ladder. This measures the SIGNAL, not
    the money management. Sizing is tested separately by size_from_dd.py.
  - Fills at the next bar's open, fees 8.5bp per side, stops filled at the
    stop price (optimistic on a gap -- flagged, not corrected).

USAGE
    python3 -u v29_test.py
    python3 -u v29_test.py --sims 500
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from fetch_spot import resolve_dir

PERP, _TRIED = resolve_dir("HYRO_TAKER_DATA_DIR", "taker_data", True)

COIN_CONFIGS = {       # (supertrend period, multiplier) -- V29's own table
    "ORDI": (28, 1.5), "ONDO": (14, 1.5), "ICP": (14, 1.5),
    "APT": (10, 2.0),  "AVAX": (28, 1.5), "LINK": (14, 1.5),
    "FIL": (21, 3.5),  "ARB": (21, 2.5),  "WLD": (14, 1.5),
}
EMA_F, EMA_S, ADX_P, ADX_T = 8, 21, 14, 18.0
CONFIRM, COOLDOWN = 5, 5
SL_ATR, TP1_ATR, TP1_FRAC = 1.5, 1.5, 0.75
FEE = 0.00085
TF = 1          # bar size in hours; --tf N resamples. The fee is the SAME
                # per round trip whatever the timeframe, so trading tiny
                # 1h wiggles at a fixed cost is structurally losing. 700
                # signals per coin over 3.7 years is a trade every 1.9 days
                # -- trends do not start every two days.


def log(m=""):
    print(m, flush=True)


_CACHE = {}


def _resample(df, tf):
    if tf <= 1:
        return df
    return df.resample(f"{tf}h").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last"}).dropna()


def _load(path):
    """Cached. The random-timing null runs hundreds of draws over the same
    coins; re-reading a 31k-row CSV each time turned a 5-minute test into
    an hour."""
    if path in _CACHE:
        return _CACHE[path]
    try:
        df = pd.read_csv(path, usecols=["open_time", "open", "high",
                                        "low", "close"])
    except Exception:
        _CACHE[path] = None
        return None
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("ts").sort_index()
    for cc in ("open", "high", "low", "close"):
        df[cc] = pd.to_numeric(df[cc], errors="coerce")
    df = df.dropna()
    _CACHE[path] = df
    return df


def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.full(len(x), np.nan)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = (x[i] - out[i - 1]) * a + out[i - 1]
    return out


def atr(h, l, c, n=14):
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.full(len(c), np.nan)
    if len(c) > n:
        out[n] = tr[1:n + 1].mean()
        for i in range(n + 1, len(c)):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def adx(h, l, c, n=14):
    """Wilder ADX. Returns an array aligned to the bars."""
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    out = np.full(len(c), np.nan)
    if len(c) < 2 * n + 2:
        return out
    atr_s = tr[:n].sum(); p_s = pdm[:n].sum(); n_s = ndm[:n].sum()
    dx_hist = []
    for i in range(n, len(tr)):
        atr_s = atr_s - atr_s / n + tr[i]
        p_s = p_s - p_s / n + pdm[i]
        n_s = n_s - n_s / n + ndm[i]
        if atr_s <= 0:
            continue
        pdi = 100 * p_s / atr_s; ndi = 100 * n_s / atr_s
        s = pdi + ndi
        dx = 100 * abs(pdi - ndi) / s if s > 0 else 0.0
        dx_hist.append(dx)
        if len(dx_hist) == n:
            out[i + 1] = float(np.mean(dx_hist))
        elif len(dx_hist) > n:
            out[i + 1] = (out[i] * (n - 1) + dx) / n
    return out


def supertrend(h, l, c, period, mult):
    a = atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    fu = np.full(len(c), np.nan); fl = np.full(len(c), np.nan)
    tr = np.zeros(len(c), dtype=int)
    start = period + 1
    if len(c) <= start:
        return tr, fu, fl
    fu[start] = ub[start]; fl[start] = lb[start]; tr[start] = 1
    for i in range(start + 1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if tr[i - 1] == 1:
            tr[i] = -1 if c[i] < fl[i] else 1
        else:
            tr[i] = 1 if c[i] > fu[i] else -1
    return tr, fu, fl


def btc_macd_dir(bars_index):
    """BTC MACD(12,26) zero-line direction, aligned to an index.

    From main.py's BTC-REGIME block: trades whose side FIGHTS the BTC MACD
    sign are cut to QUARTER size (risk x 0.25), not filtered out. This is a
    chop defence, and chop is what kills trend-followers -- omitting it
    measures a weaker V29 than the one that passed two evaluations.

    main.py reads _macd[-2], the last CLOSED bar, so this shifts by one to
    match: the direction known AT bar i is the MACD of bar i-1.
    """
    f = f"{PERP}/BTC_1h.csv"
    if not os.path.exists(f):
        return None
    df = _load(f)
    if df is None:
        return None
    if TF > 1:
        df = _resample(df, TF)
    c = df["close"].values
    def _e(v, p):
        k = 2.0 / (p + 1.0)
        out = np.empty(len(v)); e = v[0]
        for i, x in enumerate(v):
            e = x * k + e * (1 - k)
            out[i] = e
        return out
    macd = _e(c, 12) - _e(c, 26)
    s = pd.Series(np.sign(macd), index=df.index).shift(1)   # last CLOSED bar
    return s.reindex(bars_index, method="ffill")


def run_coin(coin, df, entries_override=None, btc=None):
    """Returns list of trade returns. entries_override replaces the signal
    with a given list of entry bars -- the random-timing control uses the
    SAME exit machinery, which is the whole point."""
    p, m = COIN_CONFIGS[coin]
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    tr, fu, fl = supertrend(h, l, c, p, m)
    a = atr(h, l, c, 14)
    ef, es = ema(c, EMA_F), ema(c, EMA_S)
    ax = adx(h, l, c, ADX_P)
    n = len(c)

    sigs = []
    if entries_override is None:
        hold = 0
        for i in range(1, n):
            hold = hold + 1 if tr[i] == tr[i - 1] else 1
            if not (CONFIRM <= hold <= CONFIRM + 1):
                continue
            if not np.isfinite(a[i]) or a[i] <= 0:
                continue
            if not (np.isfinite(ef[i]) and np.isfinite(es[i])):
                continue
            side = 1 if tr[i] == 1 else -1
            if side == 1 and ef[i] <= es[i]:
                continue
            if side == -1 and ef[i] >= es[i]:
                continue
            if np.isfinite(ax[i]) and ax[i] <= ADX_T:
                continue
            sigs.append((i + 1, side))
    else:
        sigs = entries_override

    out, last_exit = [], -10**9
    for eb, side in sigs:
        if eb >= n - 1 or eb - last_exit < COOLDOWN:
            continue
        aa = a[eb - 1]
        if not np.isfinite(aa) or aa <= 0:
            continue
        ent = o[eb]
        if not np.isfinite(ent) or ent <= 0:
            continue
        stop = ent - side * SL_ATR * aa
        tp1 = ent + side * TP1_ATR * aa
        got1 = False; realised = 0.0; ex = eb
        for j in range(eb, n):
            ex = j
            if side == 1:
                if l[j] <= stop:
                    realised += (1 - got1 * TP1_FRAC) * (stop / ent - 1); break
                if (not got1) and h[j] >= tp1:
                    realised += TP1_FRAC * (tp1 / ent - 1); got1 = True
                    stop = max(stop, ent)            # runner to breakeven
                # Supertrend trail is the EXIT and is live from entry --
                # config has use_supertrend_trailing=True unconditionally.
                # Gating it on got1 meant a trade that never reached TP1 and
                # never hit its stop ran to the END OF THE DATA. Rare at 1h,
                # common at 20-24h where the ATR stop is very wide in percent
                # terms, which is almost certainly the long-end collapse.
                if np.isfinite(fl[j]) and c[j] < fl[j]:
                    realised += (1 - got1 * TP1_FRAC) * (c[j] / ent - 1); break
            else:
                if h[j] >= stop:
                    realised += (1 - got1 * TP1_FRAC) * (1 - stop / ent); break
                if (not got1) and l[j] <= tp1:
                    realised += TP1_FRAC * (1 - tp1 / ent); got1 = True
                    stop = min(stop, ent)
                if np.isfinite(fu[j]) and c[j] > fu[j]:
                    realised += (1 - got1 * TP1_FRAC) * (1 - c[j] / ent); break
        else:
            realised += (1 - got1 * TP1_FRAC) * (
                (c[n - 1] / ent - 1) * side)
        w = 1.0
        if btc is not None:
            d = btc.iloc[eb] if eb < len(btc) else np.nan
            if np.isfinite(d) and d != 0 and int(d) == -side:
                w = 0.25          # FIGHTS BTC MACD -> quarter size
        out.append((realised - 2 * FEE, w, eb))
        last_exit = ex
    return out


def main() -> int:
    if PERP is None:
        log("  cannot find taker_data; tried: " + ", ".join(_TRIED))
        return 2
    global TF
    if "--tf" in sys.argv:
        TF = int(sys.argv[sys.argv.index("--tf") + 1])
    sims = 300
    if "--sims" in sys.argv:
        sims = int(sys.argv[sys.argv.index("--sims") + 1])

    # --all-coins: every coin in taker_data, using V29's own default
    # (21, 2.0) for anything outside COIN_CONFIGS -- which is exactly what
    # strategy.get_coin_config does live. The point is the DISTRIBUTION:
    # if V29 only works on a lucky subset, the lineup was the edge, not the
    # strategy, and the owner's own notes already flag coin ordering moving
    # returns 15-23%.
    if "--all-coins" in sys.argv:
        import glob as _g
        for _f in sorted(_g.glob(f"{PERP}/*_1h.csv")):
            _c = os.path.basename(_f).replace("_1h.csv", "")
            if _c != "BTC":
                COIN_CONFIGS.setdefault(_c, (21, 2.0))

    rows, bh_rows, all_real = [], [], []
    per_coin, frames, btc_cache = {}, {}, {}
    use_btc = "--no-btc-regime" not in sys.argv
    log(f"  V29: Supertrend + EMA(8/21) + ADX>{ADX_T:.0f}, {TF}h bars, "
        f"long AND short")
    if use_btc:
        if os.path.exists(f"{PERP}/BTC_1h.csv"):
            log("  BTC MACD regime sizing: ON (trades fighting BTC -> 1/4 size)")
        else:
            log("  BTC MACD regime: NO BTC_1h.csv -- running WITHOUT the chop")
            log("  filter that was added to fix exactly this strategy's")
            log("  weakness. Results will UNDERSTATE V29. Fetch BTC first.")
    else:
        log("  BTC MACD regime sizing: OFF (--no-btc-regime)")
    log(f"  {sims} random-timing draws\n")
    log(f"  {'coin':<7}{'trades':>8}{'mean/trade':>12}{'total':>10}"
        f"{'buy&hold':>11}{'beats B&H':>11}{'1/4 sz':>9}")
    for coin, (p, m) in COIN_CONFIGS.items():
        f = f"{PERP}/{coin}_1h.csv"
        if not os.path.exists(f):
            log(f"  {coin:<7} no data -- SKIPPED")
            continue
        df = _load(f)
        if df is not None and TF > 1:
            df = _resample(df, TF)
        if df is None or len(df) < (2000 // max(TF, 1)):
            log(f"  {coin:<7} only {len(df)} bars -- SKIPPED")
            continue
        btc = btc_macd_dir(df.index) if use_btc else None
        rr = run_coin(coin, df, btc=btc)
        if not rr:
            log(f"  {coin:<7} no trades -- SKIPPED")
            continue
        r_ = np.array([x[0] for x in rr]); w_ = np.array([x[1] for x in rr])
        tot = float(np.sum(r_ * w_))           # weighted: quarter-size trades
        wmean = float(np.sum(r_ * w_) / np.sum(w_))   # per unit of risk
        bh = float(df["close"].values[-1] / df["open"].values[0] - 1)
        per_coin[coin] = rr
        frames[coin] = df
        btc_cache[coin] = btc
        all_real += rr
        n_cut = int((w_ < 1.0).sum())
        rows.append((coin, len(rr), wmean, tot, bh, n_cut))
        bh_rows.append(bh)
        log(f"  {coin:<7}{len(rr):>8}{wmean*100:>11.3f}%{tot*100:>9.1f}%"
            f"{bh*100:>10.1f}%{('YES' if tot > bh else 'no'):>11}"
            f"{n_cut:>9}")

    if not rows:
        log("  no coin produced trades")
        return 2

    log("")
    tot_v29 = float(np.sum([r[3] for r in rows]))
    tot_bh = float(np.sum(bh_rows))
    wins = sum(1 for r in rows if r[3] > r[4])
    log("  " + "=" * 64)
    log(f"  V29 total (sum of per-coin)      {tot_v29*100:>8.1f}%")
    log(f"  buy & hold same coins            {tot_bh*100:>8.1f}%")
    log(f"  beats buy & hold on              {wins}/{len(rows)} coins")
    log("  " + "=" * 64)

    # ---- random-timing null: same trade count, same exits ----------------
    # ---- how lucky was the lineup? ---------------------------------------
    if len(rows) >= 14:
        import zlib as _z
        _rng = np.random.default_rng(_z.crc32(b"lineup"))
        tots = np.array([r[3] for r in rows])
        bhs = np.array([r[4] for r in rows])
        k = min(14, len(tots))
        draws = np.array([tots[_rng.choice(len(tots), k, replace=False)].mean()
                          for _ in range(20000)])
        bdraws = np.array([bhs[_rng.choice(len(bhs), k, replace=False)].mean()
                           for _ in range(20000)])
        log("")
        log("  " + "=" * 64)
        log(f"  WAS THE LINEUP LUCK? 20,000 random {k}-coin subsets")
        log("  " + "=" * 64)
        log(f"    coins tested                 {len(tots)}")
        log(f"    positive on their own        "
            f"{int((tots > 0).sum())}/{len(tots)}")
        log(f"    mean {k}-coin lineup          {draws.mean()*100:>+7.1f}%")
        log(f"    best  / worst lineup         {draws.max()*100:>+7.1f}% / "
            f"{draws.min()*100:+.1f}%")
        log(f"    P(a random lineup profits)   "
            f"{(draws > 0).mean()*100:>6.1f}%")
        log(f"    P(it beats buy-and-hold)     "
            f"{(draws > bdraws).mean()*100:>6.1f}%")
        if (draws > 0).mean() < 0.25:
            log(f"    -> fewer than 1 in 4 lineups make money. A lineup that")
            log(f"       DID is a draw from this distribution, not evidence")
            log(f"       the strategy works.")
        elif (draws > 0).mean() > 0.75:
            log(f"    -> most lineups make money. The result does NOT depend")
            log(f"       on picking the right coins.")

    # The null is O(sims x coins x trades x hold). At 63 coins and 300 draws
    # that is ~19,000 full backtests in pure Python and it does not finish.
    # It is also not the question --all-coins is asking, so it is skipped
    # there by default rather than left to hang.
    if "--no-null" in sys.argv or ("--all-coins" in sys.argv
                                   and "--null" not in sys.argv):
        log("")
        log("  RANDOM-TIMING CONTROL: skipped.")
        log("  It is O(sims x coins x trades) and does not finish at 63 coins.")
        log("  The lineup test above is what --all-coins is for. To run the")
        log("  null, use the 9-coin default (no --all-coins), or force it")
        log("  with --null and a small --sims.")
        return 0

    log("")
    log("  RANDOM-TIMING CONTROL -- same coins, same trade count, same stop,")
    log("  same TP, same trailing exit. Only the entry bar is random.")
    import zlib
    rng = np.random.default_rng(zlib.crc32(b"v29"))
    _r = np.array([x[0] for x in all_real]); _w = np.array([x[1] for x in all_real])
    real = float(np.sum(_r * _w) / np.sum(_w))
    nulls = []
    for s in range(sims):
        acc = []
        for coin, rr in per_coin.items():
            df = frames[coin]
            n = len(df)
            picks = [(int(x), int(sd)) for x, sd in
                     zip(rng.integers(60, n - 60, size=len(rr)),
                         rng.choice([1, -1], size=len(rr)))]
            acc += run_coin(coin, df, entries_override=picks,
                            btc=btc_cache.get(coin))
        if acc:
            ar = np.array([x[0] for x in acc]); aw = np.array([x[1] for x in acc])
            nulls.append(float(np.sum(ar * aw) / np.sum(aw)))
        if s == 0 or (s + 1) % max(1, sims // 10) == 0:
            log(f"    {s+1}/{sims} draws")
    nulls = np.array(nulls)
    p_emp = (1.0 + int((nulls >= real).sum())) / (1.0 + len(nulls))
    z = (real - nulls.mean()) / nulls.std() if nulls.std() > 0 else 0.0
    log(f"    real mean per trade      {real*100:>+7.3f}%")
    log(f"    random-timing null       {nulls.mean()*100:>+7.3f}% "
        f"+- {nulls.std()*100:.3f}%   (z {z:+.2f})")
    log(f"    p(random >= real)        {p_emp:.4f}")
    log("")
    if p_emp <= 0.05 and tot_v29 > tot_bh:
        log("  VERDICT: beats BOTH random timing and buy-and-hold. The entry")
        log("  rule adds value and it is not just long exposure. The 'no edge'")
        log("  call that switched this off was WRONG.")
    elif p_emp <= 0.05:
        log("  VERDICT: beats random timing but NOT buy-and-hold. The timing")
        log("  works; you would still have done better simply holding. For an")
        log("  EVAL that can be enough. For a funded account it is fragile --")
        log("  it depends on the market continuing to rise.")
    elif tot_v29 > tot_bh:
        log("  VERDICT: beats buy-and-hold but NOT random timing. That means")
        log("  the EXITS and the risk management are doing the work, not the")
        log("  entry signal. Worth keeping, but not for the reason assumed.")
    else:
        log("  VERDICT: beats neither. The 'no edge' call looks right, and the")
        log("  two eval passes were a bull market plus position sizing.")
    log("")
    log("  SIMPLIFICATIONS: TP2 omitted (runner trails Supertrend), equal-risk")
    log("  sizing not V29's ladder, stops filled AT the stop price which")
    log("  flatters it on a gap. See the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
