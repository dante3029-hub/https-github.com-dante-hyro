#!/usr/bin/env python3
"""
hist_test.py -- the crypto candidates on data NOTHING in this project has seen:
Binance USDT-M futures, Sep 2019 -> Jan 2023 (includes the 2021 bull run, the
2022 crash, LUNA and FTX). Dead coins are included. One run, rules fixed below.

  python3 hist_test.py            # downloads ~3,000 small files to ~/binance_hist, then tests
  python3 hist_test.py --selftest # offline check of the engine on fake data

DATA   data.binance.vision (free, public): monthly daily-kline and funding-rate
       archives per symbol. Delisted symbols stay in the archive, which is the
       point: LUNA, FTT, SRM etc. go to zero inside the sample.
COSTS  0.085% per side on turnover (taker + slippage, as every other test);
       funding charged to longs / paid to shorts every day.
SPLIT  H1 = Sep 2019 -> Jun 2021, H2 = Jul 2021 -> Jan 2023 (reported apart).

STRATEGIES (identical rules to published.py / bnf_dip.py; nothing re-chosen)
 1 XS MOMENTUM   past 21-day return to t-1; long top 6 / short bottom 6,
                 equal weight, gross 1; hold 7 days, 7 staggered sub-books
 2 TREND         per coin +1 if SMA10 > SMA40 at t-1 else -1 (and long-or-flat),
                 equal volatility (20-day), gross 1
 3 FUNDING CARRY minus trailing 7-day mean funding; long cheapest 5 / short
                 richest 5; hold 7, 7 sub-books
 4 SUPERTREND    ATR 10, multiplier 3 (TradingView default) on daily bars, the
                 same setting for every coin; per coin +1 / -1 by direction at
                 t-1, equal volatility, gross 1 (the flip is the trailing stop)
 5 BNF CRYPTO    close >= 20% below SMA25 -> long next open, exit at the first
                 close back at the SMA or day 10; % per trade vs random entries
                 on the same coin with the same holding length
CONTROLS 1, 3: random scores (40 seeds). 2, 4: each coin's signal circularly
         shifted (40 seeds, same persistence). 5: random entry days (40 seeds).
VERDICT  excess over control > 0 in BOTH halves and monthly t of the excess > 2.
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from datetime import date

import numpy as np
import pandas as pd

log = logging.getLogger("hist")

BASE = "https://data.binance.vision/data/futures/um/monthly"
CACHE = os.path.expanduser("~/binance_hist")
START, END = date(2019, 9, 1), date(2023, 1, 31)
SPLIT = pd.Timestamp("2021-07-01")
FEE = 0.00085
SEEDS = 40
WARMUP = 45

# Liquid USDT-M perps of 2019-2022, chosen by what traded THEN, dead ones included.
SYMBOLS = """BTC ETH BNB XRP ADA DOGE SOL DOT AVAX MATIC LINK LTC BCH TRX ETC XLM ATOM
FIL UNI AAVE SUSHI EOS XTZ THETA VET ALGO NEAR FTM SAND MANA AXS LUNA FTT SRM CHZ ENJ
ZEC DASH XMR COMP SNX 1INCH CRV GRT ICP EGLD KSM WAVES ONE ZIL NEO IOTA QTUM ONT BAT
OMG RUNE KAVA ALPHA""".split()


# ------------------------------------------------------------------ download
def _months():
    y, m = START.year, START.month
    while (y, m) <= (END.year, END.month):
        yield f"{y}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _get(url, dest, tries=4):
    """Fetch url to dest. Returns True if the file exists afterwards. A 404 is
    normal (symbol not listed that month) and is cached as an empty marker."""
    if os.path.exists(dest) or os.path.exists(dest + ".404"):
        return os.path.exists(dest)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(dest + ".404", "w").close()
                return False
            log.warning("HTTP %s on %s (try %d)", e.code, url, k + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log.warning("network error on %s: %s (try %d)", url, e, k + 1)
        time.sleep(2 * (k + 1))
    log.error("giving up on %s", url)
    return False


def _read_zip_csv(path, names):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if not str(df.iloc[0, 0]).replace(".", "").isdigit():   # some files have a header row
        df = df.iloc[1:]
    df = df.iloc[:, :len(names)]
    df.columns = names
    return df


def download():
    os.makedirs(CACHE, exist_ok=True)
    px, fn = {}, {}
    for s in SYMBOLS:
        sym = f"{s}USDT"
        kl, fr = [], []
        for ym in _months():
            kp = os.path.join(CACHE, f"{sym}-1d-{ym}.zip")
            if _get(f"{BASE}/klines/{sym}/1d/{sym}-1d-{ym}.zip", kp):
                try:
                    kl.append(_read_zip_csv(kp, ["t", "open", "high", "low", "close", "volume", "ct", "qv"]))
                except (zipfile.BadZipFile, ValueError, IndexError) as e:
                    log.warning("bad kline file %s: %s", kp, e)
            fp = os.path.join(CACHE, f"{sym}-fundingRate-{ym}.zip")
            if _get(f"{BASE}/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip", fp):
                try:
                    fr.append(_read_zip_csv(fp, ["t", "interval", "rate"]))
                except (zipfile.BadZipFile, ValueError, IndexError) as e:
                    log.warning("bad funding file %s: %s", fp, e)
        if not kl:
            log.info("%s: no data in window", sym)
            continue
        k = pd.concat(kl)
        k["t"] = pd.to_datetime(pd.to_numeric(k["t"]), unit="ms").dt.floor("D")
        k = k.drop_duplicates("t").set_index("t").sort_index()
        px[s] = k[["open", "high", "low", "close", "qv"]].apply(pd.to_numeric, errors="coerce")
        if fr:
            f = pd.concat(fr)
            f["t"] = pd.to_datetime(pd.to_numeric(f["t"]), unit="ms").dt.floor("D")
            fn[s] = pd.to_numeric(f["rate"], errors="coerce").groupby(f["t"]).sum()
        print(f"  {sym:<12} {len(k):>5} days" + ("" if fr else "  (no funding)"), flush=True)
    return px, fn


# ------------------------------------------------------------------ panels
def panels(px, fn):
    idx = pd.date_range(min(d.index.min() for d in px.values()), max(d.index.max() for d in px.values()), freq="D")
    coins = sorted(px)
    def M(field):
        return pd.DataFrame({c: px[c][field].reindex(idx) for c in coins}).to_numpy(float, copy=True)
    O, H, L, PX, QV = M("open"), M("high"), M("low"), M("close"), M("qv")
    bad = ~(np.isfinite(PX) & (PX > 0))
    PX[bad] = np.nan
    R = np.full_like(PX, np.nan)
    R[1:] = PX[1:] / PX[:-1] - 1
    R[np.abs(np.nan_to_num(R)) > 3] = np.nan            # data glitches only
    FN = pd.DataFrame({c: fn.get(c, pd.Series(dtype=float)).reindex(idx) for c in coins}).to_numpy(float, copy=True)
    has_fn = np.isfinite(FN).sum(axis=0) > 100
    return dict(dates=idx, coins=coins, O=O, H=H, L=L, PX=PX, R=R, FN=np.nan_to_num(FN), QV=QV, has_fn=has_fn)


# ------------------------------------------------------------------ engines
def run_sleeve(signal_fn, PX, R, FN, n, hold, phases, cols=None):
    """Same mechanics as oos_book.run_sleeve (no stop): weights chosen at t
    from data to t-1, applied to R[t]; fees on turnover; funding daily."""
    T, C = PX.shape
    mask = np.ones(C, bool) if cols is None else cols
    acc = np.zeros(T)
    for ph in range(phases):
        w_prev = np.zeros(C)
        out = np.zeros(T)
        for t in range(WARMUP, T):
            r_t = np.nan_to_num(R[t])
            carry = np.sum(w_prev * r_t) - np.sum(w_prev * FN[t])
            if (t - ph) % hold:
                out[t] = carry
                continue
            s = np.asarray(signal_fn(t), float).copy()
            s[~mask] = np.nan
            s[~np.isfinite(PX[t])] = np.nan
            valid = np.isfinite(s)
            if valid.sum() < 2 * n + 2:
                out[t] = carry
                continue
            order = np.argsort(np.where(valid, s, -1e9))
            w = np.zeros(C)
            w[order[-n:]] = 0.5 / n
            w[order[:n]] = -0.5 / n
            out[t] = np.sum(w * r_t) - np.abs(w - w_prev).sum() * FEE - np.sum(w * FN[t])
            w_prev = w
        acc += out
    return acc / phases


def vol_book(sig, PX, R, FN):
    """Per-coin direction from sig[t-1], equal volatility, gross 1, daily."""
    T, C = PX.shape
    vol = pd.DataFrame(R).rolling(20).std().to_numpy()
    out, w_prev = np.zeros(T), np.zeros(C)
    for t in range(WARMUP, T):
        s, v = sig[t - 1], vol[t - 1]
        ok = np.isfinite(s) & np.isfinite(v) & (v > 0) & np.isfinite(PX[t])
        w = np.zeros(C)
        if ok.sum() >= 5:
            raw = np.where(ok, s / np.where(ok, v, 1), 0.0)
            g = np.abs(raw).sum()
            w = raw / g if g > 0 else w
        out[t] = np.sum(w * np.nan_to_num(R[t])) - np.abs(w - w_prev).sum() * FEE - np.sum(w * FN[t])
        w_prev = w
    return out


def shifted(sig, seed):
    rg = np.random.default_rng(seed)
    s = sig.copy()
    for j in range(s.shape[1]):
        ok = np.flatnonzero(np.isfinite(s[:, j]))
        if len(ok) > 60:
            s[ok, j] = np.roll(s[ok, j], int(rg.integers(20, len(ok) - 20)))
    return s


def supertrend_dir(H, L, C, n=10, mult=3.0):
    """TradingView ta.supertrend(3, 10) direction per bar: +1 up, -1 down."""
    T = len(C)
    d = np.full(T, np.nan)
    pc = np.r_[np.nan, C[:-1]]
    tr = np.nanmax(np.vstack([H - L, np.abs(H - pc), np.abs(L - pc)]), axis=0)
    atr = np.full(T, np.nan)
    first = np.flatnonzero(np.isfinite(tr) & np.isfinite(C))
    if len(first) < n + 2:
        return d
    s = first[0]
    a = np.nan
    up_prev = dn_prev = np.nan
    dir_prev = 1.0
    cnt = 0
    for i in range(s, T):
        if not (np.isfinite(H[i]) and np.isfinite(L[i]) and np.isfinite(C[i])):
            continue
        cnt += 1
        a = tr[i] if not np.isfinite(a) else (a * (n - 1) + tr[i]) / n
        if cnt < n:
            continue
        hl2 = (H[i] + L[i]) / 2
        up, dn = hl2 - mult * a, hl2 + mult * a
        if np.isfinite(up_prev) and np.isfinite(pc[i]):
            up = max(up, up_prev) if pc[i] > up_prev else up
            dn = min(dn, dn_prev) if pc[i] < dn_prev else dn
            if dir_prev < 0 and C[i] > dn_prev:
                dir_prev = 1.0
            elif dir_prev > 0 and C[i] < up_prev:
                dir_prev = -1.0
        up_prev, dn_prev = up, dn
        d[i] = dir_prev
    return d


def bnf_trades(o, c, thresh=0.20, sma_n=25, max_hold=10, cost=0.0017):
    sma = pd.Series(c).rolling(sma_n).mean().to_numpy()
    out, t, n = [], sma_n, len(c)
    while t < n - 1:
        if np.isfinite(sma[t]) and np.isfinite(c[t]) and c[t] <= sma[t] * (1 - thresh) and np.isfinite(o[t + 1]):
            eb, ex = t + 1, min(t + max_hold, n - 1)
            for j in range(eb, ex + 1):
                if np.isfinite(c[j]) and c[j] >= sma[j]:
                    ex = j
                    break
            while ex > eb and not np.isfinite(c[ex]):       # delisted mid-trade: last real close
                ex -= 1
            out.append((eb, ex, c[ex] / o[eb] - 1 - cost))
            t = ex + 1
            continue
        t += 1
    return out


# ------------------------------------------------------------------ reporting
def verdict(name, dates, real, ctrl):
    m0 = dates >= dates[WARMUP]
    exc = real - ctrl
    def sh(x):
        return x.mean() / x.std() * math.sqrt(365) if x.std() > 0 else 0.0
    s = pd.Series(exc[m0], index=dates[m0]).resample("ME").sum()
    s = s[s != 0]
    t = s.mean() / (s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 2 else 0.0
    h1 = m0 & (dates < SPLIT)
    h2 = m0 & (dates >= SPLIT)
    ok = sh(exc[h1]) > 0 and sh(exc[h2]) > 0 and t > 2
    print(f"  {name}")
    print(f"    strategy Sharpe {sh(real[m0]):5.2f} ({real[m0].mean()*365*100:+.0f}%/yr)   control {sh(ctrl[m0]):5.2f}"
          f"   excess: H1 {sh(exc[h1]):5.2f}  H2 {sh(exc[h2]):5.2f}   monthly t {t:+.2f} ({int((s>0).sum())}/{len(s)})"
          f"   verdict: {'PASS' if ok else 'REJECT'}")


def run_all(P, rg):
    PX, R, FN, dates = P["PX"], P["R"], P["FN"], P["dates"]
    T, C = PX.shape
    print(f"\n  {C} coins, {dates[0].date()} -> {dates[-1].date()}   (H1 < {SPLIT.date()} <= H2)\n")
    lp = np.log(PX)
    mom = lambda t: lp[t - 1] - lp[t - 22] if t >= 23 else np.full(C, np.nan)
    real = run_sleeve(mom, PX, R, FN, 6, 7, 7)
    ctrl = np.mean([run_sleeve(lambda t, g=np.random.default_rng(zlib.crc32(f"m{s}".encode())): g.random(C),
                               PX, R, FN, 6, 7, 7) for s in range(SEEDS)], axis=0)
    verdict("1  XS MOMENTUM 21d, 6/6, weekly", dates, real, ctrl)

    px = pd.DataFrame(PX)
    s10, s40 = px.rolling(10).mean().to_numpy(), px.rolling(40).mean().to_numpy()
    tsig = np.where(np.isfinite(s10) & np.isfinite(s40), np.where(s10 > s40, 1.0, -1.0), np.nan)
    for lab, sg in (("long/short", tsig), ("long or flat", np.where(tsig > 0, 1.0, np.where(np.isfinite(tsig), 0.0, np.nan)))):
        real = vol_book(sg, PX, R, FN)
        ctrl = np.mean([vol_book(shifted(sg, zlib.crc32(f"t{lab}{s}".encode())), PX, R, FN) for s in range(SEEDS)], axis=0)
        verdict(f"2  TREND SMA10/40, {lab}", dates, real, ctrl)

    fcols = np.asarray(P["has_fn"], bool)
    f7 = pd.DataFrame(np.where(FN == 0, np.nan, FN)).rolling(7, min_periods=5).mean().to_numpy()
    carry = lambda t: -f7[t - 1] if t >= 8 else np.full(C, np.nan)
    real = run_sleeve(carry, PX, R, FN, 5, 7, 7, fcols)
    ctrl = np.mean([run_sleeve(lambda t, g=np.random.default_rng(zlib.crc32(f"c{s}".encode())): g.random(C),
                               PX, R, FN, 5, 7, 7, fcols) for s in range(SEEDS)], axis=0)
    verdict(f"3  FUNDING CARRY 5/5, weekly ({int(fcols.sum())} coins with funding)", dates, real, ctrl)

    st = np.column_stack([supertrend_dir(P["H"][:, j], P["L"][:, j], PX[:, j]) for j in range(C)])
    real = vol_book(st, PX, R, FN)
    ctrl = np.mean([vol_book(shifted(st, zlib.crc32(f"st{s}".encode())), PX, R, FN) for s in range(SEEDS)], axis=0)
    verdict("4  SUPERTREND 10/3, every coin, long/short", dates, real, ctrl)

    rows = []
    for j in range(C):
        for eb, ex, r in bnf_trades(P["O"][:, j], PX[:, j]):
            rows.append((dates[eb], j, eb, ex, r))
    if len(rows) < 30:
        print(f"\n  5  BNF CRYPTO: only {len(rows)} trades")
        return
    df = pd.DataFrame(rows, columns=["ts", "j", "eb", "ex", "r"])
    ctrl = []
    for _ in range(SEEDS):
        acc = []
        for j, eb, ex in zip(df.j, df.eb, df.ex):
            ok = np.flatnonzero(np.isfinite(P["O"][:, j]) & np.isfinite(PX[:, j]))
            ok = ok[ok < T - (ex - eb) - 1]
            if len(ok) < 10:
                continue
            s = int(rg.choice(ok))
            e = s + (ex - eb)
            if np.isfinite(PX[e, j]):
                acc.append(PX[e, j] / P["O"][s, j] - 1 - 0.0017)
        ctrl.append(np.mean(acc))
    cm = float(np.mean(ctrl))
    mon = df.groupby(df.ts.dt.to_period("M")).r.mean()
    t = mon.mean() / (mon.std(ddof=1) / math.sqrt(len(mon)))
    h1, h2 = df.r[df.ts < SPLIT].mean(), df.r[df.ts >= SPLIT].mean()
    ok = df.r.mean() > cm and h1 > 0 and h2 > 0 and t > 2
    print(f"\n  5  BNF CRYPTO 20% below SMA25 (dead coins included)")
    print(f"    {len(df)} trades  win {(df.r>0).mean()*100:.0f}%  avg {df.r.mean()*100:+.2f}%  random {cm*100:+.2f}%"
          f"  worst {df.r.min()*100:+.0f}%   halves {h1*100:+.2f}% / {h2*100:+.2f}%   monthly t {t:+.2f}"
          f"   verdict: {'PASS' if ok else 'REJECT'}")


def selftest():
    """Engine check on fake random-walk coins: every strategy must show ~0 edge."""
    rg = np.random.default_rng(1)
    idx = pd.date_range("2019-09-01", "2023-01-31", freq="D")
    px, fn = {}, {}
    for k in range(40):
        c = 100 * np.exp(np.cumsum(rg.normal(0, 0.04, len(idx))))
        o = c * np.exp(rg.normal(0, 0.01, len(idx)))
        h, l = np.maximum(o, c) * 1.02, np.minimum(o, c) * 0.98
        px[f"C{k}"] = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "qv": 1e6}, index=idx)
        fn[f"C{k}"] = pd.Series(rg.normal(0.0001, 0.0002, len(idx)), index=idx)
    global SEEDS
    SEEDS = 5
    print("  SELFTEST on random walks -- expect no PASS anywhere")
    run_all(panels(px, fn), rg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if a.selftest:
        selftest()
        return 0
    print("\n  Downloading Binance futures history (cached in ~/binance_hist; rerun resumes)\n")
    px, fn = download()
    if len(px) < 15:
        print(f"  only {len(px)} symbols downloaded -- check the network and rerun")
        return 1
    run_all(panels(px, fn), np.random.default_rng(zlib.crc32(b"hist_test")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
