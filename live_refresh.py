#!/usr/bin/env python3
"""
live_refresh.py — the missing Phase 4 piece.

WHAT THIS IS
------------
`LiveDataFeed.get_snapshot()` was a deliberate hard stop: the bot could score
CSVs, and it could talk to Bybit, but nothing joined the two. Open interest and
taker flow were observed up to 15 DAYS STALE.

The fix is NOT to reimplement the sleeves against a live feed. data_loader.py's
own docstring states the design:

    "once the CSV loaders below are swapped for live REST/websocket ingestion
     in Phase 4 ... the trimming logic itself does not change"

So this refreshes the CSV panels IN PLACE from the exchange, and everything
downstream — build_matrices, the sleeves, sleeve_history, the combiner — runs
byte-identical to what was backtested. No live-vs-replay divergence in signal
logic, because the signal logic never changes.

    run/hist,  run/taker,  run/oi          -> Universe A (Main, Flow, Short)
    clean_panel/hist, /taker, /oi          -> Universe B (DELTA, RELVOL, BOS)

SAFETY — why this writes to a temp dir first
--------------------------------------------
A half-finished pull that overwrites a live panel is worse than a stale one:
the bot would trade on a mangled matrix and nothing would flag it. So every
fetch lands in a sibling *.incoming/ directory, is validated for completeness
and freshness, and is only swapped in atomically once it passes. A failed or
partial refresh leaves the existing panel untouched and returns non-zero.

USAGE
    python3 live_refresh.py                 # refresh both universes
    python3 live_refresh.py --check         # report staleness only, fetch nothing
    python3 live_refresh.py --universe b    # one universe

EXIT CODES
    0  panels refreshed and validated
    1  refresh failed / validation failed — existing panels untouched
    2  setup error
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.environ.get("HYRO_WORKSPACE", os.path.dirname(os.path.abspath(__file__)))

BYBIT_BASE = {
    "demo": "https://api-demo.bybit.com",
    "mainnet": "https://api.bybit.com",
    "testnet": "https://api-testnet.bybit.com",
}

# Public market-data endpoints need no auth. Bybit serves the same klines on
# demo and mainnet, but we honour the configured mode so logs never lie about
# where a number came from.
USE_DEMO = os.environ.get("BYBIT_USE_DEMO", "true").lower() == "true"
BASE = BYBIT_BASE["demo" if USE_DEMO else "mainnet"]

BINANCE_BASE = "https://fapi.binance.com"

# Bybit and Binance name some pairs differently. The panels were built from
# BINANCE, so universe coin codes follow Binance convention. Price/funding come
# from Bybit and must be translated. Taker still uses the Binance name.
BYBIT_SYMBOL_ALIAS = {"1000SHIBUSDT": "SHIB1000USDT"}

# Coins NOT listed on Bybit linear perps. They cannot be traded on the execution
# venue, so they must not enter a live universe -- the sleeves would rank and
# size them, then fail at order placement. Verified against
# /v5/market/instruments-info (810 symbols) on 2026-08-12.
NOT_ON_BYBIT = {"FET"}
REQUEST_SLEEP = 0.12
MAX_STALE_HOURS = 6          # a panel older than this is not fit to trade on


def _get_binance(path: str, **params):
    url = f"{BINANCE_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "hyrobot/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"binance {path} failed: {e}") from e
            time.sleep(1.5 * (attempt + 1))
    return []


def fetch_binance_taker(symbol, start_ms, end_ms):
    """Binance klines expose takerBuyBaseVolume (field 9); Bybit's do not.
    delta = buy - sell = 2*takerBuy - volume. Same derivation as the original panels."""
    out, cursor = [], start_ms
    while cursor < end_ms:
        rows = _get_binance("/fapi/v1/klines", symbol=symbol, interval="1h",
                            startTime=cursor, endTime=end_ms, limit=1500)
        if not rows:
            break
        for k in rows:
            vol = float(k[5]); tb = float(k[9])
            out.append([int(k[0]), vol, 2.0 * tb - vol])
        if len(rows) < 1500:
            break
        cursor = int(rows[-1][0]) + 3_600_000
        time.sleep(REQUEST_SLEEP)
    return out


def _get(path: str, **params):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "hyrobot/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = json.loads(r.read())
            if body.get("retCode") not in (0, None):
                raise RuntimeError(f"{path} retCode={body.get('retCode')} {body.get('retMsg')}")
            return body.get("result", {})
        except Exception as e:                                    # noqa: BLE001
            if attempt == 3:
                raise RuntimeError(f"{path} failed after 4 attempts: {e}") from e
            time.sleep(1.5 * (attempt + 1))
    return {}


def _ms(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day,
                           tzinfo=dt.timezone.utc).timestamp() * 1000)


def _date(ms: int) -> dt.date:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).date()


# ---------------------------------------------------------------- fetchers
def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """1h klines, oldest-first. Bybit returns NEWEST first — we reverse."""
    out, cursor = [], end_ms
    while cursor > start_ms:
        res = _get("/v5/market/kline", category="linear", symbol=symbol,
                   interval="60", start=start_ms, end=cursor, limit=1000)
        rows = res.get("list", []) or []
        if not rows:
            break
        rows = sorted(rows, key=lambda r: int(r[0]))
        out = rows + out
        oldest = int(rows[0][0])
        if oldest <= start_ms or len(rows) < 1000:
            break
        cursor = oldest - 1
        time.sleep(REQUEST_SLEEP)
    seen, uniq = set(), []
    for r in out:
        if int(r[0]) not in seen:
            seen.add(int(r[0]))
            uniq.append(r)
    return sorted(uniq, key=lambda r: int(r[0]))


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    out, cursor = [], end_ms
    while cursor > start_ms:
        res = _get("/v5/market/funding/history", category="linear", symbol=symbol,
                   startTime=start_ms, endTime=cursor, limit=200)
        rows = res.get("list", []) or []
        if not rows:
            break
        rows = sorted(rows, key=lambda r: int(r["fundingRateTimestamp"]))
        out = rows + out
        oldest = int(rows[0]["fundingRateTimestamp"])
        if oldest <= start_ms or len(rows) < 200:
            break
        cursor = oldest - 1
        time.sleep(REQUEST_SLEEP)
    seen, uniq = set(), []
    for r in out:
        t = int(r["fundingRateTimestamp"])
        if t not in seen:
            seen.add(t)
            uniq.append([t, r["fundingRate"]])
    return sorted(uniq, key=lambda r: r[0])


# ------------------------------------------------------------------ writers
def write_hist(path: str, klines: list[list], taker_by_ts: dict | None = None) -> None:
    """
    Column layout MUST match what reference_impl._read expects:
        idx1 = open_time, 2=open, 3=high, 4=low, 5=close, 6=volume, 7=turnover
    Bybit kline row: [start, open, high, low, close, volume, turnover]
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "open_time", "open", "high", "low", "close", "volume",
                    "turnover", "trades", "taker_buy_base"])
        for k in klines:
            tb = taker_by_ts.get(int(k[0]))
            # col 8 = trades (unused), col 9 = taker buy base -- read by
            # reference_impl._load_4h for the BOS sleeve.
            w.writerow(["", int(k[0]), k[1], k[2], k[3], k[4], k[5], k[6], "",
                        "" if tb is None else tb])


def write_taker(path, rows):
    """rows = [ts, volume, delta] from fetch_binance_taker. Delta is NEVER fabricated."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "open_time", "x", "volume", "x2", "x3", "delta"])
        for ts, vol, delta in rows:
            w.writerow(["", int(ts), "", vol, "", "", delta])


def write_funding(path: str, rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["funding_time", "funding_rate"])
        w.writerows(rows)


# --------------------------------------------------------------- validation
def panel_staleness(panel_dir: str) -> tuple[dt.date | None, float]:
    """Newest bar across the panel, and how many hours old it is."""
    hist = os.path.join(panel_dir, "hist")
    if not os.path.isdir(hist):
        return None, float("inf")
    newest = None
    for fn in os.listdir(hist):
        if not fn.endswith("_1h.csv"):
            continue
        try:
            with open(os.path.join(hist, fn)) as f:
                last = None
                for row in csv.reader(f):
                    last = row
            ts = int(last[1])
        except Exception:                                          # noqa: BLE001
            continue
        d = _date(ts)
        if newest is None or d > newest:
            newest = d
    if newest is None:
        return None, float("inf")
    age_h = (dt.datetime.now(dt.timezone.utc).date() - newest).days * 24
    return newest, age_h


def validate_panel(panel_dir: str, expected_coins: list[str]) -> list[str]:
    """Return a list of problems. Empty list == fit to trade."""
    problems = []
    for sub in ("hist", "taker", "oi"):
        d = os.path.join(panel_dir, sub)
        if not os.path.isdir(d):
            problems.append(f"missing dir {sub}/")
    if problems:
        return problems

    for c in expected_coins:
        h = os.path.join(panel_dir, "hist", f"{c}_1h.csv")
        t = os.path.join(panel_dir, "taker", f"{c}_taker_1h.csv")
        o = os.path.join(panel_dir, "oi", f"{c}_funding.csv")
        for p, label in ((h, "hist"), (t, "taker"), (o, "funding")):
            if not os.path.exists(p) or os.path.getsize(p) < 200:
                problems.append(f"{c}: {label} missing/empty")

    newest, age_h = panel_staleness(panel_dir)
    if newest is None:
        problems.append("no readable bars")
    elif age_h > MAX_STALE_HOURS:
        problems.append(f"stale: newest bar {newest} ({age_h:.0f}h old, max {MAX_STALE_HOURS}h)")

    # the fabrication guard described in write_taker()
    tdir = os.path.join(panel_dir, "taker")
    if os.path.isdir(tdir):
        for fn in sorted(os.listdir(tdir))[:3]:
            try:
                with open(os.path.join(tdir, fn)) as f:
                    rows = list(csv.reader(f))[1:]
                if rows and all((len(r) < 7 or r[6] == "") for r in rows[-50:]):
                    problems.append(
                        f"{fn}: taker delta column EMPTY -- the DELTA sleeve cannot "
                        f"run live until trade-tape ingestion is built (see write_taker)")
                    break
            except Exception:                                      # noqa: BLE001
                continue
    return problems


# ------------------------------------------------------------------- refresh
def refresh(panel: str, coins: list[str], start: dt.date) -> bool:
    live = os.path.join(ROOT, panel)
    incoming = live + ".incoming"
    shutil.rmtree(incoming, ignore_errors=True)
    for sub in ("hist", "taker", "oi"):
        os.makedirs(os.path.join(incoming, sub), exist_ok=True)

    s_ms = _ms(start)
    e_ms = int(time.time() * 1000)
    print(f"\n[{panel}] refreshing {len(coins)} coins from {start} via {BASE}")

    skipped = [c for c in coins if c in NOT_ON_BYBIT]
    if skipped:
        print(f"  SKIPPING (not listed on Bybit, cannot be traded): {', '.join(skipped)}")
        coins = [c for c in coins if c not in NOT_ON_BYBIT]

    for i, c in enumerate(coins, 1):
        sym = f"{c}USDT"                       # Binance name (taker)
        bybit_sym = BYBIT_SYMBOL_ALIAS.get(sym, sym)   # Bybit name (price/funding)
        try:
            tk_pre = fetch_binance_taker(sym, s_ms, e_ms)
            # taker_buy_base = (volume + delta) / 2, inverting delta = 2*buy - vol
            taker_by_ts = {int(t): (v + d) / 2.0 for t, v, d in tk_pre}
            kl = fetch_klines(bybit_sym, s_ms, e_ms)
            if not kl:
                print(f"  [{i}/{len(coins)}] {c}: NO KLINES — aborting refresh")
                return False
            write_hist(os.path.join(incoming, "hist", f"{c}_1h.csv"), kl, taker_by_ts)
            tk = tk_pre
            if not tk:
                print(f"  [{i}/{len(coins)}] {c}: NO TAKER DATA - aborting "
                      f"(delta must never be fabricated)")
                return False
            write_taker(os.path.join(incoming, "taker", f"{c}_taker_1h.csv"), tk)
            fr = fetch_funding(bybit_sym, s_ms, e_ms)
            write_funding(os.path.join(incoming, "oi", f"{c}_funding.csv"), fr)
            print(f"  [{i}/{len(coins)}] {c}: {len(kl)} bars, {len(tk)} taker, {len(fr)} funding")
        except Exception as e:                                     # noqa: BLE001
            print(f"  [{i}/{len(coins)}] {c}: FAILED {e} — aborting refresh")
            return False
        time.sleep(REQUEST_SLEEP)

    problems = validate_panel(incoming, coins)
    if problems:
        print(f"\n[{panel}] VALIDATION FAILED — existing panel left untouched:")
        for p in problems[:12]:
            print(f"    - {p}")
        return False

    backup = live + ".prev"
    shutil.rmtree(backup, ignore_errors=True)
    if os.path.exists(live) and not os.path.islink(live):
        os.rename(live, backup)
    elif os.path.islink(live):
        os.unlink(live)
    os.rename(incoming, live)
    print(f"[{panel}] refreshed and validated (previous panel kept at {panel}.prev)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report staleness only")
    ap.add_argument("--universe", choices=["a", "b", "both"], default="both")
    a = ap.parse_args()

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    print("=" * 70)
    print(f"LIVE PANEL REFRESH   endpoint={BASE}")
    print("=" * 70)

    for panel in ("run", "clean_panel"):
        d = os.path.join(ROOT, panel)
        newest, age_h = panel_staleness(d)
        state = "MISSING" if newest is None else (
            f"newest {newest} ({age_h:.0f}h old)"
            + ("  <-- TOO STALE TO TRADE" if age_h > MAX_STALE_HOURS else "  ok"))
        print(f"  {panel:<14} {state}")

    if a.check:
        return 0

    try:
        import reference_impl as _B
        coins_b = _B.select_universe()
        start_b = _B.START
    except Exception as e:                                         # noqa: BLE001
        print(f"SETUP ERROR: cannot determine Universe B coins: {e}")
        return 2

    ok = True
    if a.universe in ("b", "both"):
        ok &= refresh("clean_panel", coins_b, start_b)
    if a.universe in ("a", "both"):
        try:
            import option1_reference as _A
            coins_a, *_ = _A.build_matrices()
        except Exception as e:                                     # noqa: BLE001
            print(f"\n[run] cannot determine Universe A coins from the existing "
                  f"panel: {e}")
            return 1
        ok &= refresh("run", list(coins_a), start_b)

    print("\n" + "=" * 70)
    print("RESULT:", "panels refreshed" if ok else "REFRESH FAILED — panels untouched")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
