#!/usr/bin/env python3
"""
fetch_funding.py — funding rate history from Bybit.

WHY: funding z-fade scored 0.83 with halves -0.74 / 2.40 -- the same decay
profile that killed FVG at 4h and 6h. FVG survived at 12h, so this deserves a
sweep before being written off. But there is no funding data in the repo.

NOTE ON "TIMEFRAMES": funding settles every 8 hours, so there are only three
observations a day. The sweep axis is the Z-SCORE LOOKBACK and the HOLDING
PERIOD, not bar size -- resampling to 4h would just interpolate.

Writes ~/funding_data/<COIN>_funding.csv:  ts,rate

USAGE
    python3 -u fetch_funding.py --check
    python3 -u fetch_funding.py --run --limit 2
    python3 -u fetch_funding.py --run
"""
from __future__ import annotations
import os, sys, csv, glob, time, json, urllib.request, urllib.error
import datetime as dt

BASE = "https://api.bybit.com/v5/market/funding/history"
OUT = os.path.expanduser("~/funding_data")
SRC = os.path.expanduser("~/bot_hyrotrader_v1/clean_panel/hist")
YEARS = 3.6
PER_CALL = 200
SLEEP = 0.15

ALIAS = {"SHIB": "SHIB1000USDT", "1000SHIB": "SHIB1000USDT"}


def log(m):
    print(m, flush=True)


def sym_for(c):
    return ALIAS.get(c, f"{c}USDT")


def get(url, tries=3):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "funding/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode())
            if d.get("retCode") == 0:
                return d.get("result", {}).get("list", [])
            return []
        except Exception as e:
            if a == tries - 1:
                log(f"    {type(e).__name__}: {e}")
                return []
            time.sleep(1.2 * (a + 1))
    return []


def coins():
    n = {os.path.basename(f).replace("_1h.csv", "")
         for f in glob.glob(f"{SRC}/*_1h.csv")}
    return sorted(x for x in n if x != "BTC")


def fetch(coin, start_ms, end_ms):
    """Page backward -- Bybit returns newest first."""
    sym = sym_for(coin)
    rows = {}
    cursor = end_ms
    blanks = 0
    while cursor > start_ms:
        url = (f"{BASE}?category=linear&symbol={sym}"
               f"&endTime={cursor}&limit={PER_CALL}")
        batch = get(url)
        if not batch:
            blanks += 1
            if blanks >= 2:
                break
            cursor -= PER_CALL * 8 * 3600 * 1000
            continue
        blanks = 0
        for r in batch:
            try:
                rows[int(r["fundingRateTimestamp"])] = float(r["fundingRate"])
            except (ValueError, KeyError, TypeError):
                continue
        oldest = min(int(r["fundingRateTimestamp"]) for r in batch)
        if oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(SLEEP)
    return rows


def write(coin, rows):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{coin}_funding.csv"
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "rate"])
        for ts in sorted(rows):
            w.writerow([ts, f"{rows[ts]:.10f}"])
    os.replace(tmp, path)


if __name__ == "__main__":
    want = coins()
    have = set()
    if os.path.isdir(OUT):
        have = {f.replace("_funding.csv", "") for f in os.listdir(OUT)
                if f.endswith("_funding.csv")}
    todo = [c for c in want if c not in have]
    log(f"  {len(want)} coins   already have {len(have)}   to fetch {len(todo)}")
    if "--run" not in sys.argv:
        log("\n  run with --run  (--limit N to test first)")
        raise SystemExit
    if "--limit" in sys.argv:
        todo = todo[:int(sys.argv[sys.argv.index("--limit") + 1])]

    end = int(time.time() * 1000)
    start = end - int(YEARS * 365 * 86400 * 1000)
    log(f"\n  fetching {len(todo)} coins from "
        f"{dt.datetime.fromtimestamp(start/1000, dt.timezone.utc).date()}")
    log("  funding settles every 8h -> ~3,900 rows per coin over 3.6y\n")

    ok = skip = 0
    for i, c in enumerate(todo, 1):
        t0 = time.time()
        rows = fetch(c, start, end)
        if len(rows) < 500:
            log(f"  [{i}/{len(todo)}] {c}: only {len(rows)} rows -- skipped")
            skip += 1
            continue
        write(c, rows)
        ok += 1
        neg = sum(1 for v in rows.values() if v < 0)
        log(f"  [{i}/{len(todo)}] {c}: {len(rows):,} rows, "
            f"{neg/len(rows)*100:.0f}% negative  ({time.time()-t0:.0f}s)")
    log(f"\n  done: {ok} written, {skip} skipped -> {OUT}")
    log("  push with:  cp -r ~/funding_data ~/bot_hyrotrader_v1/ && "
        "echo '!funding_data/*.csv' >> ~/bot_hyrotrader_v1/.gitignore")
