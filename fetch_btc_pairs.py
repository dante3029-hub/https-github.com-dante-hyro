#!/usr/bin/env python3
"""
fetch_btc_pairs.py — real BTC-denominated pairs (SOLBTC, ADABTC, ...).

These are actual markets with their own order books and volume, not a ratio
computed by division. Division gives the right PRICE but the alt's USDT volume
-- and S/R builds its levels from candle-direction volume, so the real pair
matters there.

SOURCE: Binance SPOT (api.binance.com). The USDT data is futures (fapi), so
these are a different venue -- fine for signal generation, and the trade is
still executed on the USDT perp.

Not every alt has a BTC pair, and the ones that do have shorter histories and
thinner volume. The script reports what it got so thin pairs can be dropped.

Writes ~/btc_pairs/<COIN>BTC_1h.csv:
    open_time,open,high,low,close,volume,taker_buy,delta

USAGE
    python3 -u fetch_btc_pairs.py --check
    python3 -u fetch_btc_pairs.py --run --limit 3
    python3 -u fetch_btc_pairs.py --run
"""
from __future__ import annotations
import os, sys, csv, glob, time, json, urllib.request, urllib.error
import datetime as dt

BASE = "https://api.binance.com/api/v3/klines"
OUT = os.path.expanduser("~/btc_pairs")
SRC = os.path.expanduser("~/bot_hyrotrader_v1/taker_data")
YEARS = 3.6
PER_CALL = 1000
SLEEP = 0.15

# the 1000x tickers are USDT-perp conventions; spot uses the plain symbol
UNPREFIX = {"1000PEPE": "PEPE", "1000SHIB": "SHIB", "1000BONK": "BONK",
            "1000FLOKI": "FLOKI", "1000RATS": "RATS", "1000LUNC": "LUNC"}


def log(m):
    print(m, flush=True)


def spot_symbol(coin):
    return UNPREFIX.get(coin, coin) + "BTC"


def get(url, tries=3):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "btcpairs/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return None                 # symbol does not exist
            if e.code == 451:
                log("    HTTP 451 — Binance is geo-blocking this host")
                return "BLOCKED"
            log(f"    HTTP {e.code} (try {a+1}/{tries})")
        except Exception as e:
            log(f"    {type(e).__name__} (try {a+1}/{tries})")
        time.sleep(1.2 * (a + 1))
    return []


def coins():
    n = {os.path.basename(f).replace("_1h.csv", "")
         for f in glob.glob(f"{SRC}/*_1h.csv")}
    return sorted(x for x in n if x != "BTC")


def fetch(coin, start_ms, end_ms):
    sym = spot_symbol(coin)
    rows, cursor, calls = {}, start_ms, 0
    while cursor < end_ms:
        calls += 1
        if calls % 10 == 0:
            log(f"      {sym}: {len(rows):,} bars...")
        batch = get(f"{BASE}?symbol={sym}&interval=1h"
                    f"&startTime={cursor}&limit={PER_CALL}")
        if batch == "BLOCKED":
            return "BLOCKED"
        if batch is None:
            return None                     # no such pair
        if not batch:
            break
        for k in batch:
            try:
                rows[int(k[0])] = k
            except (ValueError, IndexError):
                continue
        newest = max(int(k[0]) for k in batch)
        if newest <= cursor:
            break
        cursor = newest + 1
        time.sleep(SLEEP)
    return rows


def write(coin, rows):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{spot_symbol(coin)}_1h.csv"
    tmp = path + ".tmp"
    bad = 0
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close",
                    "volume", "taker_buy", "delta"])
        for ts in sorted(rows):
            k = rows[ts]
            try:
                vol, tb = float(k[5]), float(k[9])
            except (ValueError, IndexError):
                continue
            if tb > vol:
                bad += 1
            w.writerow([ts, k[1], k[2], k[3], k[4],
                        f"{vol:.8f}", f"{tb:.8f}", f"{2*tb - vol:.8f}"])
    os.replace(tmp, path)
    return bad


if __name__ == "__main__":
    want = coins()
    have = {f.replace("_1h.csv", "") for f in os.listdir(OUT)} if os.path.isdir(OUT) else set()
    todo = [c for c in want if spot_symbol(c) not in have]
    log(f"  {len(want)} alts   already have {len(have)}   to fetch {len(todo)}")
    if "--run" not in sys.argv:
        log("\n  run with --run   (--limit N to test a few first)")
        raise SystemExit
    if "--limit" in sys.argv:
        todo = todo[:int(sys.argv[sys.argv.index("--limit") + 1])]

    end = int(time.time() * 1000)
    start = end - int(YEARS * 365 * 86400 * 1000)
    log(f"\n  fetching {len(todo)} BTC pairs from Binance SPOT")
    log(f"  from {dt.datetime.fromtimestamp(start/1000, dt.timezone.utc).date()}\n")

    ok = missing = thin = 0
    for i, c in enumerate(todo, 1):
        t0 = time.time()
        rows = fetch(c, start, end)
        if rows == "BLOCKED":
            log("  ABORTING — geo-blocked")
            break
        if rows is None:
            log(f"  [{i}/{len(todo)}] {spot_symbol(c):<12} no such pair on spot")
            missing += 1
            continue
        if len(rows) < 5000:
            log(f"  [{i}/{len(todo)}] {spot_symbol(c):<12} only {len(rows):,} bars — thin, skipped")
            thin += 1
            continue
        bad = write(c, rows)
        ok += 1
        flag = f"  ** {bad} bars taker>volume **" if bad else ""
        log(f"  [{i}/{len(todo)}] {spot_symbol(c):<12} {len(rows):,} bars "
            f"({time.time()-t0:.0f}s){flag}")

    log(f"\n  done: {ok} written, {missing} no pair, {thin} too thin -> {OUT}")
    log("  push with:")
    log("    cp -r ~/btc_pairs ~/bot_hyrotrader_v1/ && "
        "echo '!btc_pairs/*.csv' >> ~/bot_hyrotrader_v1/.gitignore")
