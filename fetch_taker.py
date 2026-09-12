#!/usr/bin/env python3
"""
fetch_taker.py — matched-venue price + taker volume from Binance futures.

WHY: clean_panel's taker_buy_base exceeds total volume on 60% of bars (median
ratio 1.07, stdev 0.30). It is Binance taker volume spliced against Bybit price
volume -- two venues, so the "delta" derived from it is meaningless. That is
why the BOS sleeve has produced zero weights on every live cycle (`dn < 0` is
unsatisfiable when the value is always positive) and why the delta sleeve ranks
on noise.

Binance klines return BOTH in one response, so they cannot drift apart:
    field  5  volume
    field  9  taker buy base asset volume
    sell volume   = volume - taker_buy
    signed delta  = 2*taker_buy - volume

Writes ~/taker_data/<COIN>_1h.csv with columns:
    open_time,open,high,low,close,volume,taker_buy,delta

NOTE: Binance blocks some hosting regions. If every request 451s, that is
geo-blocking, not a bug -- the same data is on Bybit's public trade endpoint
but only for the last ~1000 prints, which is no use for history.

USAGE
    python3 -u fetch_taker.py --check
    python3 -u fetch_taker.py --run --limit 2
    python3 -u fetch_taker.py --run
"""
from __future__ import annotations
import os, sys, csv, glob, time, json, urllib.request, urllib.error
import datetime as dt

BASE = "https://fapi.binance.com/fapi/v1/klines"
OUT = os.path.expanduser("~/taker_data")
SRC = os.path.expanduser("~/bot_hyrotrader_v1/clean_panel/hist")
YEARS = 3.6
PER_CALL = 1500
SLEEP = 0.15
PROGRESS_EVERY = 10

ALIAS = {"SHIB": "1000SHIBUSDT", "PEPE": "1000PEPEUSDT",
         "BONK": "1000BONKUSDT", "FLOKI": "1000FLOKIUSDT",
         "1000SHIB": "1000SHIBUSDT", "1000PEPE": "1000PEPEUSDT",
         "1000RATS": "1000RATSUSDT"}


def log(m):
    print(m, flush=True)


def sym_for(c):
    return ALIAS.get(c, f"{c}USDT")


def get(url, tries=3):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "taker/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 451:
                log("    HTTP 451 -- Binance is geo-blocking this host")
                return None
            if e.code == 400:
                return []            # symbol not listed
            log(f"    HTTP {e.code} (try {a+1}/{tries})")
        except Exception as e:
            log(f"    {type(e).__name__} (try {a+1}/{tries})")
        time.sleep(1.2 * (a + 1))
    return []


# The scanner's 60-coin universe. Backtests so far used only the 24 in
# clean_panel; this widens them to match what the live scanner watches.
SCANNER_COINS = [
    "AAVE","ADA","AIXBT","ALGO","APT","ARB","ASTER","ATOM","AVAX","BCH","BNB",
    "BONK","BTC","CRV","DOGE","DOT","ETC","ETH","FARTCOIN","FIL","FLOKI",
    "GRASS","HBAR","HYPE","INJ","JTO","JUP","KAITO","LDO","LINK","LIT","LTC",
    "MOODENG","NEAR","ONDO","OP","ORDI","PENGU","PEPE","PNUT","POL","POPCAT",
    "PUMP","RENDER","S","SHIB","SOL","STX","SUI","TAO","TIA","TRUMP","TRX",
    "UNI","VIRTUAL","WIF","WLD","XPL","XRP","ZEC",
]


def coins():
    if "--scanner" in sys.argv:
        return sorted(c for c in SCANNER_COINS if c != "BTC")
    names = {os.path.basename(f).replace("_1h.csv", "")
             for f in glob.glob(f"{SRC}/*_1h.csv")}
    return sorted(n for n in names if n != "BTC")


def fetch(coin, start_ms, end_ms):
    sym = sym_for(coin)
    rows = {}
    cursor = start_ms
    calls = 0
    while cursor < end_ms:
        calls += 1
        if calls % PROGRESS_EVERY == 0:
            log(f"      {coin}: {len(rows):,} bars ({calls} calls)...")
        url = (f"{BASE}?symbol={sym}&interval=1h"
               f"&startTime={cursor}&limit={PER_CALL}")
        batch = get(url)
        if batch is None:
            return None                     # geo-blocked, abort entirely
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
    path = f"{OUT}/{coin}_1h.csv"
    tmp = path + ".tmp"
    bad = 0
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close",
                    "volume", "taker_buy", "delta"])
        for ts in sorted(rows):
            k = rows[ts]
            try:
                vol = float(k[5]); tb = float(k[9])
            except (ValueError, IndexError):
                continue
            if tb > vol:            # must never happen from a single venue
                bad += 1
            w.writerow([ts, k[1], k[2], k[3], k[4],
                        f"{vol:.8f}", f"{tb:.8f}", f"{2*tb - vol:.8f}"])
    os.replace(tmp, path)
    return bad


if __name__ == "__main__":
    want = coins()
    log(f"  {len(want)} coins from {SRC}")
    have = set()
    if os.path.isdir(OUT):
        have = {f.replace("_1h.csv", "") for f in os.listdir(OUT)
                if f.endswith("_1h.csv")}
    todo = [c for c in want if c not in have]
    log(f"  already fetched: {len(have)}   to fetch: {len(todo)}")
    if "--run" not in sys.argv:
        log("\n  run with --run  (--limit N to test a few first)")
        raise SystemExit

    if "--limit" in sys.argv:
        todo = todo[:int(sys.argv[sys.argv.index("--limit") + 1])]

    end = int(time.time() * 1000)
    start = end - int(YEARS * 365 * 86400 * 1000)
    log(f"\n  fetching {len(todo)} coins from "
        f"{dt.datetime.fromtimestamp(start/1000, dt.timezone.utc).date()}\n")

    ok = skip = 0
    for i, c in enumerate(todo, 1):
        t0 = time.time()
        rows = fetch(c, start, end)
        if rows is None:
            log("  ABORTING -- Binance is blocking this host")
            break
        if len(rows) < 5000:
            log(f"  [{i}/{len(todo)}] {c}: only {len(rows)} bars -- skipped")
            skip += 1
            continue
        bad = write(c, rows)
        ok += 1
        flag = f"  ** {bad} bars where taker>volume **" if bad else ""
        log(f"  [{i}/{len(todo)}] {c}: {len(rows):,} bars "
            f"({time.time()-t0:.0f}s){flag}")

    log(f"\n  done: {ok} written, {skip} skipped -> {OUT}")
    log("  sanity check one file:")
    log("    python3 -c \"import csv;r=list(csv.reader(open('$HOME/taker_data/SOL_1h.csv')))[1:];"
        "print(sum(1 for x in r if float(x[6])>float(x[5])),'of',len(r),'bad')\"")
