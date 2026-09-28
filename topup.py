#!/usr/bin/env python3
"""
topup.py — append recent bars to existing data files.

`fetch_taker.py` and `fetch_oi.py` SKIP coins they already have, so they never
extend a file. Run once and the data silently ages: doctor.py found taker_data
431 hours stale, which is exactly the 408-hour failure that let the old bot
"succeed" on frozen data for weeks.

This reads the last timestamp in each file, fetches from there, and appends
only genuinely new bars. Idempotent -- running it twice adds nothing the second
time.

    python3 topup.py                # taker + OI
    python3 topup.py --taker-only
    python3 topup.py --check        # report ages, change nothing

Put it in cron AHEAD of the bot:

    0 * * * * cd ~/bot_hyrotrader_v1 && python3 topup.py >> ~/topup.log 2>&1
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
import datetime as dt

BINANCE = "https://fapi.binance.com/fapi/v1/klines"
BYBIT_OI = "https://api.bybit.com/v5/market/open-interest"
TAKER_DIR = "taker_data"
OI_DIR = "oi_data"
SLEEP = 0.12

ALIAS = {"SHIB": "1000SHIBUSDT", "PEPE": "1000PEPEUSDT",
         "BONK": "1000BONKUSDT", "FLOKI": "1000FLOKIUSDT",
         "1000SHIB": "1000SHIBUSDT", "1000PEPE": "1000PEPEUSDT",
         "1000RATS": "1000RATSUSDT", "1000BONK": "1000BONKUSDT",
         "1000FLOKI": "1000FLOKIUSDT"}
BYBIT_ALIAS = {"SHIB": "SHIB1000USDT", "1000SHIB": "SHIB1000USDT"}


def log(m):
    print(m, flush=True)


def get(url, tries=3):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "topup/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (400, 451):
                return None
            time.sleep(1.2 * (a + 1))
        except Exception:
            time.sleep(1.2 * (a + 1))
    return None


def last_ts(path: str, col: str) -> int | None:
    """Last timestamp in a CSV, read from the END rather than parsing it all."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode(errors="ignore").strip().splitlines()
        with open(path) as f:
            header = f.readline().strip().split(",")
        if col not in header:
            return None
        i = header.index(col)
        for line in reversed(tail):
            parts = line.split(",")
            if len(parts) <= i:
                continue
            try:
                return int(float(parts[i]))
            except ValueError:
                continue
    except Exception:
        return None
    return None


def topup_taker(coin: str) -> int:
    path = f"{TAKER_DIR}/{coin}_1h.csv"
    if not os.path.exists(path):
        return 0
    last = last_ts(path, "open_time")
    if last is None:
        log(f"  {coin}: cannot read last timestamp, skipped")
        return 0
    start = last + 1
    now = int(time.time() * 1000)
    if now - start < 3600_000:
        return 0                                   # less than one bar behind
    sym = ALIAS.get(coin, f"{coin}USDT")
    rows, cursor, added = [], start, 0
    while cursor < now:
        batch = get(f"{BINANCE}?symbol={sym}&interval=1h"
                    f"&startTime={cursor}&limit=1000")
        if not batch:
            break
        rows.extend(batch)
        newest = max(int(k[0]) for k in batch)
        if newest <= cursor:
            break
        cursor = newest + 1
        time.sleep(SLEEP)
    if not rows:
        return 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for k in sorted(rows, key=lambda r: int(r[0])):
            ts = int(k[0])
            if ts <= last:
                continue                            # never duplicate
            try:
                vol, tb = float(k[5]), float(k[9])
            except (ValueError, IndexError):
                continue
            w.writerow([ts, k[1], k[2], k[3], k[4],
                        f"{vol:.8f}", f"{tb:.8f}", f"{2*tb - vol:.8f}"])
            added += 1
    return added


def topup_oi(coin: str) -> int:
    path = f"{OI_DIR}/{coin}_oi_1h.csv"
    if not os.path.exists(path):
        return 0
    last = last_ts(path, "timestamp")
    if last is None:
        return 0
    now = int(time.time() * 1000)
    if now - last < 3600_000:
        return 0
    sym = BYBIT_ALIAS.get(coin, f"{coin}USDT")
    d = get(f"{BYBIT_OI}?category=linear&symbol={sym}&intervalTime=1h&limit=200")
    if not d or d.get("retCode") != 0:
        return 0
    out = []
    for r in d.get("result", {}).get("list", []):
        try:
            ts = int(r["timestamp"])
            if ts > last:
                out.append((ts, float(r["openInterest"])))
        except (KeyError, ValueError, TypeError):
            continue
    if not out:
        return 0
    # MATCH THE FILE'S OWN HEADER. The originals were written by pandas WITH an
    # index column (idx,timestamp,open_interest); appending two columns to a
    # three-column file shifts every new row one place left, and the loader
    # then reads the open-interest value AS the timestamp. Every OI-gated
    # sleeve breaks silently.
    with open(path) as f:
        header = f.readline().strip().split(",")
    has_idx = len(header) == 3 and header[0] in ("idx", "")
    n_existing = sum(1 for _ in open(path)) - 1
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for i, (ts, oi) in enumerate(sorted(out)):
            w.writerow([n_existing + i, ts, f"{oi:.8f}"] if has_idx
                       else [ts, f"{oi:.8f}"])
    return len(out)


def report_ages():
    now = time.time()
    for d, pat, col in ((TAKER_DIR, "*_1h.csv", "open_time"),
                        (OI_DIR, "*_oi_1h.csv", "timestamp")):
        files = sorted(glob.glob(f"{d}/{pat}"))
        if not files:
            log(f"  {d}: no files")
            continue
        ages = []
        for p in files:
            t = last_ts(p, col)
            if t:
                ages.append((os.path.basename(p), (now - t / 1000) / 3600))
        if not ages:
            continue
        ages.sort(key=lambda x: -x[1])
        log(f"  {d}: {len(files)} files, "
            f"oldest bar {ages[0][1]:.1f}h ago ({ages[0][0]}), "
            f"newest {ages[-1][1]:.1f}h ago")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--taker-only", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(TAKER_DIR):
        log(f"run this from the directory containing {TAKER_DIR}/")
        sys.exit(1)

    if a.check:
        log("DATA AGES (last BAR, not file mtime)")
        report_ages()
        sys.exit(0)

    coins = sorted(os.path.basename(f).replace("_1h.csv", "")
                   for f in glob.glob(f"{TAKER_DIR}/*_1h.csv"))
    log(f"topping up {len(coins)} coins  {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC")
    tk = oi = 0
    for c in coins:
        n = topup_taker(c)
        tk += n
        if n:
            log(f"  {c:<12}+{n} taker bars")
    if not a.taker_only:
        for c in coins:
            n = topup_oi(c)
            oi += n
            if n:
                log(f"  {c:<12}+{n} OI bars")
    log(f"done: +{tk} taker bars, +{oi} OI bars")
    report_ages()
