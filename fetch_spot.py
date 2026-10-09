#!/usr/bin/env python3
"""
fetch_spot.py -- Binance SPOT taker flow, matched-venue, 1h bars.

WHY THIS IS NEW DATA
--------------------
Every sleeve in the nine-sleeve book reads the PERP tape: perp price, perp
taker delta, perp OI. Seven candidate 10th sleeves were rejected because they
all re-read price and correlated 0.4-0.7 with `sr`.

Spot flow is a different tape. The same hour can show:

    spot delta  >0   and   perp delta  <0   ->  cash buyers absorbing
                                               leveraged sellers. Historically
                                               the constructive configuration:
                                               someone is paying up to OWN the
                                               coin while the derivative crowd
                                               is positioned against them.

    spot delta  <0   and   perp delta  >0   ->  leveraged longs chasing while
                                               holders distribute into them.
                                               Supply is being handed to the
                                               weakest hands.

Neither of those is visible in the perp tape alone, which is the entire
argument for fetching this.

SOURCE
------
Binance SPOT klines: https://api.binance.com/api/v3/klines
Field layout is IDENTICAL to the futures endpoint fetch_taker.py uses:

    field  5   volume                (base units)
    field  9   taker buy base volume

so    sell  = volume - taker_buy
      delta = 2*taker_buy - volume

Because both numbers come from the same response, they cannot drift apart --
the same single-venue guarantee that fixed the clean_panel splice bug.

SYMBOL MAPPING (the trap)
-------------------------
Perps use 1000x contracts for small-unit coins: 1000PEPEUSDT, 1000SHIBUSDT,
1000BONKUSDT, 1000FLOKIUSDT, 1000RATSUSDT. Spot has NO such pair -- it is
PEPEUSDT, SHIBUSDT, etc. UNPREFIX below handles it.

CONSEQUENCE, and it matters: for those coins the spot PRICE is 1000x smaller
and the spot base VOLUME is 1000x larger than the perp series. So:

    DO NOT compare spot vs perp price levels or raw volumes for these coins.
    DO     compare scale-free quantities -- delta/volume, the sign of delta,
           z-scores of delta/volume. Those are unit-invariant, which is what
           any flow-divergence signal should be built on anyway.

OUTPUT
------
~/spot_data/<COIN>_spot_1h.csv, same columns as ~/taker_data/<COIN>_1h.csv:

    open_time,open,high,low,close,volume,taker_buy,delta

<COIN> is the PERP name (1000PEPE, not PEPE), so files line up 1:1 with
taker_data/ by filename and need no translation at research time.

USAGE
-----
    python3 -u fetch_spot.py --check              # what exists, what is missing
    python3 -u fetch_spot.py --run --limit 3      # fetch 3 coins, verify, then
    python3 -u fetch_spot.py --run                # the rest
    python3 -u fetch_spot.py --verify             # audit files already written

Resumable: a coin already present in ~/spot_data is skipped. Delete its file
to refetch.
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
BASE = "https://api.binance.com/api/v3/klines"
INFO = "https://api.binance.com/api/v3/exchangeInfo"

def resolve_dir(env_var: str, name: str, must_exist: bool):
    """Find a data directory the SAME way book.py does: env var, then cwd,
    then $HOME, then the /tmp/hyro fallback.

    This exists because hardcoding one path is a bug this project has already
    paid for: every sleeve once raised FileNotFoundError on the server because
    hourly_sim/sr2/fvg had `/tmp/hyro` baked in. Returns the first candidate
    that exists; if none do and must_exist, returns None along with the list
    of paths tried so the caller can PRINT them instead of failing silently.
    """
    tried = []
    env = os.environ.get(env_var)
    if env:
        tried.append(env)
        if os.path.isdir(env):
            return env, tried
    for cand in (name,
                 os.path.expanduser(f"~/{name}"),
                 os.path.expanduser(f"~/bot_hyrotrader_v1/{name}"),
                 f"/tmp/hyro/{name}"):
        tried.append(cand)
        if os.path.isdir(cand):
            return cand, tried
    if not must_exist:
        # writing: default to $HOME, created on first write
        return os.path.expanduser(f"~/{name}"), tried
    return None, tried


OUT, _OUT_TRIED = resolve_dir("HYRO_SPOT_DATA_DIR", "spot_data", False)
PERP, _PERP_TRIED = resolve_dir("HYRO_TAKER_DATA_DIR", "taker_data", True)

YEARS = 3.6                                        # match the perp panel
PER_CALL = 1000                                    # spot endpoint caps at 1000
SLEEP = 0.15
PROGRESS_EVERY = 10
MIN_BARS = 5000                                    # ~7 months; below this the
                                                   # coin is useless for a
                                                   # cross-sectional sleeve

# perp contract name  ->  spot pair base
UNPREFIX = {
    "1000PEPE":  "PEPE",
    "1000SHIB":  "SHIB",
    "1000BONK":  "BONK",
    "1000FLOKI": "FLOKI",
    "1000RATS":  "RATS",
    "1000LUNC":  "LUNC",
    "1000XEC":   "XEC",
    "1000SATS":  "1000SATS",   # genuinely named 1000SATS on spot too
}

# coins that are perp-only -- no spot pair exists, do not waste calls or
# print scary warnings for them. (HYPE trades spot on Hyperliquid, not
# Binance; FARTCOIN/ASTER/XPL and friends are listed as perps first.)
KNOWN_PERP_ONLY = {
    "FARTCOIN", "ASTER", "XPL", "HYPE", "AIXBT", "MOODENG", "PNUT",
    "POPCAT", "PUMP", "KAITO", "GRASS", "VIRTUAL", "PENGU", "TRUMP",
}


def log(m: str) -> None:
    print(m, flush=True)


def spot_symbol(coin: str) -> str:
    """Perp coin name -> Binance spot symbol."""
    return f"{UNPREFIX.get(coin, coin)}USDT"


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------
def get(url: str, tries: int = 3):
    """Returns parsed JSON, [] if the symbol is not listed, None if the host
    is geo-blocked (caller must abort -- retrying is pointless)."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "spot/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 451:
                log("    HTTP 451 -- Binance is geo-blocking this host")
                return None
            if e.code == 400:
                return []                      # bad/unlisted symbol
            if e.code == 418 or e.code == 429:
                wait = 5 * (attempt + 1)
                log(f"    HTTP {e.code} rate limit, sleeping {wait}s")
                time.sleep(wait)
                continue
            log(f"    HTTP {e.code} (try {attempt+1}/{tries})")
        except Exception as e:
            log(f"    {type(e).__name__}: {e} (try {attempt+1}/{tries})")
        time.sleep(1.2 * (attempt + 1))
    return []


def listed_spot_symbols() -> set[str]:
    """Which USDT spot pairs are actually TRADING. One call, saves us from
    paginating 3.6 years of nothing for a coin that has no spot market."""
    data = get(INFO)
    if not data or not isinstance(data, dict):
        log("  ! could not read exchangeInfo -- will probe per coin instead")
        return set()
    return {
        s["symbol"] for s in data.get("symbols", [])
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
    }


# --------------------------------------------------------------------------
# which coins
# --------------------------------------------------------------------------
def coins() -> list[str]:
    """Mirror whatever the perp panel holds, so every spot file has a perp
    counterpart to diff against. BTC included here (unlike fetch_taker, which
    excludes it as the market factor) because spot-vs-perp divergence on BTC
    is itself a market-wide signal worth having."""
    if PERP is None:
        log("  ! cannot find the perp panel (taker_data). Paths tried:")
        for p in _PERP_TRIED:
            log(f"      {p}")
        log("  fix: run fetch_taker.py first, or point HYRO_TAKER_DATA_DIR")
        log("       at the directory holding <COIN>_1h.csv")
        raise SystemExit(2)
    names = {
        os.path.basename(f).replace("_1h.csv", "")
        for f in glob.glob(f"{PERP}/*_1h.csv")
    }
    if not names:
        log(f"  ! {PERP} exists but holds no <COIN>_1h.csv files")
        log(f"    contents: {sorted(os.listdir(PERP))[:10]}")
        raise SystemExit(2)
    return sorted(names)


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------
def fetch(coin: str, start_ms: int, end_ms: int):
    """Paginate forward. Returns {open_time: raw_kline} or None if blocked."""
    sym = spot_symbol(coin)
    rows: dict[int, list] = {}
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
            return None                        # geo-blocked -> abort run
        if not batch:
            break
        for k in batch:
            try:
                rows[int(k[0])] = k
            except (ValueError, IndexError, TypeError):
                continue                       # malformed bar, drop it
        try:
            newest = max(int(k[0]) for k in batch)
        except (ValueError, TypeError):
            break
        if newest <= cursor:                   # no forward progress -> stop,
            break                              # otherwise this loops forever
        cursor = newest + 1
        time.sleep(SLEEP)

    return rows


def write(coin: str, rows: dict[int, list]) -> tuple[int, int]:
    """Write atomically via tmp+replace so a crash never leaves a half file
    that the next run would happily skip. Returns (bad_taker, gaps)."""
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{coin}_spot_1h.csv"
    tmp = path + ".tmp"

    bad = 0          # taker_buy > volume: impossible from one venue
    gaps = 0         # missing hours (exchange downtime / thin listing)
    prev_ts = None

    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close",
                    "volume", "taker_buy", "delta"])
        for ts in sorted(rows):
            k = rows[ts]
            try:
                vol = float(k[5])
                tb = float(k[9])
            except (ValueError, IndexError, TypeError):
                continue
            if tb > vol:
                bad += 1
            if prev_ts is not None and ts - prev_ts > 3_600_000:
                gaps += int((ts - prev_ts) / 3_600_000) - 1
            prev_ts = ts
            w.writerow([ts, k[1], k[2], k[3], k[4],
                        f"{vol:.8f}", f"{tb:.8f}", f"{2 * tb - vol:.8f}"])

    os.replace(tmp, path)
    return bad, gaps


# --------------------------------------------------------------------------
# verify: self-diagnosing, same discipline as doctor.py
# --------------------------------------------------------------------------
def verify() -> int:
    """Audit everything already in ~/spot_data. Exit 0 clean, 1 warnings,
    2 do-not-use."""
    log(f"  spot dir : {OUT}")
    log(f"  perp dir : {PERP}")
    files = sorted(glob.glob(f"{OUT}/*_spot_1h.csv"))
    if not files:
        log(f"  nothing in {OUT} -- run --run first")
        return 2

    log(f"  auditing {len(files)} spot files against {PERP}\n")
    worst = 0
    for path in files:
        coin = os.path.basename(path).replace("_spot_1h.csv", "")
        try:
            with open(path) as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            log(f"  {coin:<10} FAIL unreadable: {e}")
            worst = max(worst, 2)
            continue

        if len(rows) < MIN_BARS:
            log(f"  {coin:<10} FAIL only {len(rows):,} bars")
            worst = max(worst, 2)
            continue

        bad = sum(1 for r in rows
                  if float(r["taker_buy"]) > float(r["volume"]))
        zero = sum(1 for r in rows if float(r["volume"]) == 0)
        first = int(rows[0]["open_time"])
        last = int(rows[-1]["open_time"])
        age_h = (time.time() * 1000 - last) / 3_600_000

        # overlap with the perp series -- a spot file we cannot align to a
        # perp file is useless for divergence, however clean it looks
        ppath = f"{PERP}/{coin}_1h.csv"
        overlap = "no perp file"
        if os.path.exists(ppath):
            try:
                with open(ppath) as f:
                    pts = {int(r["open_time"]) for r in csv.DictReader(f)}
                sts = {int(r["open_time"]) for r in rows}
                shared = len(pts & sts)
                pct = 100.0 * shared / max(len(pts), 1)
                overlap = f"{shared:,} shared ({pct:.1f}% of perp)"
                if pct < 80:
                    worst = max(worst, 1)
            except Exception as e:
                overlap = f"perp read failed: {e}"
                worst = max(worst, 1)

        flags = []
        if bad:
            flags.append(f"**{bad} taker>vol**")
            worst = max(worst, 2)
        if zero > len(rows) * 0.02:
            flags.append(f"{zero} zero-vol bars")
            worst = max(worst, 1)
        if age_h > 48:
            flags.append(f"stale {age_h:.0f}h")
            worst = max(worst, 1)

        d0 = dt.datetime.fromtimestamp(first / 1000, dt.timezone.utc).date()
        d1 = dt.datetime.fromtimestamp(last / 1000, dt.timezone.utc).date()
        log(f"  {coin:<10} {len(rows):>6,} bars  {d0} -> {d1}  "
            f"{overlap}  {' '.join(flags)}")

    log("")
    if worst == 0:
        log("  CLEAN -- spot panel is usable")
    elif worst == 1:
        log("  WARNINGS -- usable, read the flags above")
    else:
        log("  DO NOT USE -- fix the FAIL/taker>vol rows first")
    return worst


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    if "--verify" in sys.argv:
        return verify()

    want = coins()
    have = set()
    if os.path.isdir(OUT):
        have = {
            f.replace("_spot_1h.csv", "")
            for f in os.listdir(OUT) if f.endswith("_spot_1h.csv")
        }

    listed = listed_spot_symbols()
    todo, no_spot = [], []
    for c in want:
        if c in have:
            continue
        if c in KNOWN_PERP_ONLY:
            no_spot.append(c)
            continue
        if listed and spot_symbol(c) not in listed:
            no_spot.append(c)
            continue
        todo.append(c)

    log(f"  perp dir:        {PERP}")
    log(f"  spot dir:        {OUT}")
    log(f"  perp panel:      {len(want)} coins")
    log(f"  already fetched: {len(have)}")
    log(f"  no spot pair:    {len(no_spot)}"
        + (f"  ({', '.join(no_spot)})" if no_spot else ""))
    log(f"  to fetch:        {len(todo)}")

    if "--run" not in sys.argv:
        log("\n  run with --run   (--limit N to test a few first)")
        log("  then             --verify")
        return 0

    if "--limit" in sys.argv:
        try:
            n = int(sys.argv[sys.argv.index("--limit") + 1])
            todo = todo[:n]
        except (IndexError, ValueError):
            log("  ! --limit needs an integer")
            return 2

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
        if len(rows) < MIN_BARS:
            log(f"  [{i}/{len(todo)}] {c}: only {len(rows)} bars -- skipped")
            skip += 1
            continue
        bad, gaps = write(c, rows)
        ok += 1
        flags = ""
        if bad:
            flags += f"  ** {bad} bars taker>volume **"
        if gaps:
            flags += f"  ({gaps} missing hours)"
        log(f"  [{i}/{len(todo)}] {c} -> {spot_symbol(c)}: "
            f"{len(rows):,} bars ({time.time()-t0:.0f}s){flags}")

    log(f"\n  done: {ok} written, {skip} skipped -> {OUT}")
    log("  now run:  python3 -u fetch_spot.py --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
