#!/usr/bin/env python3
"""
scanner.py — multi-timeframe pattern + divergence alerts to Discord.

TWO CHANNELS
    FAST  4h, 5h        compact grid, fires often
    SLOW  6h, 12h, 1d   full detail with entry/stop/target, fires rarely

CELL FORMAT
    BP🔥 S5 NEW
    |    |  |  └ detected on the bar that just closed
    |    |  └─── divergence: S strong · R regular · H hidden, + osc count
    |    └────── open interest expanding on the signal bar
    └─────────── pattern code (legend printed with each scan)

Note: the source only grades REGULAR divergences as strong, so S/R appear but
never "strong hidden" -- that is the indicator's design, not a limitation here.

WHAT THE TESTING SAID, so you can weight these properly:
  * 6h patterns alone were the best result: Sharpe 1.47, both halves positive
    (1.04 / 1.95), 2,269 trades, holding across a 5h-7h cluster.
  * The OI filter is the one component that reliably helped -- at 6h it took
    the random control from 1.01 to -0.72, widening the gap to 1.63.
  * Divergence alone NEVER had both halves positive across 20 configurations.
    Requiring STRONG divergence made patterns WORSE (1.56 -> -0.12 at 6h).
    So treat S5 as information, not as a better signal than R3.
  * ~160 tests were run in development. Paper-trade before funding.

SETUP
    export DISCORD_WEBHOOK_FAST='https://discord.com/api/webhooks/...'
    export DISCORD_WEBHOOK_SLOW='https://discord.com/api/webhooks/...'
    python3 -u scanner.py --once --dry
    python3 -u scanner.py --loop
"""
from __future__ import annotations
import os, sys, json, time, math, urllib.request, urllib.error
import datetime as dt
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BYBIT = "https://api.bybit.com"
WH_FAST = os.environ.get("DISCORD_WEBHOOK_FAST", os.environ.get("DISCORD_WEBHOOK", ""))
WH_SLOW = os.environ.get("DISCORD_WEBHOOK_SLOW", WH_FAST)
STATE = os.path.expanduser("~/scanner_state.json")

FAST_TFS = [4, 5]
SLOW_TFS = [6, 12, 24]
# The most recent signal is shown until a NEWER one replaces it -- mirroring the
# indicator's own max_patterns=1 behaviour, where a drawn pattern stays on the
# chart until the next one appears. NEW marks the bar it fired on; anything older
# carries its age in bars, so a stale signal is visible as stale rather than
# silently dropped.
FRESH_BARS = None           # None = no cutoff, show until replaced

COINS = [
    "AAVE", "ADA", "AIXBT", "ALGO", "APT", "ARB", "ASTER", "ATOM", "AVAX",
    "BCH", "BNB", "BONK", "BTC", "CRV", "DOGE", "DOT", "ETC", "ETH",
    "FARTCOIN", "FIL", "FLOKI", "GRASS", "HBAR", "HYPE", "INJ", "JTO",
    "JUP", "KAITO", "LDO", "LINK", "LIT", "LTC", "MOODENG", "NEAR", "ONDO",
    "OP", "ORDI", "PENGU", "PEPE", "PNUT", "POL", "POPCAT", "PUMP",
    "RENDER", "S", "SHIB", "SOL", "STX", "SUI", "TAO", "TIA", "TRUMP",
    "TRX", "UNI", "VIRTUAL", "WIF", "WLD", "XPL", "XRP", "ZEC",
]

ALIAS = {"SHIB": "SHIB1000USDT", "PEPE": "1000PEPEUSDT",
         "BONK": "1000BONKUSDT", "FLOKI": "1000FLOKIUSDT"}

# Two chars, never distinguished by case alone -- "DT" (double top) vs "dT"
# (descending triangle) was unreadable at a glance.
CODE = {
    "Double Top": "DT", "Double Bottom": "DB",
    "Triple Top": "TT", "Triple Bottom": "TB",
    "Head & Shoulders": "HS", "Inv Head & Shoulders": "IH",
    "Bullish Flag": "FB", "Bearish Flag": "FR",
    "Bullish Pennant": "PB", "Bearish Pennant": "PR",
    "Rising Wedge": "WR", "Falling Wedge": "WF",
    "Ascending Triangle": "TA", "Descending Triangle": "TD",
    "Symmetrical Triangle": "TS", "Rectangle": "RC",
    "Cup & Handle": "CU", "Inv Cup & Handle": "CI",
}


def log(m):
    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} {m}", flush=True)


def sym_for(c):
    return ALIAS.get(c, f"{c}USDT")


def get(url, tries=3):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scanner/1"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, TimeoutError, OSError):
            if a == tries - 1:
                return None
            time.sleep(1.5 * (a + 1))
    return None


_bar_cache: dict[str, object] = {}


def fetch_1h(coin):
    """Hourly bars once per coin per cycle; every timeframe resamples from these."""
    if coin in _bar_cache:
        return _bar_cache[coin]
    import pandas as pd
    rows, end = [], int(time.time() * 1000)
    for _ in range(8):
        d = get(f"{BYBIT}/v5/market/kline?category=linear&symbol={sym_for(coin)}"
                f"&interval=60&end={end}&limit=1000")
        if not d or d.get("retCode") != 0:
            _bar_cache[coin] = None
            return None
        batch = d.get("result", {}).get("list", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(k[0]) for k in batch)
        if oldest >= end:
            break
        end = oldest - 1
        time.sleep(0.12)
    if len(rows) < 4000:
        _bar_cache[coin] = None
        return None
    seen = {}
    for k in rows:
        try:
            seen[int(k[0])] = k
        except (ValueError, IndexError):
            continue
    ts = sorted(seen)
    df = pd.DataFrame({
        "ts": pd.to_datetime(ts, unit="ms"),
        "open": [float(seen[t][1]) for t in ts],
        "high": [float(seen[t][2]) for t in ts],
        "low": [float(seen[t][3]) for t in ts],
        "close": [float(seen[t][4]) for t in ts],
        "volume": [float(seen[t][5]) for t in ts],
    }).set_index("ts").sort_index()
    _bar_cache[coin] = df
    return df


def fetch_native(coin, hours):
    """12h and 1d need ~8,400 and ~16,800 hourly bars to make 700 -- more than
    the hourly fetch returns, which is why those columns were always empty.
    Bybit serves these intervals directly, so pull them natively instead."""
    import pandas as pd
    iv = {12: "720", 24: "D"}.get(hours)
    if iv is None:
        return None
    rows, end = [], int(time.time() * 1000)
    for _ in range(3):
        d = get(f"{BYBIT}/v5/market/kline?category=linear&symbol={sym_for(coin)}"
                f"&interval={iv}&end={end}&limit=1000")
        if not d or d.get("retCode") != 0:
            return None
        batch = d.get("result", {}).get("list", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(k[0]) for k in batch)
        if oldest >= end:
            break
        end = oldest - 1
        time.sleep(0.12)
    if len(rows) < 700:
        return None
    seen = {}
    for k in rows:
        try:
            seen[int(k[0])] = k
        except (ValueError, IndexError):
            continue
    ts = sorted(seen)
    d = pd.DataFrame({
        "ts": pd.to_datetime(ts, unit="ms"),
        "open": [float(seen[t][1]) for t in ts],
        "high": [float(seen[t][2]) for t in ts],
        "low": [float(seen[t][3]) for t in ts],
        "close": [float(seen[t][4]) for t in ts],
        "volume": [float(seen[t][5]) for t in ts],
    }).set_index("ts").sort_index()
    now_ms = int(time.time() * 1000)
    if len(d) and int(d.index[-1].timestamp()*1000) + hours*3600*1000 > now_ms:
        d = d.iloc[:-1]
    return d


def resample(df, hours):
    import pandas as pd
    r = f"{hours}h"
    d = pd.DataFrame({
        "open": df["open"].resample(r).first(),
        "high": df["high"].resample(r).max(),
        "low": df["low"].resample(r).min(),
        "close": df["close"].resample(r).last(),
        "volume": df["volume"].resample(r).sum(),
    }).dropna()
    now_ms = int(time.time() * 1000)
    if len(d) and int(d.index[-1].timestamp()*1000) + hours*3600*1000 > now_ms:
        d = d.iloc[:-1]                 # drop the in-progress bar
    return d


def oi_state(coin):
    d = get(f"{BYBIT}/v5/market/open-interest?category=linear"
            f"&symbol={sym_for(coin)}&intervalTime=4h&limit=2")
    if not d or d.get("retCode") != 0:
        return None, None
    rows = d.get("result", {}).get("list", [])
    if len(rows) < 2:
        return None, None
    try:
        a, b = float(rows[0]["openInterest"]), float(rows[1]["openInterest"])
    except (ValueError, KeyError, TypeError):
        return None, None
    return (a > b, (a/b - 1)*100) if b > 0 else (None, None)


def scan_tf(coin, df, hours):
    """Returns (cell_text, detail_dict) or (None, None)."""
    from markittick_detector import detect as pdetect
    from divergence_detector import detect as ddetect
    d = fetch_native(coin, hours) if hours in (12, 24) else resample(df, hours)
    if d is None or len(d) < 700:
        return None, None
    last_bar = len(d) - 1

    pats = pdetect(d)
    div = ddetect(d)

    pat = pats[-1] if pats else None
    dv = div[-1] if div else None
    if FRESH_BARS is not None:
        if pat and last_bar - pat[0] > FRESH_BARS:
            pat = None
        if dv and last_bar - (dv["bar"] + 5) > FRESH_BARS:
            dv = None
    if pat is None and dv is None:
        return None, None

    parts = []
    fresh = False
    age = None
    if pat:
        b, ts, is_bull, name, entry, stop, target = pat
        parts.append(CODE.get(name, name[:3]))
        age = last_bar - b
        if age == 0:
            fresh = True
    if dv:
        letter = "S" if dv["strong"] else ("H" if dv["kind"] == "hidden" else "R")
        parts.append(f"{letter}{dv['count']}")
        d_age = last_bar - (dv["bar"] + 5)
        if d_age == 0:
            fresh = True
        age = d_age if age is None else min(age, d_age)
    return " ".join(parts), dict(pat=pat, div=dv, fresh=fresh, tf=hours, age=age)


def build_report(results, oi, tfs, title):
    """Grid: one row per coin, one column per timeframe. Columns are 8 wide so
    three of them still fit a phone without wrapping."""
    rows = []
    for c in COINS:
        cells = [results.get((c, t), (None, None))[0] for t in tfs]
        if not any(cells):
            continue
        flame = "\U0001F525" if oi.get(c, (None, None))[0] else ""
        out = []
        for cell, t in zip(cells, tfs):
            if not cell:
                out.append("\u00b7")
                continue
            det = results.get((c, t), (None, {}))[1] or {}
            x = cell + (flame if det.get("pat") else "")
            if det.get("fresh"):
                x += "\u2022"
            elif det.get("age"):
                x += f" {det['age']}"
            out.append(x)
        rows.append((c, out))
    if not rows:
        return None
    when = dt.datetime.now(dt.timezone.utc).strftime("%d %b %H:%M")
    hdr = "".join(f"{(f'{t}h' if t < 24 else '1d'):>9}" for t in tfs)
    L = [f"`{title}` \u00b7 `{when} UTC` \u00b7 {len(rows)}/{len(COINS)}",
         "```", f"{'':<8}{hdr}"]
    for c, out in rows:
        L.append(f"{c:<8}" + "".join(f"{o:>9}" for o in out))
    L.append("```")
    return "\n".join(L)


LEGEND = """**LEGEND**
```
DT DB  double top / bottom
TT TB  triple top / bottom
HS IH  head & shoulders / inv
FB FR  flag  bull / bear
PB PR  pennant  bull / bear
WR WF  wedge  rising / falling
TA TD  triangle  asc / desc
TS     triangle  symmetrical
RC     rectangle
CU CI  cup & handle / inv

S6 R3 H4   divergence: strong,
           regular, hidden + count

\U0001F525  open interest expanding
\u2022  fired this bar
number  bars since it fired
```"""


def post(msg, webhook, dry=False):
    if not msg:
        return
    print(msg)
    if dry or not webhook:
        return
    chunks, cur = [], ""
    for ln in msg.split("\n"):
        if len(cur) + len(ln) + 1 > 1900:
            chunks.append(cur); cur = ln
        else:
            cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        chunks.append(cur)
    for ch in chunks:
        # Discord rejects requests with urllib's default User-Agent
        # ("Python-urllib/3.x") with a 403 -- curl worked, the script did not.
        req = urllib.request.Request(
            webhook, data=json.dumps({"content": ch}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "DiscordBot (scanner, 1.0)"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            log(f"  post failed: {e}")
        time.sleep(0.4)


_legend_sent = False


def run_once(dry=False):
    global _legend_sent
    _bar_cache.clear()
    results, oi = {}, {}
    log(f"scanning {len(COINS)} coins")
    for i, c in enumerate(COINS, 1):
        try:
            df = fetch_1h(c)
            if df is None:
                log(f"  [{i}/{len(COINS)}] {c}: no data")
                continue
            hit = False
            for t in FAST_TFS + SLOW_TFS:
                cell, det = scan_tf(c, df, t)
                if cell:
                    results[(c, t)] = (cell, det)
                    hit = True
            if hit:
                oi[c] = oi_state(c)
                log(f"  [{i}/{len(COINS)}] {c}: "
                    + ", ".join(f"{t}h={results[(c,t)][0]}"
                                for t in FAST_TFS+SLOW_TFS if (c, t) in results))
        except Exception as e:
            log(f"  [{i}/{len(COINS)}] {c}: {type(e).__name__}: {e}")
        time.sleep(0.1)
    # Only post when at least one signal fired on the bar that just closed.
    # The loop ticks hourly so it catches every timeframe's close, but posting
    # every hour regardless would be noise -- most hours nothing new happens.
    fast_new = any((results.get((c, t), (None, {}))[1] or {}).get("fresh")
                   for c in COINS for t in FAST_TFS)
    slow_new = any((results.get((c, t), (None, {}))[1] or {}).get("fresh")
                   for c in COINS for t in SLOW_TFS)
    if not _legend_sent:
        post(LEGEND, WH_FAST, dry)
        post(LEGEND, WH_SLOW, dry)
        _legend_sent = True
    if fast_new:
        post(build_report(results, oi, FAST_TFS, "4h \u00b7 5h"), WH_FAST, dry)
    else:
        log("  fast: nothing new this hour -- not posting")
    if slow_new:
        post(build_report(results, oi, SLOW_TFS, "6h \u00b7 12h \u00b7 1d"), WH_SLOW, dry)
    else:
        log("  slow: nothing new this hour -- not posting")
    if not results:
        log("  nothing to report")


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    if "--loop" in sys.argv:
        while True:
            try:
                run_once(dry)
            except Exception as e:
                log(f"cycle failed: {type(e).__name__}: {e}")
            now = time.time()
            nxt = (math.floor(now/3600) + 1)*3600 + 120   # hourly, catches all TFs
            time.sleep(max(60, nxt - now))
    else:
        run_once(dry)
