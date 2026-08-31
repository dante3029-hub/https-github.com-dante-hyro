#!/usr/bin/env python3
"""
Market-data ingestion for the live feed.

READ THIS BEFORE CHANGING PROVIDERS -- the provider split is not arbitrary.

The signals were fit on a panel whose hist CSVs use the BINANCE kline schema:

    ,open_time,open,high,low,close,volume,quote_volume,trades,taker_buy_base

reference_impl.build_matrices() reads column 9, `taker_buy_base`, and
sig_delta() is built directly from it. That column is the entire DELTA sleeve,
which is the strongest single sleeve in the book (Sharpe 1.89 standalone).

Bybit's v5 kline endpoint returns SEVEN fields and nothing else:

    startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover

confirmed against Bybit's own API documentation. There is no taker buy base
volume and no trade count. It is therefore IMPOSSIBLE to feed the DELTA sleeve
from Bybit's public kline endpoint. Substituting Bybit klines would not degrade
the signal, it would silently compute a DIFFERENT signal from the one that was
backtested -- so this module does not offer that option at all.

Correct provider assignment:

    klines + taker_buy_base   -> Binance USDT-M futures (fapi), the schema the
                                 panel was actually built from
    open interest             -> Bybit v5 /v5/market/open-interest (execution
                                 venue's own OI is the right one to use) or
                                 Coinalyze for the aggregated series the panel
                                 currently holds
    taker delta series        -> Coinalyze (run/taker/*.csv came from there;
                                 the `delta` column has no Bybit equivalent)

NOTE ON A REAL, UNRESOLVED RISK: signals are computed from Binance/Coinalyze
data while orders execute on Bybit. That is a defensible and common setup, but
it is a genuine basis/divergence exposure and it was never quantified in any
prior phase. It is not quantified here either. Flagged, not solved.

VALIDATION STATUS: the HTTP layer in this module has NEVER been executed
against a live endpoint. See NETWORK, below. Only the parsers are tested.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger("Ingest")

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")

BINANCE_FAPI = "https://fapi.binance.com"
BYBIT_V5 = "https://api.bybit.com"
COINALYZE = "https://api.coinalyze.net/v1"

# NETWORK
# -------
# Neither endpoint is reachable from the build sandbox:
#   api.bybit.com    -> HTTP 403, body: "The Amazon CloudFront distribution is
#                       configured to block access from your country"
#   fapi.binance.com -> HTTP 451 (unavailable for legal reasons)
#   api.github.com   -> HTTP 200 from the same host, so this is not a general
#                       egress failure; it is a per-destination geo block.
# No API key changes this. It must be run from a permitted jurisdiction.
USER_AGENT = "hyrotrader-bot/1.0"


class IngestError(RuntimeError):
    pass


def _get_json(url: str, params: Optional[dict] = None, timeout: int = 20,
              retries: int = 3, backoff: float = 1.5) -> dict:
    """GET with bounded retries. Raises IngestError rather than returning
    partial or empty data -- a silently empty candle set would be interpreted
    downstream as a flat market."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:200]
            except Exception:                                  # noqa: BLE001
                pass
            last = f"HTTP {e.code}: {body}"
            # 4xx other than 429 will not fix themselves
            if e.code != 429 and 400 <= e.code < 500:
                raise IngestError(f"{url} -> {last}") from e
        except Exception as e:                                 # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(backoff ** attempt)
    raise IngestError(f"{url} failed after {retries} attempts -- {last}")


# ------------------------------------------------------------------ parsers
# These are separated from the HTTP layer precisely so they can be tested
# without network access, which is the only testing possible here.

def parse_binance_klines(raw: list) -> List[dict]:
    """
    Binance USDT-M /fapi/v1/klines returns a list of 12-element arrays:
      0 openTime, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 closeTime,
      7 quoteAssetVolume, 8 numberOfTrades, 9 takerBuyBaseAssetVolume,
      10 takerBuyQuoteAssetVolume, 11 ignore

    Mapped onto the panel's on-disk column order, which is NOT the same order:
      open_time, open, high, low, close, volume, quote_volume, trades,
      taker_buy_base
    """
    out = []
    for k in raw:
        if len(k) < 10:
            raise IngestError(f"malformed kline row, expected >=10 fields, got {len(k)}: {k}")
        out.append({
            "open_time": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "quote_volume": float(k[7]),
            "trades": int(k[8]),
            "taker_buy_base": float(k[9]),
        })
    return out


def parse_bybit_open_interest(payload: dict) -> List[dict]:
    """Bybit v5 /v5/market/open-interest -> result.list[{openInterest, timestamp}]."""
    if payload.get("retCode") not in (0, None):
        raise IngestError(f"bybit error retCode={payload.get('retCode')} "
                          f"retMsg={payload.get('retMsg')!r}")
    rows = (payload.get("result") or {}).get("list") or []
    out = []
    for r in rows:
        try:
            out.append({"open_time": int(r["timestamp"]),
                        "open_interest": float(r["openInterest"])})
        except (KeyError, TypeError, ValueError) as e:
            raise IngestError(f"malformed OI row {r}: {e}") from e
    out.sort(key=lambda x: x["open_time"])
    return out


def parse_bybit_ticker(payload: dict) -> dict:
    if payload.get("retCode") not in (0, None):
        raise IngestError(f"bybit error retCode={payload.get('retCode')} "
                          f"retMsg={payload.get('retMsg')!r}")
    rows = (payload.get("result") or {}).get("list") or []
    if not rows:
        raise IngestError("empty ticker result")
    return rows[0]


# ------------------------------------------------------------------ fetchers
def fetch_binance_klines(symbol: str, interval: str = "1h",
                         limit: int = 1000, start_ms: Optional[int] = None) -> List[dict]:
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
    if start_ms:
        params["startTime"] = start_ms
    return parse_binance_klines(_get_json(f"{BINANCE_FAPI}/fapi/v1/klines", params))


def fetch_bybit_open_interest(symbol: str, interval: str = "1h",
                              limit: int = 200) -> List[dict]:
    params = {"category": "linear", "symbol": symbol,
              "intervalTime": {"1h": "1h", "4h": "4h", "1d": "1d"}.get(interval, "1h"),
              "limit": min(limit, 200)}
    return parse_bybit_open_interest(_get_json(f"{BYBIT_V5}/v5/market/open-interest", params))


# ------------------------------------------------------- incremental CSV append
HIST_COLUMNS = ["", "open_time", "open", "high", "low", "close",
                "volume", "quote_volume", "trades", "taker_buy_base"]


def last_open_time(path: str) -> Optional[int]:
    """Last open_time already on disk, so ingestion can resume incrementally
    instead of re-downloading 884 days every cycle."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if len(row) > 1 and row[1].strip():
                try:
                    last = int(row[1])
                except ValueError:
                    continue
    return last


def append_klines_csv(path: str, rows: List[dict]) -> int:
    """
    Append only strictly-newer bars. Returns the number appended.

    De-duplication is on open_time and is not optional: re-appending an
    existing bar would double-count that hour's volume in the RELVOL baseline
    and corrupt the taker-delta series.
    """
    if not rows:
        return 0
    existing = last_open_time(path)
    new = [r for r in rows if existing is None or r["open_time"] > existing]
    if not new:
        return 0
    new.sort(key=lambda r: r["open_time"])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    idx = 0
    if not write_header:
        with open(path, newline="") as fh:
            idx = max(0, sum(1 for _ in fh) - 1)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(HIST_COLUMNS)
        for r in new:
            w.writerow([idx, r["open_time"], f"{r['open']:.8f}", f"{r['high']:.8f}",
                        f"{r['low']:.8f}", f"{r['close']:.8f}", f"{r['volume']:.8f}",
                        f"{r['quote_volume']:.8f}", r["trades"], f"{r['taker_buy_base']:.8f}"])
            idx += 1
    logger.info(f"{os.path.basename(path)}: appended {len(new)} bar(s)")
    return len(new)


def refresh_universe(coins: List[str], hist_dir: str, interval: str = "1h") -> Dict[str, int]:
    """
    Bring every coin's hist CSV up to the current hour.

    Per-coin failures are collected and returned, never raised -- one delisted
    or renamed symbol must not stop the other 26 from updating. The caller is
    responsible for deciding whether the resulting staleness is tolerable;
    see LiveDataFeed's freshness gate.
    """
    results: Dict[str, int] = {}
    for coin in coins:
        path = os.path.join(hist_dir, f"{coin}_1h.csv")
        try:
            start = last_open_time(path)
            rows = fetch_binance_klines(f"{coin}USDT", interval=interval,
                                        start_ms=(start + 1) if start else None)
            results[coin] = append_klines_csv(path, rows)
        except Exception as e:                                 # noqa: BLE001
            logger.error(f"{coin}: ingestion failed -- {type(e).__name__}: {e}")
            results[coin] = -1
    return results
