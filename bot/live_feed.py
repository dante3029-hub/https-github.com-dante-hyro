"""
bot/live_feed.py — the live data feed the old bot never had.

`LiveDataFeed` in data_feed.py is a deliberate stub that raises. Its reasoning
was correct at the time:

    "live open-interest ingestion (data_loader.py only reads static
     run/oi/*.csv snapshots, most recently updated up to 15 days stale)"
    "live taker-flow ingestion (same staleness issue)"
    "a live 4h/1h kline poller"

All three were about STALENESS, and `topup.py` on an hourly cron fixes all
three: taker_data/ and oi_data/ are now current to the hour, and the klines
they contain ARE the poller's output.

The stub's other objections -- order book depth, cross-venue basis -- were
spec'd for a different design. **No sleeve in this book uses them.** All nine
need only OHLCV, taker delta and open interest.

WHAT THIS DOES NOT DO
It does not fetch. It reads what `topup.py` maintains, and REFUSES if that
data is stale. Separating fetch from read means a dead fetcher shows up as a
hard stop rather than as silently frozen prices -- which is precisely the
408-hour failure the old bot had.
"""
from __future__ import annotations

import datetime as dt
import glob
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_BAR_AGE_HOURS = 8.0       # beyond this the feed refuses to serve
MIN_COINS = 18                # of CORE24; below this the cross-section is thin


class StaleDataError(RuntimeError):
    """Raised instead of returning frozen prices.

    The old bot ran for weeks on 408-hour-old data while every cycle reported
    success. A feed that cannot prove its data is current must fail loudly.
    """


@dataclass
class LiveSnapshot:
    as_of: pd.Timestamp
    coins: List[str]
    weights: Dict[str, Dict[str, float]]     # {sleeve: {coin: weight}}
    stops: Dict[str, float]                  # {coin: stop price}
    data_age_hours: float
    data_source: str = "live"


def _taker_dir() -> str:
    return os.environ.get("HYRO_TAKER_DIR") or (
        "taker_data" if os.path.isdir("taker_data") else "/tmp/hyro/taker_data")


def _oi_dir() -> str:
    return os.environ.get("HYRO_OI_DIR") or (
        "oi_data" if os.path.isdir("oi_data") else "/tmp/hyro/oi_data")


def _last_bar_age_hours(path: str, col: str) -> Optional[float]:
    """Age of the newest BAR, not the file's mtime.

    mtime lies: `git checkout` refreshes it while restoring old content, and
    a fetcher that writes nothing still touches the file. Only the data's own
    timestamp is trustworthy.
    """
    try:
        with open(path) as f:
            header = f.readline().strip().split(",")
        if col not in header:
            return None
        i = header.index(col)
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode(errors="ignore").strip().splitlines()
        best = None
        for line in reversed(tail):
            p = line.split(",")
            if len(p) <= i:
                continue
            try:
                ts = int(float(p[i]))
            except ValueError:
                continue
            if ts > 1_500_000_000_000:
                best = ts
                break
        if best is None:
            return None
        return (dt.datetime.now(dt.timezone.utc).timestamp() * 1000 - best) / 3_600_000
    except Exception:
        return None


def check_freshness(coins: List[str], max_age_h: float = MAX_BAR_AGE_HOURS) -> dict:
    """Ages of every required file. Never raises -- the caller decides."""
    t, o = _taker_dir(), _oi_dir()
    taker, oi, missing = {}, {}, []
    for c in coins:
        pt = f"{t}/{c}_1h.csv"
        if os.path.exists(pt):
            a = _last_bar_age_hours(pt, "open_time")
            if a is not None:
                taker[c] = a
        else:
            missing.append(c)
        po = f"{o}/{c}_oi_1h.csv"
        if os.path.exists(po):
            a = _last_bar_age_hours(po, "timestamp")
            if a is not None:
                oi[c] = a
    return dict(
        taker_ages=taker, oi_ages=oi, missing=missing,
        worst_taker=max(taker.values()) if taker else None,
        worst_oi=max(oi.values()) if oi else None,
        fresh_taker=sum(1 for a in taker.values() if a <= max_age_h),
        fresh_oi=sum(1 for a in oi.values() if a <= max_age_h),
    )


class LiveDataFeed:
    """Reads what topup.py maintains, and refuses when it is stale."""

    def __init__(self, coins: Optional[List[str]] = None,
                 max_age_hours: float = MAX_BAR_AGE_HOURS,
                 min_coins: int = MIN_COINS):
        from bot.book_v2 import CORE24
        self.coins = coins or list(CORE24)
        self.max_age_hours = max_age_hours
        self.min_coins = min_coins
        logger.info("LiveDataFeed: %d coins, refuses above %.0fh bar age",
                    len(self.coins), max_age_hours)

    def get_snapshot(self, tracker_states: dict) -> LiveSnapshot:
        """Current targets for all nine sleeves.

        Raises StaleDataError rather than returning frozen prices.
        `tracker_states` is mutated in place by the event trackers.
        """
        f = check_freshness(self.coins, self.max_age_hours)

        if f["missing"]:
            raise StaleDataError(
                f"taker data missing for {len(f['missing'])} coins: "
                f"{f['missing'][:6]}")
        if f["fresh_taker"] < self.min_coins:
            raise StaleDataError(
                f"only {f['fresh_taker']}/{len(self.coins)} coins have taker "
                f"data under {self.max_age_hours:.0f}h "
                f"(worst {f['worst_taker']:.0f}h). Is topup.py running?")

        # OI starvation does NOT stop the whole book -- it disables one sleeve.
        # But it must be LOUD, because a flat oirank is indistinguishable from
        # a quiet market.
        if f["fresh_oi"] < 12:
            logger.error("OI fresh on only %d coins (worst %.0fh) -- oirank "
                         "will be flat. Check the OI fetcher.",
                         f["fresh_oi"],
                         f["worst_oi"] if f["worst_oi"] is not None else -1)

        from bot.book_v2 import compute_all_weights, protective_stops
        weights = compute_all_weights(tracker_states, core=self.coins)
        stops = protective_stops(tracker_states)

        live = [s for s, w in weights.items() if w]
        logger.info("live snapshot: %d/%d sleeves with positions, "
                    "worst bar age %.1fh",
                    len(live), len(weights), f["worst_taker"] or 0.0)

        return LiveSnapshot(
            as_of=pd.Timestamp.now(tz="UTC"),
            coins=self.coins,
            weights=weights,
            stops=stops,
            data_age_hours=float(f["worst_taker"] or 0.0),
        )
