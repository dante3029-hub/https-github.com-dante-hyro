"""
Abstract DataFeed interface plus two concrete implementations:

  ReplayDataFeed -- wraps signal_engine.engine.compute_snapshot(as_of_date)
                    + bot.sleeve_history.compute_sleeve_histories(as_of_date)
                    + bot.event_sleeves trackers, all against the EXISTING
                    STATIC CSVs in run/hist, run/oi, run/taker, clean_panel.
                    Fully functional today for historical replay / dry-run /
                    smoke-testing. This is what bot/main.py uses when
                    --dry-run is set (the default).

  LiveDataFeed  -- STUB. Raises NotImplementedError with a message citing
                    exactly what's missing. DO NOT attempt to make this
                    "work" by silently falling back to stale CSVs or
                    fabricating a data source -- that would let the bot run
                    live on data confirmed to be 4-15 days stale
                    (see data_loader.py's own docstring + AUDIT_FINDINGS.md
                    + "Data Acquisition Spec" doc, all previously confirmed
                    in this session). The correct fix is building the real
                    ingestion pipeline (order book, cross-venue basis, OI,
                    taker-flow) that spec called for -- not faking this class.
"""
import os
import sys
import datetime as dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from signal_engine.engine import compute_snapshot
from bot import clock
from bot.sleeve_history import compute_sleeve_histories
from bot.event_sleeves import ShortSleeveTracker, BOSSleeveTracker

logger = logging.getLogger("DataFeed")


@dataclass
class MarketSnapshot:
    as_of_date: dt.date
    signal_snapshot: object            # signal_engine.engine.SignalSnapshot
    sleeve_histories: dict             # bot.sleeve_history.compute_sleeve_histories() output
    short_weights: dict                # {coin: weight} from ShortSleeveTracker
    bos_weights: dict                  # {coin: weight} from BOSSleeveTracker
    data_source: str                   # "replay" or "live" -- stamped so downstream code/logs
                                        # never have to guess where numbers came from


class DataFeed(ABC):
    @abstractmethod
    def get_snapshot(self, as_of_date: Optional[dt.date],
                      short_tracker_state: dict, bos_tracker_state: dict) -> MarketSnapshot:
        ...


class ReplayDataFeed(DataFeed):
    """Historical replay against the static CSVs already validated in
    Phase 2/3. Safe for dry-run, smoke-testing, and backtesting the wiring
    itself -- NOT a source of live market data."""

    def __init__(self):
        logger.info("ReplayDataFeed initialized -- reads static CSVs under run/ and clean_panel/, "
                     "NOT live market data.")

    def get_snapshot(self, as_of_date: Optional[dt.date],
                      short_tracker_state: dict, bos_tracker_state: dict) -> MarketSnapshot:
        snap = compute_snapshot(as_of_date)
        hist = compute_sleeve_histories(as_of_date)

        short_tracker = ShortSleeveTracker(short_tracker_state)
        short_weights = short_tracker.update(snap.universe_a_coins)

        bos_tracker = BOSSleeveTracker(bos_tracker_state)
        bos_weights = bos_tracker.update(snap.universe_b_coins)

        return MarketSnapshot(
            # clock.trading_day() not date.today() -- date.today() reads the
            # HOST's local timezone, which would roll the trading day over at
            # the wrong moment on any non-UTC server. See bot/clock.py.
            as_of_date=as_of_date or clock.trading_day(),
            signal_snapshot=snap,
            sleeve_histories=hist,
            short_weights=short_weights,
            bos_weights=bos_weights,
            data_source="replay",
        )


class LiveDataFeed(DataFeed):
    """
    STUB -- intentionally non-functional.

    CONFIRMED MISSING (from this session's audit, not a guess):
      - Order book depth / liquidity screening (spec'd in "Data Acquisition
        Spec -- Order Book & Cross-Venue", never implemented)
      - Cross-venue basis feed
      - Live open-interest ingestion (data_loader.py only reads static
        run/oi/*.csv snapshots, most recently updated up to 15 days stale
        as observed directly: as_of_a=2026-07-25, as_of_b=2026-08-05,
        as_of_s=2026-07-25 when queried on/around 2026-08-09)
      - Live taker-flow ingestion (same staleness issue, run/taker/*.csv)
      - A live 4h/1h kline poller feeding sleeve_bos.py's swing-pivot logic
        (currently reads static CSVs via reference_impl._load_4h)

    Instantiating this class is fine (for wiring tests); calling
    get_snapshot() raises immediately. This is a deliberate hard stop, not
    an oversight -- see BUILD_PLAN.md Phase 4 findings.
    """

    def __init__(self):
        logger.warning("LiveDataFeed instantiated -- get_snapshot() WILL raise. "
                        "Live ingestion pipeline was never built (see class docstring).")

    def get_snapshot(self, as_of_date: Optional[dt.date],
                      short_tracker_state: dict, bos_tracker_state: dict) -> MarketSnapshot:
        raise NotImplementedError(
            "LiveDataFeed.get_snapshot() is not implemented. This bot cannot safely trade "
            "live today: order book, cross-venue basis, live OI, and live taker-flow "
            "ingestion were never built (only the 'Data Acquisition Spec' document exists, "
            "no code). Confirmed 4-15 day staleness in the current static CSVs "
            "(run/oi, run/taker) makes them unsafe as a live data source -- this is not "
            "a rate-limit or auth problem you can retry past. Use ReplayDataFeed for "
            "dry-run/backtesting; build the real ingestion pipeline before attempting live "
            "trading."
        )
