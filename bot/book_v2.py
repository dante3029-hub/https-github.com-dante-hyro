"""
bot/book_v2.py — all nine sleeves, one entry point.

`data_feed.get_snapshot()` calls `compute_all_weights()` and gets back
`{sleeve: {coin: weight}}`, ready for `size_portfolio()`. Everything the new
book needs lives here, so `data_feed.py` changes by about five lines instead
of being rewritten.

WHY A SEPARATE MODULE
The old book's sleeves are computed inside `signal_engine.engine.compute_snapshot`
and handed back as attributes (`snap.delta_weights`). Adding seven more that
way would mean editing the engine, the snapshot dataclass, the feed and the
orchestrator for every sleeve. This keeps the new book in one file that the
feed calls once.

THE FOUR THINGS THAT MUST NOT BE "TIDIED UP"

1. **The OI gate fires at ORDER time, inside the tracker.** Not at detection.
   On SOL 6h there are 60 raw sr breaks: gating at the SIGNAL bar keeps 31,
   gating at the ENTRY bar keeps 23, and the two sets overlap by 12 -- 29%.
   Both are causal. They are different strategies.

2. **Universes are PINNED, never globbed.** `book.py` once built its universe
   from `glob(taker_data/*)`; when the 60-coin fetch landed it silently
   switched from 24 coins to 60 and skew went from +0.88 to -0.16 with no code
   change.

3. **delta runs on the WIDE universe, everything else on CORE24.** delta is
   1.11 on 24 coins and 1.45 on 60; relvol degrades (1.58 -> 0.93) and skew
   breaks entirely. Pairing delta-60 with relvol-24 was worth 1.78 -> 2.07.

4. **cascade returning all-zero weights is NORMAL**, not a failure. It holds a
   position ~3.3% of the time. `sleeve_health` knows a flat cascade is expected
   and a flat delta is not.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EVENT_SLEEVES = ("sr", "srflip", "pattern", "fvg")
XS_SLEEVES = ("skew", "oirank", "cascade")

CORE24 = ["1000PEPE", "1000RATS", "1000SHIB", "AAVE", "ADA", "AVAX", "BCH",
          "BNB", "DOGE", "DOT", "ETH", "FIL", "LDO", "LINK", "LTC", "NEAR",
          "SOL", "SUI", "TRX", "UNI", "WLD", "XLM", "XRP", "ZEC"]


def _taker_dir() -> str:
    return os.environ.get("HYRO_TAKER_DIR") or (
        "taker_data" if os.path.isdir("taker_data") else "/tmp/hyro/taker_data")


def _oi_dir() -> str:
    return os.environ.get("HYRO_OI_DIR") or (
        "oi_data" if os.path.isdir("oi_data") else "/tmp/hyro/oi_data")


def wide_universe() -> List[str]:
    """Every coin with taker data. delta alone uses this."""
    import glob
    d = _taker_dir()
    return sorted(os.path.basename(f).replace("_1h.csv", "")
                  for f in glob.glob(f"{d}/*_1h.csv"))


# ── daily matrices, built once per cycle ─────────────────────────────────
def _daily_panel(coins: List[str]):
    """Returns (coins_present, close, returns, volume, delta, open_interest)
    as numpy arrays aligned on a common daily index."""
    import hourly_sim as H
    frames, oi = {}, {}
    for c in coins:
        p = f"{_taker_dir()}/{c}_1h.csv"
        if not os.path.exists(p):
            continue
        try:
            b = H.bars(c, 24)
        except Exception:
            continue
        if len(b) < 400:
            continue
        frames[c] = b
        q = f"{_oi_dir()}/{c}_oi_1h.csv"
        if os.path.exists(q):
            try:
                x = pd.read_csv(q)
                x["ts"] = pd.to_datetime(x["timestamp"], unit="ms")
                oi[c] = x.set_index("ts").sort_index()["open_interest"].resample("1D").last()
            except Exception:
                pass
    if not frames:
        return [], None, None, None, None, None
    idx = None
    for v in frames.values():
        idx = v.index if idx is None else idx.union(v.index)
    cs = sorted(frames)
    PX = pd.DataFrame({c: frames[c]["close"].reindex(idx) for c in cs})
    VOL = pd.DataFrame({c: frames[c]["volume"].reindex(idx) for c in cs})
    DN = pd.DataFrame({c: frames[c]["delta"].reindex(idx) for c in cs})
    # FORWARD-FILL OI, with a limit. OI and price are topped up by separate
    # fetchers, so OI can lag price by an hour or two. Without this, a single
    # lagging bar leaves the last row all-NaN, oi_scores finds nothing, and
    # oirank silently returns FLAT -- the same shape as every other silent
    # failure on this project. The 3-day limit stops it quietly trading on
    # stale positioning if the OI fetcher dies.
    OI = pd.DataFrame({c: (oi[c].reindex(idx).ffill(limit=3) if c in oi
                           else pd.Series(np.nan, index=idx)) for c in cs})
    return cs, PX.to_numpy(), PX.pct_change().to_numpy(), VOL.to_numpy(), \
        DN.to_numpy(), OI.to_numpy()


# ── the cross-sectional sleeves ──────────────────────────────────────────
def _xs_weights(coins, R, OI) -> Dict[str, Dict[str, float]]:
    from signal_engine.sleeve_skew import latest_target_weights as skew_w
    from signal_engine.sleeve_oirank import latest_target_weights as oi_w
    from signal_engine.sleeve_cascade import latest_target_weights as cas_w
    out: Dict[str, Dict[str, float]] = {}
    for name, fn, mat in (("skew", skew_w, R), ("oirank", oi_w, OI),
                          ("cascade", cas_w, R)):
        try:
            w, ok = fn(mat)
            if not ok and name != "cascade":
                # cascade legitimately returns flat; the others do not
                logger.warning("sleeve %s could not rank -- treated as flat", name)
            out[name] = {c: float(x) for c, x in zip(coins, w) if abs(x) > 1e-12}
        except Exception:
            logger.exception("sleeve %s raised", name)
            out[name] = {}          # the health check sees the flat sleeve
    return out


# ── the event sleeves ────────────────────────────────────────────────────
def _event_weights(coins: List[str], tracker_states: dict) -> Dict[str, Dict[str, float]]:
    import hourly_sim as H
    from bot.price_event_tracker import PriceEventTracker
    from signal_engine.sleeve_price_events import PARAMS

    bar_cache: Dict[tuple, Optional[pd.DataFrame]] = {}
    oi_cache: Dict[tuple, Optional[pd.Series]] = {}

    def bars_for(coin: str, tf: int):
        key = (coin, tf)
        if key not in bar_cache:
            try:
                b = H.bars(coin, tf)
                bar_cache[key] = b if len(b) >= 700 else None
            except Exception:
                bar_cache[key] = None
        return bar_cache[key]

    def oi_expanding(coin: str, ts, tf: int) -> bool:
        key = (coin, tf)
        if key not in oi_cache:
            try:
                oi_cache[key] = H.oi_up(coin, tf)
            except Exception:
                oi_cache[key] = None
        ou = oi_cache[key]
        if ou is None:
            return False            # cannot verify OI -> do not trade
        v = ou.reindex([ts], method="ffill")
        return bool(len(v) and bool(v.iloc[0]))

    out: Dict[str, Dict[str, float]] = {}
    for s in EVENT_SLEEVES:
        tf = PARAMS[s]["tf"]
        st = tracker_states.setdefault(s, {"slots": {}})
        try:
            tr = PriceEventTracker(
                s, st,
                bars_fn=lambda c, _tf=tf: bars_for(c, _tf),
                oi_fn=((lambda c, ts, _tf=tf: oi_expanding(c, ts, _tf))
                       if PARAMS[s]["needs_oi"] else None))
            out[s] = tr.update(coins)
        except Exception:
            logger.exception("event sleeve %s raised", s)
            out[s] = {}
    return out


def compute_all_weights(tracker_states: dict,
                        core: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Every sleeve except delta and relvol, which the existing engine already
    computes. Returns {sleeve: {coin: weight}}.

    tracker_states is mutated IN PLACE, exactly as ShortSleeveTracker expects,
    so the caller's state file picks the changes up without reassignment.
    """
    core = core or CORE24
    cs, PX, R, VOL, DN, OI = _daily_panel(core)
    if not cs:
        logger.error("no daily panel could be built -- every sleeve flat")
        return {s: {} for s in XS_SLEEVES + EVENT_SLEEVES}
    # oirank is the one sleeve that can be starved by a lagging feed. Say so
    # loudly rather than letting it look like a quiet market.
    if OI is not None and OI.shape[0]:
        fin = int(np.isfinite(OI[-1]).sum())
        if fin < 12:
            logger.error("OI has only %d/%d coins on the latest bar -- oirank "
                         "will be flat. Check the OI fetcher.", fin, len(cs))
    weights = _xs_weights(cs, R, OI)
    weights.update(_event_weights(core, tracker_states))
    return weights


def protective_stops(tracker_states: dict) -> Dict[str, float]:
    """{coin: stop price} across every event sleeve, for
    execution.sync_protective_stops(). Stops belong ON THE EXCHANGE -- if the
    bot dies, the stop still works."""
    from bot.price_event_tracker import PriceEventTracker
    out: Dict[str, float] = {}
    for s in EVENT_SLEEVES:
        st = tracker_states.get(s)
        if not st:
            continue
        for coin, slot in st.get("slots", {}).items():
            stop = slot.get("stop")
            if stop is not None:
                out[coin] = float(stop)   # one position per coin per sleeve
    return out
