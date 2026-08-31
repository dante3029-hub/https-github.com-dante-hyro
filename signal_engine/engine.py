"""
Signal engine orchestrator. Computes target weights/signals for all six
sleeves as of the latest available data. Does NOT place orders, size
positions in dollars, apply the drawdown throttle, or manage the kill switch
-- that is Phase 3 (portfolio/risk layer) and Phase 4 (execution). This
module's only job is: given today's data, what does each sleeve want to hold.

Universe note: universe A (25 coins, Main/Flow/Short) and universe B (27
coins, DELTA/RELVOL/BOS) share 10 coins. This module keeps each sleeve's
weights indexed to ITS OWN universe's coin list -- deliberately, since the
independent-additive sleeve model locked in Phase 0/1 means overlapping coins
combine additively at the portfolio layer, not here.
"""
import datetime as dt
from dataclasses import dataclass, field

from . import data_loader
from . import sleeve_main, sleeve_flow, sleeve_delta, sleeve_relvol
from . import sleeve_short, sleeve_bos


@dataclass
class SignalSnapshot:
    as_of: object
    universe_a_coins: list
    universe_b_coins: list
    main_weights: dict = field(default_factory=dict)     # coin -> weight
    flow_weights: dict = field(default_factory=dict)
    delta_weights: dict = field(default_factory=dict)
    relvol_weights: dict = field(default_factory=dict)
    short_signals: dict = field(default_factory=dict)    # coin -> latest_signal dict (or None)
    bos_signals: dict = field(default_factory=dict)
    main_ok: bool = False
    flow_ok: bool = False
    delta_ok: bool = False
    relvol_ok: bool = False


def compute_snapshot(as_of_date: "dt.date | None" = None) -> SignalSnapshot:
    """
    as_of_date: trims both universes' data to <= this date (for historical /
    parity use). None = use all available data (i.e. "today", once the CSV
    loaders in data_loader.py are swapped for live ingestion in Phase 4).
    """
    # Strategy A disabled (see portfolio_layer/position_sizer.py). Do not load
    # its 240h-stale panel at all -- an unused stale read is still a trap.
    from portfolio_layer.position_sizer import STRATEGY_A_ENABLED
    if STRATEGY_A_ENABLED:
        ua = data_loader.load_universe_a(as_of_date)
    else:
        ua = dict(coins=[], dates=[], PX=None, R=None, FN=None, DN=None, BF=None)
    ub = data_loader.load_universe_b(as_of_date)

    snap = SignalSnapshot(
        as_of=(ua["dates"][-1] if len(ua["dates"]) else ub["dates"][-1]),
        universe_a_coins=ua["coins"],
        universe_b_coins=ub["coins"],
    )

    if STRATEGY_A_ENABLED:
        w, ok = sleeve_main.latest_target_weights(ua["PX"], ua["FN"])
        snap.main_weights = dict(zip(ua["coins"], w))
        snap.main_ok = ok

        w, ok = sleeve_flow.latest_target_weights(ua["DN"], ua["BF"])
        snap.flow_weights = dict(zip(ua["coins"], w))
        snap.flow_ok = ok

    w, ok = sleeve_delta.latest_target_weights(ub["DN"])
    snap.delta_weights = dict(zip(ub["coins"], w))
    snap.delta_ok = ok

    w, ok = sleeve_relvol.latest_target_weights(ub["RV"])
    snap.relvol_weights = dict(zip(ub["coins"], w))
    snap.relvol_ok = ok

    # Short and BOS are per-coin event-driven, not cross-sectional -- loaded
    # coin-by-coin from raw OHLC/OI/taker CSVs (not the dense matrices above).
    for coin in ua["coins"]:
        snap.short_signals[coin] = sleeve_short.latest_signal(coin)
    for coin in ub["coins"]:
        snap.bos_signals[coin] = sleeve_bos.latest_signal(coin)

    return snap


def summarize(snap: SignalSnapshot) -> str:
    lines = [f"Signal snapshot as of {snap.as_of}"]
    for name, weights, ok in [
        ("Main", snap.main_weights, snap.main_ok),
        ("Flow", snap.flow_weights, snap.flow_ok),
        ("DELTA", snap.delta_weights, snap.delta_ok),
        ("RELVOL", snap.relvol_weights, snap.relvol_ok),
    ]:
        longs = sorted([c for c, w in weights.items() if w > 0], key=lambda c: -weights[c])
        shorts = sorted([c for c, w in weights.items() if w < 0], key=lambda c: weights[c])
        lines.append(f"  {name} (ok={ok}): long {longs} / short {shorts}")
    short_trig = [c for c, s in snap.short_signals.items() if s and s["triggered"]]
    bos_trig = [c for c, s in snap.bos_signals.items() if s and s["triggered"]]
    lines.append(f"  Short triggered: {short_trig}")
    lines.append(f"  BOS triggered: {bos_trig}")
    return "\n".join(lines)


if __name__ == "__main__":
    snap = compute_snapshot()
    print(summarize(snap))
