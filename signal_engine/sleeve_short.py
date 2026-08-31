"""
Strategy A - Short sleeve. Bear-regime market-structure short, 4h bars,
event-driven. Universe: 25 coins (run/hist).

RECONSTRUCTED FROM PROSE, NOT VERIFIED: sleeve_S_reconstructed.py itself
states that the original backtest file (short_engine_backtest.py) was never
found among the uploaded files, and several parameters (swing-lookback=20,
Ichimoku 9/26/52, delta z-window=48/threshold=-1.0, ATR period=14, per-slot
sizing 1/6) are explicitly [ASSUMED], not sourced from OPTION-1-FINAL-SPEC.pdf.
This live wrapper reuses `build_signals()` from sleeve_S_reconstructed.py
UNCHANGED, so it inherits both the honest reconstruction and its caveats --
it is not more trustworthy than the backtest it's built on, and this file
does not attempt to resolve or paper over that uncertainty.
"""
from sleeve_S_reconstructed import build_signals, CHAND_MULT, MAX_CONCURRENT


def latest_signal(coin: str):
    """
    Returns None if no data / not enough history for this coin, otherwise a
    dict: {triggered: bool, entry_ref: float, initial_stop: float, atr: float}

    triggered = short_ok at the LAST available (most recently confirmed) 4h
    bar for this coin -- i.e. regime (bear) AND trigger (swing-low break or
    Ichimoku cloud breakdown) AND confirm (delta z < -1.0 AND OI falling 3
    bars) all satisfied, exactly as build_signals() computes it.

    entry_ref is that bar's close (build_signals is causal: signal known at
    close of t, tradeable from open of t+1 per its own docstring) -- for a
    live system, treat this as the reference price, not a fill price.

    initial_stop mirrors backtest_sleeve_S()'s entry-time chandelier seed:
    at entry, `lowest_close_high` is initialized to that bar's high, so the
    initial chandelier stop = high + CHAND_MULT * atr. This is NOT a fixed
    stop -- backtest_sleeve_S() tightens it every subsequent bar as new highs
    print (`lowest_close_high = min(lowest_close_high, row['high'])`), so
    Phase 3/4's live position tracker must replicate that trailing update,
    not just place this initial value once.

    Max concurrent shorts (6) and per-slot sizing (1/6 of sleeve capital) are
    portfolio-level concerns handled in Phase 3/4, not here.
    """
    df = build_signals(coin)
    if df is None or len(df) == 0:
        return None
    row = df.iloc[-1]
    triggered = bool(row['short_ok'])
    return dict(
        triggered=triggered,
        entry_ref=float(row['close']),
        initial_stop=float(row['high'] + CHAND_MULT * row['atr']),
        atr=float(row['atr']),
        timestamp=df.index[-1],
    )
