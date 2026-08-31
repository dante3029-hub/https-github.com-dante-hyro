"""
Data loading for both universes. Reuses the exact matrix-building functions
from the reference backtests (not reimplemented) so the resulting PX/R/FN/DN/
RV/BF matrices are guaranteed identical to what was backtested, for the same
input CSVs.

Universe A (Strategy A: Main, Flow, Short) -- 25 coins, source dir 'run/'
Universe B (Strategy B: DELTA, RELVOL, BOS) -- 27 coins, source dir 'clean_panel/'

`as_of_date`, when given, trims the returned date axis (and all matrices) to
rows with date <= as_of_date -- this is what makes these loaders usable both
for historical parity testing (as_of_date = a backtest date) and for live runs
(as_of_date = today, once the CSV loaders below are swapped for live REST/
websocket ingestion in Phase 4). The trimming logic itself does not change
between backtest and live use.
"""
import sys
import os
import datetime as dt

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import reference_impl as _B      # Strategy B: DELTA / RELVOL / BOS reference
import option1_reference as _A   # Strategy A: Main / Flow reference


def _trim(as_of_date, dates, *matrices):
    if as_of_date is None:
        return (dates, *matrices)
    cutoff = [i for i, d in enumerate(dates) if d <= as_of_date]
    if not cutoff:
        raise ValueError(f"no data on or before {as_of_date}")
    last = cutoff[-1] + 1
    return (dates[:last], *(m[:last] for m in matrices))


def load_universe_a(as_of_date: dt.date | None = None):
    """
    Returns dict with keys: coins, dates, PX, R, FN, DN, BF
    (BF = cross-sectional mean of DN, used by sleeve_F as the flow baseline).
    Identical to option1_reference.build_matrices(), optionally trimmed.
    """
    coins, dates, PX, R, FN, DN, BF = _A.build_matrices()
    dates, PX, R, FN, DN = _trim(as_of_date, dates, PX, R, FN, DN)
    BF = BF[:len(dates)]
    return dict(coins=coins, dates=dates, PX=PX, R=R, FN=FN, DN=DN, BF=BF)


def load_universe_b(as_of_date: dt.date | None = None):
    """
    Returns dict with keys: coins, dates, PX, R, FN, DN, RV
    Identical to reference_impl.build_matrices(select_universe()), optionally
    trimmed. `select_universe()` already enforces the fixed-universe /
    liquidity-floor rules (RULE 1 in reference_impl.py) -- reused unchanged.
    """
    coins = _B.select_universe()
    dates, PX, R, FN, DN, RV = _B.build_matrices(coins)
    dates, PX, R, FN, DN, RV = _trim(as_of_date, dates, PX, R, FN, DN, RV)
    return dict(coins=coins, dates=dates, PX=PX, R=R, FN=FN, DN=DN, RV=RV)
