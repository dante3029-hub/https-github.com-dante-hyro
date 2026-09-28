#!/usr/bin/env python3
"""
fix_cadence.py — add the nine sleeves to CADENCE_HOURS.

apply_v2.py missed this: bot/orchestrator.py builds its own CADENCE_HOURS dict
from config constants, and the cadence loop raised `KeyError: 'skew'` on the
first cycle.

    python3 fix_cadence.py --check
    python3 fix_cadence.py
"""
import shutil
import sys

PATH = "bot/orchestrator.py"
BAK = PATH + ".bak_cadence"

OLD = '''CADENCE_HOURS = {
    "main": config.MAIN_CADENCE_HOURS,
    "flow": config.FLOW_CADENCE_HOURS,
    "delta": config.DELTA_CADENCE_HOURS,
    "relvol": config.RELVOL_CADENCE_HOURS,
    "short": config.SHORT_CHECK_HOURS,
    "bos": config.BOS_CHECK_HOURS,
}'''

NEW = '''CADENCE_HOURS = {
    # ---- cross-sectional ----
    "delta": 336,    # was 168. The IC strengthens all the way out to 30 days
                     # (t-stat 0.75 at 1d, 9.98 at 14d, 12.83 at 30d), so
                     # hold=7 was a grid-search artifact. hold 14 gives 1.28 at
                     # 26x turnover against hold 5's 1.18 at 73x -- and at 26x
                     # the never-measured maker fill rate stops being
                     # load-bearing.
    "relvol": 168,   # unchanged -- the research agrees, N=8, hold 7d
    "skew": 1080,    # 45-day hold on a 60-day lookback
    "oirank": 72,    # 3-day hold, rank blended over 3/5/7/10/14d

    # ---- event-driven ----
    # These are BAR SIZES, not rebalance schedules. An open slot can need
    # closing at any time, so PriceEventTracker runs every cycle regardless of
    # what this table says; the value only gates fresh entries.
    "cascade": 24,
    "sr": 6,
    "srflip": 6,
    "pattern": 6,
    "fvg": 12,
}'''

if __name__ == "__main__":
    try:
        s = open(PATH).read()
    except FileNotFoundError:
        print(f"{PATH} not found -- run this from the repo root")
        sys.exit(1)

    if '"skew": 1080' in s:
        print("already applied")
        sys.exit(0)
    n = s.count(OLD)
    if n != 1:
        print(f"anchor found {n} times, expected 1 -- not patching.")
        print("a half-applied patch to a trading bot is worse than none.")
        sys.exit(1)
    if "--check" in sys.argv:
        print("ready to patch CADENCE_HOURS")
        sys.exit(0)
    shutil.copy2(PATH, BAK)
    open(PATH, "w").write(s.replace(OLD, NEW, 1))
    print(f"patched {PATH}  (backup: {BAK})")
    print("\nnext:")
    print("  python3 -c 'import bot.orchestrator; print(len(bot.orchestrator.CADENCE_HOURS))'")
    print("  python3 bot/burn_in.py --mode offline --cycles 12")
