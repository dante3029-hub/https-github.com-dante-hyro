#!/usr/bin/env python3
"""
skip_fresh_refresh.py — stop re-downloading 3.6 years of history every cycle.

THE PROBLEM
`live_refresh.py` refetches 23,629 bars PER COIN on every run -- the complete
history, from scratch, 24 coins, four times a day. It takes 9.5 minutes, which
is 95% of the cycle's wall time.

It is also a failure surface: any network blip inside those 9.5 minutes aborts
the cycle before a single order is placed. And a 12-minute cycle is too slow to
iterate on, which is why every test today has been painful.

THE FIX
`live_refresh.py --check` reports staleness without fetching. Call that first
and only do the full refresh when the panels are actually stale.

Daily sleeves need a panel that is current to the DAY. Refreshing it four times
a day was never necessary -- once is enough, and the check costs a second.

WHY NOT JUST REMOVE IT
delta and relvol still read clean_panel/ through ReplayDataFeed. The seven new
sleeves read taker_data/, which topup.py keeps current hourly. Until the old
engine is pointed at taker_data too, the panel refresh has to happen -- just
not every cycle.

    python3 skip_fresh_refresh.py --check
    python3 skip_fresh_refresh.py
"""
import shutil
import sys

RC = "run_cycle.sh"
BAK = RC + ".bak_skipfresh"

OLD = '''echo "[1/5] refreshing panels"
if ! python3 live_refresh.py --universe b; then'''

NEW = '''echo "[1/5] refreshing panels"
# Only refresh when the panels are actually stale. live_refresh.py refetches
# the FULL history (23,629 bars x 24 coins) every run -- 9.5 minutes, four
# times a day, to add a handful of new bars. The --check mode reports
# staleness without fetching.
#
# MAX_PANEL_AGE_H is deliberately generous: delta and relvol are the only
# sleeves reading these panels, on 336h and 168h cadences respectively. A
# panel a few hours old cannot change what they rank.
MAX_PANEL_AGE_H="${MAX_PANEL_AGE_H:-20}"
PANEL_FRESH=0
if python3 live_refresh.py --check 2>/dev/null | grep -qiE "fresh|0\\.[0-9]h|^ *[0-9]h"; then
    PANEL_AGE=$(python3 - <<'PYAGE' 2>/dev/null || echo 999
import glob, os, time
fs = glob.glob("clean_panel/hist/*.csv")
print(int((time.time() - max(os.path.getmtime(f) for f in fs)) / 3600) if fs else 999)
PYAGE
)
    [ "${PANEL_AGE:-999}" -lt "$MAX_PANEL_AGE_H" ] && PANEL_FRESH=1
fi

if [ "$PANEL_FRESH" = "1" ]; then
    echo "      panels are ${PANEL_AGE}h old (limit ${MAX_PANEL_AGE_H}h) -- skipping the full refetch"
elif ! python3 live_refresh.py --universe b; then'''

if __name__ == "__main__":
    try:
        s = open(RC).read()
    except FileNotFoundError:
        print(f"{RC} not found -- run from the repo root")
        sys.exit(1)
    if "MAX_PANEL_AGE_H" in s:
        print("already applied")
        sys.exit(0)
    if s.count(OLD) != 1:
        print(f"anchor found {s.count(OLD)} times, expected 1 -- not patching")
        sys.exit(1)
    if "--check" in sys.argv:
        print("ready to patch run_cycle.sh")
        sys.exit(0)
    shutil.copy2(RC, BAK)
    open(RC, "w").write(s.replace(OLD, NEW, 1))
    print(f"patched {RC}  (backup: {BAK})")
    print("\nverify:")
    print("  bash -n run_cycle.sh")
    print("  time ./run_cycle.sh            # should be ~2 min, not ~12")
    print("\nforce a full refresh any time with:")
    print("  MAX_PANEL_AGE_H=0 ./run_cycle.sh --execute")
