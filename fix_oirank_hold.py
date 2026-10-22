#!/usr/bin/env python3
"""
fix_oirank_hold.py -- move oirank from a 3-day to a 5-day hold.

WHY: hold_sweep.py on 1,357 days of the live panel.

    hold   net Sharpe   fees/yr   GROSS signal
      3d      0.90       15.5%       2.71      <- live
      5d      1.06        9.3%       2.14
      7d      0.83        6.6%       1.60
     10d      0.56        4.7%       1.10

The gross signal decays monotonically -- a real signal with a short
half-life, not a fitted parameter. Fees fall as 1/hold. 5 days is the
crossing point, and the shape check reports a plateau, not a spike.

The Sharpe gain (0.90 -> 1.06) is INSIDE the ~0.64 standard error on a
3.7-year Sharpe, so it is not the justification on its own. The fee saving
is: 15.5% -> 9.3% of notional per year, certain rather than estimated.

WHY A SCRIPT RATHER THAN AN EDIT: the hold is declared in TWO places.

    signal_engine/sleeve_oirank.py   CADENCE_HOURS = 3 * 24   (changed in git)
    bot/orchestrator.py              "oirank": 72             (patched here)

The orchestrator builds its own dict and does not read the sleeve modules,
so changing one alone silently does nothing. That duplication is the same
shape as several bugs this project has already paid for, and it should be
removed -- the orchestrator should import each sleeve's own CADENCE_HOURS.
This script does the narrow fix; the duplication is noted, not solved.

IDEMPOTENT. Reports clearly if the anchor is missing rather than exiting 0
having done nothing.

USAGE
    python3 fix_oirank_hold.py --check
    python3 fix_oirank_hold.py --apply
"""
from __future__ import annotations

import re
import sys

TARGET = "bot/orchestrator.py"
OLD = '"oirank": 72'
NEW = '"oirank": 120'


def main() -> int:
    try:
        with open(TARGET) as f:
            s = f.read()
    except FileNotFoundError:
        print(f"  FAIL {TARGET} not found -- run from the repo root")
        return 2

    if NEW in s:
        print(f"  already applied: {TARGET} has {NEW}")
        return 0

    if OLD not in s:
        m = re.search(r'"oirank"\s*:\s*(\d+)', s)
        if m:
            print(f"  FAIL expected {OLD!r} but found "
                  f'"oirank": {m.group(1)} -- the cadence has been changed')
            print(f"       elsewhere. NOT patching blindly. Fix by hand to 120.")
        else:
            print(f"  FAIL no \"oirank\" entry in {TARGET}'s CADENCE_HOURS.")
            print(f"       The orchestrator may not know about oirank at all,")
            print(f"       which would be a bigger problem than the hold.")
        return 2

    n = s.count(OLD)
    if n != 1:
        print(f"  FAIL {OLD!r} appears {n} times -- refusing to guess which")
        return 2

    with open(TARGET, "w") as f:
        f.write(s.replace(OLD, NEW, 1))
    print(f"  patched {TARGET}: {OLD} -> {NEW}  (3-day -> 5-day hold)")
    print()
    print("  VERIFY on the next cycle: the report's 'next action' for oirank")
    print("  should read ~5d, not ~3d. If it still says 3d, the live")
    print("  orchestrator is not the file that was patched.")
    return 0


if __name__ == "__main__":
    if "--apply" not in sys.argv:
        print(__doc__)
        print("  dry run -- re-run with --apply")
        raise SystemExit(0)
    raise SystemExit(main())
