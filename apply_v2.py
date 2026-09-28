#!/usr/bin/env python3
"""
apply_v2.py — wire the nine-sleeve book into the live bot.

    python3 apply_v2.py --check     show what would change, touch nothing
    python3 apply_v2.py             apply, writing .bak_v2 backups
    python3 apply_v2.py --revert    restore the backups

Patches three files:
    portfolio_layer/portfolio.py   SLEEVE_NAMES -> the nine
    bot/state.py                   tracker_states, flat_cycles, last_cycle_ts
    bot/orchestrator.py            book_v2 call, weight assembly, stops, gate

Every edit is anchored on an exact string. If an anchor is missing the script
STOPS rather than guessing -- a half-applied patch to a live trading bot is
worse than no patch.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

BAK = ".bak_v2"

EDITS = [
    # ── portfolio_layer/portfolio.py ────────────────────────────────────
    ("portfolio_layer/portfolio.py",
     'SLEEVE_NAMES = ("main", "short", "flow", "delta", "relvol", "bos")',
     'SLEEVE_NAMES = (\n'
     '    # cross-sectional, ranked long/short\n'
     '    "delta", "relvol", "skew", "oirank",\n'
     '    # event-driven\n'
     '    "cascade", "sr", "srflip", "pattern", "fvg",\n'
     ')\n'
     '# main/flow never earned a place in the validated book. short is\n'
     '# superseded by the event sleeves. bos scored 0.33 at 4h against a\n'
     '# random-direction control of 0.74 -- the one setting where random beats\n'
     '# it -- and the taker-data fix made it able to fire, so leaving it\n'
     '# enabled is actively harmful.',
     "SLEEVE_NAMES"),

    # ── bot/state.py ────────────────────────────────────────────────────
    ("bot/state.py",
     '    # event-sleeve slot state (see bot/event_sleeves.py)\n'
     '    short_tracker_state: dict = field(default_factory=lambda: {"slots": {}})\n'
     '    bos_tracker_state: dict = field(default_factory=lambda: {"slots": {}})',
     '    # event-sleeve slot state (see bot/event_sleeves.py)\n'
     '    short_tracker_state: dict = field(default_factory=lambda: {"slots": {}})\n'
     '    bos_tracker_state: dict = field(default_factory=lambda: {"slots": {}})\n'
     '\n'
     '    # nine-sleeve book: one slot-tracker state per event sleeve, mutated\n'
     '    # IN PLACE by PriceEventTracker exactly as the old trackers are.\n'
     '    tracker_states: dict = field(default_factory=lambda: {\n'
     '        s: {"slots": {}} for s in ("sr", "srflip", "pattern", "fvg")})\n'
     '\n'
     '    # how many consecutive cycles each sleeve has produced NO weights.\n'
     '    # sleeve_health escalates on this: BOS produced zero weights for 300+\n'
     '    # cycles while the log reported success every time.\n'
     '    flat_cycles: dict = field(default_factory=dict)\n'
     '\n'
     '    # epoch seconds of the last completed cycle. The watchdog cron reads\n'
     '    # this -- a bot that has STOPPED sends no alerts, which looks exactly\n'
     '    # like a healthy quiet period.\n'
     '    last_cycle_ts: float = 0.0',
     "tracker_states"),

    ("bot/state.py",
     '    last_rebalance: Dict[str, Optional[str]] = field(default_factory=lambda: {\n'
     '        "main": None, "flow": None, "delta": None, "relvol": None,\n'
     '        "short": None, "bos": None,\n'
     '    })',
     '    last_rebalance: Dict[str, Optional[str]] = field(default_factory=lambda: {\n'
     '        "delta": None, "relvol": None, "skew": None, "oirank": None,\n'
     '        "cascade": None, "sr": None, "srflip": None, "pattern": None,\n'
     '        "fvg": None,\n'
     '    })',
     "last_rebalance nine sleeves"),

    # ── bot/orchestrator.py: the snapshot ───────────────────────────────
    ("bot/orchestrator.py",
     '        report["data_source"] = snap.data_source',
     '        report["data_source"] = snap.data_source\n'
     '\n'
     '        # ── the new book ──\n'
     '        # delta and relvol still come from the engine above, untouched --\n'
     '        # that path has been running for months and a failure in the new\n'
     '        # book must not take it out. book_v2 supplies the other seven.\n'
     '        #\n'
     '        # LiveDataFeed RAISES on stale data rather than returning weights\n'
     '        # computed on old prices. That is the whole point of it, so the\n'
     '        # except is narrow, not bare.\n'
     '        from bot.live_feed import LiveDataFeed, StaleDataError\n'
     '        try:\n'
     '            _live = LiveDataFeed()\n'
     '            _ls = _live.get_snapshot(self.state.tracker_states)\n'
     '            book_weights = _ls.weights\n'
     '            book_stops = _ls.stops\n'
     '            report["book_data_age_h"] = round(_ls.data_age_hours, 2)\n'
     '        except StaleDataError as e:\n'
     '            report["flags"].append(f"NEW BOOK HALTED -- stale data: {e}")\n'
     '            logger.error("new book halted on stale data: %s", e)\n'
     '            book_weights = {s: {} for s in ("skew", "oirank", "cascade",\n'
     '                                            "sr", "srflip", "pattern", "fvg")}\n'
     '            book_stops = {}\n'
     '        except Exception as e:\n'
     '            # a sleeve that throws is NOT silently flat -- that is how BOS\n'
     '            # hid for 300+ cycles.\n'
     '            report["flags"].append(f"NEW BOOK RAISED: {type(e).__name__}: {e}")\n'
     '            logger.exception("new book raised")\n'
     '            book_weights = {s: {} for s in ("skew", "oirank", "cascade",\n'
     '                                            "sr", "srflip", "pattern", "fvg")}\n'
     '            book_stops = {}',
     "book_v2 snapshot"),

    # ── bot/orchestrator.py: weight assembly ────────────────────────────
    ("bot/orchestrator.py",
     '        raw_weights_this_cycle = {\n'
     '            "main": snap.signal_snapshot.main_weights,\n'
     '            "flow": snap.signal_snapshot.flow_weights,\n'
     '            "delta": snap.signal_snapshot.delta_weights,\n'
     '            "relvol": snap.signal_snapshot.relvol_weights,\n'
     '            "short": snap.short_weights,\n'
     '            "bos": snap.bos_weights,',
     '        raw_weights_this_cycle = {\n'
     '            # kept from the old book, unchanged\n'
     '            "delta": snap.signal_snapshot.delta_weights,\n'
     '            "relvol": snap.signal_snapshot.relvol_weights,\n'
     '            # the new book\n'
     '            **book_weights,',
     "weight assembly"),

    # ── bot/orchestrator.py: protective stops ───────────────────────────
    ("bot/orchestrator.py",
     '        stops_by_symbol = {config.to_bybit_symbol(c): v\n'
     '                           for c, v in list(short_sl.items()) + list(bos_sl.items())}',
     '        # the new book stops EVERY event-sleeve position, not just bos.\n'
     '        # Stops live ON THE EXCHANGE -- if this process dies, the stop\n'
     '        # still works.\n'
     '        stops_by_symbol = {config.to_bybit_symbol(c): v\n'
     '                           for c, v in list(short_sl.items())\n'
     '                           + list(bos_sl.items())\n'
     '                           + list(book_stops.items())}',
     "protective stops"),
    # ── portfolio_layer/portfolio.py: equal-risk multipliers ────────────
    ("portfolio_layer/portfolio.py",
     '    mult = compute_sleeve_multipliers(\n'
     '        hist("main"), hist("short"), hist("flow"),\n'
     '        hist("delta"), hist("relvol"), hist("bos"),\n'
     '        window=window, max_multiplier=max_sleeve_multiplier,\n'
     '    )',
     '    # EQUAL-RISK across N sleeves: scale[k] = median(vol) / vol[k].\n'
     '    #\n'
     '    # The old compute_sleeve_multipliers took six POSITIONAL history\n'
     '    # arrays built around an A/B combo structure those sleeves had. The\n'
     '    # nine-sleeve book does not use that structure -- hourly_sim.simulate,\n'
     '    # which produced every validated number (4.13 Sharpe, 3.7% maxDD),\n'
     '    # uses plain equal-risk. Live sizing must match it, or the live book\n'
     '    # is not the book that was validated.\n'
     '    from portfolio_layer.multipliers_v2 import compute_multipliers_equal_risk\n'
     '    mult = compute_multipliers_equal_risk(\n'
     '        {name: hist(name) for name in SLEEVE_NAMES},\n'
     '        window=window,\n'
     '    )',
     "equal-risk multipliers"),

    ("portfolio_layer/portfolio.py",
     '    for name in ("delta", "relvol", "bos"):',
     '    for name in SLEEVE_NAMES:',
     "history warning over all sleeves"),

]


def read(p):
    with open(p) as f:
        return f.read()


def check(root):
    ok = True
    for path, old, new, label in EDITS:
        f = os.path.join(root, path)
        if not os.path.exists(f):
            print(f"  MISSING FILE  {path}")
            ok = False
            continue
        s = read(f)
        n = s.count(old)
        if n == 1:
            print(f"  ready         {path:<34} {label}")
        elif n == 0:
            if new.split("\n")[0].strip() in s:
                print(f"  already done  {path:<34} {label}")
            else:
                print(f"  ANCHOR NOT FOUND  {path:<30} {label}")
                ok = False
        else:
            print(f"  AMBIGUOUS ({n}x)  {path:<30} {label}")
            ok = False
    return ok


def apply(root):
    if not check(root):
        print("\nnot applying -- fix the anchors above first.")
        print("a half-applied patch to a live trading bot is worse than none.")
        return False
    print()
    done = set()
    for path, old, new, label in EDITS:
        f = os.path.join(root, path)
        s = read(f)
        if old not in s:
            continue
        if path not in done:
            shutil.copy2(f, f + BAK)
            done.add(path)
        with open(f, "w") as fh:
            fh.write(s.replace(old, new, 1))
        print(f"  patched  {path:<34} {label}")
    print(f"\nbackups written: {' '.join(sorted(p + BAK for p in done))}")
    return True


def revert(root):
    n = 0
    for path in sorted({p for p, _, _, _ in EDITS}):
        f = os.path.join(root, path)
        if os.path.exists(f + BAK):
            shutil.move(f + BAK, f)
            print(f"  reverted {path}")
            n += 1
    print(f"{n} file(s) restored" if n else "no backups found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    print("apply_v2 -- wiring the nine-sleeve book\n")
    if a.revert:
        revert(a.root)
    elif a.check:
        sys.exit(0 if check(a.root) else 1)
    else:
        if not apply(a.root):
            sys.exit(1)
        print("\nnext:")
        print("  python3 -c 'import bot.orchestrator, bot.state, portfolio_layer.portfolio'")
        print("  python3 doctor.py")
        print("  python3 bot/burn_in.py --mode offline --cycles 12")
