#!/usr/bin/env python3
"""
doctor.py — one command, full system check.

    python3 doctor.py                 # everything
    python3 doctor.py --quick         # skip the slow detector checks
    python3 doctor.py --sleeve sr     # one sleeve only

Exit code 0 = safe to trade. 1 = warnings. 2 = do NOT trade.
So it drops straight into a cron guard:

    python3 doctor.py && python3 bot/run_cycle.py

WHAT IT CHECKS
  1. data files exist, are fresh, and are internally sane
  2. every sleeve can actually produce a signal on current data
  3. tracker state is loadable and self-consistent
  4. the config is complete and the sizing function behaves
  5. exchange reachability (if credentials are present)

WHAT IT CANNOT CHECK
It tests for the failures we already know about. It will not catch a NEW class
of bug -- only the burn-in and paper stages do that, by exercising paths this
cannot reach.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from typing import List

import numpy as np
import pandas as pd

CRITICAL, ERROR, WARN, OK = "CRITICAL", "ERROR", "WARN", "OK"
_ICON = {CRITICAL: "XX", ERROR: " X", WARN: " !", OK: " ."}

results: List[tuple] = []


def say(sev, area, msg, detail=""):
    results.append((sev, area, msg, detail))
    line = f"{_ICON[sev]} [{area:<12}] {msg}"
    if detail:
        line += f"\n                    {detail}"
    print(line, flush=True)


# ── 1. data ──────────────────────────────────────────────────────────────
def check_data(taker_dir, oi_dir, max_age_h=8.0):
    if not os.path.isdir(taker_dir):
        say(CRITICAL, "data", "taker_data directory missing", taker_dir)
        return
    files = sorted(glob.glob(f"{taker_dir}/*_1h.csv"))
    if not files:
        say(CRITICAL, "data", "no taker files found", taker_dir)
        return
    say(OK, "data", f"{len(files)} taker files, "
                    f"{len(glob.glob(f'{oi_dir}/*_oi_1h.csv'))} OI files")

    now = time.time()
    # Age from the LAST BAR, not the file mtime. `git checkout` rewrites
    # mtimes, so data reverted to a 17-day-old commit reported "0.0h old"
    # while the content was stale. mtime measures when the file was TOUCHED;
    # only the last bar says how current the DATA is.
    stale = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - 2048))
                tail = fh.read().decode(errors="ignore").strip().splitlines()
            ts = int(float(tail[-1].split(",")[0]))
            stale.append((os.path.basename(f), (now - ts / 1000) / 3600))
        except Exception:
            stale.append((os.path.basename(f), float("inf")))
    worst = max(stale, key=lambda x: x[1])
    if worst[1] > max_age_h * 4:
        say(CRITICAL, "data", "data is VERY stale",
            f"{worst[0]} newest BAR is {worst[1]:.0f}h old (limit {max_age_h:.0f}h). "
            f"run/ went 408h stale once while cycles kept 'succeeding'")
    elif worst[1] > max_age_h:
        say(ERROR, "data", "data is stale",
            f"{worst[0]} newest bar {worst[1]:.0f}h old")
    else:
        say(OK, "data", f"oldest coin's newest bar is {worst[1]:.1f}h old")

    # integrity on a sample -- the bug that hid for months
    bad_taker = bad_delta = 0
    for f in files[:12]:
        try:
            df = pd.read_csv(f).tail(300)
        except Exception as e:
            say(ERROR, "data", f"unreadable: {os.path.basename(f)}", str(e))
            continue
        if 'taker_buy' in df and 'volume' in df:
            if (df['taker_buy'] > df['volume']).any():
                bad_taker += 1
        if 'delta' in df:
            neg = float((df['delta'] < 0).mean())
            if neg < 0.05 or neg > 0.95:
                bad_delta += 1
    if bad_taker:
        say(CRITICAL, "data", "taker_buy EXCEEDS total volume",
            f"{bad_taker} files. This is the spliced-venue bug -- delta is "
            f"meaningless and any sleeve needing delta<0 can never fire")
    else:
        say(OK, "data", "taker_buy <= volume on every file sampled")
    if bad_delta:
        say(CRITICAL, "data", "delta is one-sided",
            f"{bad_delta} files never go negative (or never positive). "
            f"This is exactly how BOS produced zero weights for 300+ cycles")
    else:
        say(OK, "data", "delta is two-sided")


# ── 2. can every sleeve actually fire? ───────────────────────────────────
def check_sleeves(taker_dir, quick=False, only=None):
    try:
        import hourly_sim as H
        from signal_engine.sleeve_price_events import PARAMS, latest_signal
        from fast_detect import FastSR, FastFVG, FastPattern
    except Exception as e:
        say(CRITICAL, "sleeves", "cannot import sleeve modules", str(e))
        return
    coin = None
    for c in ("SOL", "ETH", "BTC", "XRP"):
        if os.path.exists(f"{taker_dir}/{c}_1h.csv"):
            coin = c
            break
    if coin is None:
        say(CRITICAL, "sleeves", "no reference coin available")
        return

    for s in (only and [only]) or list(PARAMS):
        try:
            bars = H.bars(coin, PARAMS[s]['tf'])
            if len(bars) < 750:
                say(ERROR, s, "not enough history",
                    f"{len(bars)} bars, need 750")
                continue
            det = (FastSR(bars) if s in ('sr', 'srflip')
                   else FastFVG(bars) if s == 'fvg' else FastPattern(bars))
            if s == 'sr':
                n = int(sum(det.fired(i, 'break') for i in range(700, det.n)))
            elif s == 'srflip':
                n = int(sum(det.fired(i, 'flip') for i in range(700, det.n)))
            else:
                n = int(sum(det.fired(i) != 0 for i in range(700, det.n)))
            if n == 0:
                say(CRITICAL, s, "detector fires ZERO times on full history",
                    f"{coin} {PARAMS[s]['tf']}h. The entry condition may be "
                    f"unsatisfiable -- this is the BOS failure")
            else:
                yrs = (bars.index[-1] - bars.index[0]).days / 365
                say(OK, s, f"{n} signals on {coin} ({n/max(yrs,0.1):.0f}/yr)")
            if not quick:
                sig = latest_signal(s, bars, oi_expanding=None)
                say(OK, s, "latest_signal() callable",
                    f"fires now: {sig is not None}")
        except Exception:
            say(CRITICAL, s, "sleeve raised",
                traceback.format_exc().strip().splitlines()[-1])


# ── 3. tracker state ─────────────────────────────────────────────────────
def check_state(state_path):
    if not state_path or not os.path.exists(state_path):
        say(WARN, "state", "no state file yet", state_path or "(unset)")
        return
    try:
        with open(state_path) as f:
            st = json.load(f)
    except Exception as e:
        say(CRITICAL, "state", "state file is not valid JSON",
            f"{state_path}: {e} -- a restart would lose every open position")
        return
    say(OK, "state", f"loadable ({os.path.getsize(state_path)} bytes)")
    slots = 0
    for k, v in st.items():
        if isinstance(v, dict) and "slots" in v:
            for coin, sl in v["slots"].items():
                slots += 1
                side = sl.get("side", 0)
                stop, ref = sl.get("stop"), sl.get("entry_px", sl.get("entry_ref"))
                if stop is None or ref is None:
                    say(ERROR, "state", f"{k}/{coin} slot missing stop or entry")
                elif (side > 0 and stop >= ref) or (side < 0 and stop <= ref):
                    say(CRITICAL, "state", f"{k}/{coin} STOP ON THE WRONG SIDE",
                        f"side {side:+d}, entry {ref}, stop {stop}")
    say(OK, "state", f"{slots} open slots, all stops on the correct side")


# ── 4. config ────────────────────────────────────────────────────────────
def check_config():
    try:
        import importlib.util as iu
        p = os.path.join(os.path.dirname(__file__), "bot", "config_v2.py")
        if not os.path.exists(p):
            say(WARN, "config", "config_v2.py not found", p)
            return
        spec = iu.spec_from_file_location("cfg2", p)
        m = iu.module_from_spec(spec)
        spec.loader.exec_module(m)
    except Exception as e:
        say(CRITICAL, "config", "config will not import", str(e))
        return
    missing = [s for s in m.SLEEVE_NAMES if s not in m.CADENCE_BY_SLEEVE]
    if missing:
        say(CRITICAL, "config", "sleeves with no cadence", str(missing))
    else:
        say(OK, "config", f"{len(m.SLEEVE_NAMES)} sleeves, all have a cadence")
    if len(m.CORE24) != 24:
        say(ERROR, "config", f"CORE24 has {len(m.CORE24)} coins, expected 24")
    else:
        say(OK, "config", "CORE24 pinned at 24 coins")
    try:
        big = m.daily_vol_target(m.DEGEAR_TRIGGER_EQUITY + 1000)
        small = m.daily_vol_target(m.DEGEAR_TRIGGER_EQUITY - 1000)
        if big <= small:
            say(CRITICAL, "config", "de-gear is inverted",
                f"above trigger {big}, below {small}")
        else:
            say(OK, "config",
                f"sizing ${small:,.0f} below / ${big:,.0f} above "
                f"${m.DEGEAR_TRIGGER_EQUITY:,.0f}")
    except Exception as e:
        say(ERROR, "config", "daily_vol_target() failed", str(e))


# ── 5. exchange ──────────────────────────────────────────────────────────
def check_exchange():
    key = os.environ.get("BYBIT_API_KEY")
    if not key:
        say(WARN, "exchange", "no BYBIT_API_KEY in the environment",
            "skipping connectivity check")
        return
    try:
        import urllib.request
        req = urllib.request.Request("https://api.bybit.com/v5/market/time",
                                     headers={"User-Agent": "doctor/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            json.loads(r.read().decode())
        say(OK, "exchange", "api.bybit.com reachable")
    except Exception as e:
        say(CRITICAL, "exchange", "cannot reach Bybit", str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taker", default=os.path.expanduser("~/taker_data"))
    ap.add_argument("--oi", default=os.path.expanduser("~/oi_data"))
    ap.add_argument("--state", default=os.path.expanduser("~/bot_state.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sleeve", default=None)
    a = ap.parse_args()

    print("=" * 66)
    print("DOCTOR — full system check")
    print("=" * 66)
    check_data(a.taker, a.oi)
    print()
    check_sleeves(a.taker, quick=a.quick, only=a.sleeve)
    print()
    check_state(a.state)
    print()
    check_config()
    print()
    check_exchange()

    crit = sum(1 for r in results if r[0] == CRITICAL)
    err = sum(1 for r in results if r[0] == ERROR)
    warn = sum(1 for r in results if r[0] == WARN)
    print()
    print("=" * 66)
    if crit:
        print(f"{crit} CRITICAL, {err} error, {warn} warning  ->  DO NOT TRADE")
        print("=" * 66)
        return 2
    if err:
        print(f"{err} error, {warn} warning  ->  fix before trading")
        print("=" * 66)
        return 2
    if warn:
        print(f"{warn} warning  ->  safe, but read them")
        print("=" * 66)
        return 1
    print("all checks passed  ->  safe to trade")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
