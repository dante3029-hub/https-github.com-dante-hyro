#!/usr/bin/env python3
"""
repair_oi.py — fix OI files where appended rows have the wrong column count.

THE BUG
The original files were written by pandas WITH an index column:

    idx,timestamp,open_interest
    0,1674644400000,2902275.4

topup.py appended rows WITHOUT it:

    1790582400000,6945944.50000000

So every appended row is shifted one column left. A loader reading the
`timestamp` column gets the OPEN INTEREST value, which parses as 1970.

WHY IT MATTERS
sr, pattern and fvg all gate on open interest at the entry bar. With the
timestamp garbage, `reindex(..., method='ffill')` finds nothing and the gate
either rejects every signal or accepts every signal. Neither is visible
without looking -- the same shape as the BOS failure, where an impossible
condition silently produced zero trades for 300+ cycles.

WHAT THIS DOES
  * detects the header and each row's column count
  * rewrites every row in the file's OWN format
  * drops rows whose timestamp is not a plausible millisecond epoch
  * de-duplicates on timestamp, keeping the last value
  * sorts ascending
  * writes atomically via a .tmp file, so an interrupted run cannot corrupt
    what is already there

    python3 repair_oi.py --check     report only
    python3 repair_oi.py             repair in place
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import datetime as dt

MIN_MS = 1_500_000_000_000      # 2017-07 — anything earlier is not real OI data
MAX_MS = 4_000_000_000_000      # 2096


def analyse(path: str) -> dict:
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return dict(path=path, empty=True)
    header = rows[0]
    has_idx = len(header) == 3 and header[0] in ("idx", "")
    ts_col = 1 if has_idx else 0
    widths = {}
    good = bad = 0
    for r in rows[1:]:
        widths[len(r)] = widths.get(len(r), 0) + 1
        col = ts_col if len(r) == len(header) else 0
        try:
            ts = int(float(r[col]))
            if MIN_MS <= ts <= MAX_MS:
                good += 1
            else:
                bad += 1
        except (ValueError, IndexError):
            bad += 1
    return dict(path=path, header=header, has_idx=has_idx, ts_col=ts_col,
                widths=widths, good=good, bad=bad, n=len(rows) - 1)


def repair(path: str) -> dict:
    a = analyse(path)
    if a.get("empty"):
        return dict(path=path, action="skipped (empty)")
    with open(path) as f:
        rows = list(csv.reader(f))
    header, has_idx = a["header"], a["has_idx"]
    expect = len(header)
    seen = {}
    dropped = 0
    for r in rows[1:]:
        if not r:
            continue
        # a short row is a mis-appended one: timestamp and value only
        if len(r) == expect:
            ts_raw, oi_raw = r[a["ts_col"]], r[a["ts_col"] + 1]
        elif len(r) == expect - 1:
            ts_raw, oi_raw = r[0], r[1]
        else:
            dropped += 1
            continue
        try:
            ts = int(float(ts_raw))
            oi = float(oi_raw)
        except (ValueError, IndexError):
            dropped += 1
            continue
        if not (MIN_MS <= ts <= MAX_MS):
            dropped += 1
            continue
        seen[ts] = oi              # last value for a timestamp wins
    out = sorted(seen.items())
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, (ts, oi) in enumerate(out):
            w.writerow([i, ts, f"{oi:.8f}"] if has_idx else [ts, f"{oi:.8f}"])
    os.replace(tmp, path)
    return dict(path=path, action="repaired", before=a["n"], after=len(out),
                dropped=dropped,
                last=dt.datetime.fromtimestamp(out[-1][0] / 1000, dt.timezone.utc)
                if out else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dir", default="oi_data")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dir}/*_oi_1h.csv"))
    if not files:
        print(f"no files in {a.dir}/")
        sys.exit(1)

    if a.check:
        print(f"{'file':<24}{'rows':>7}{'good':>8}{'BAD':>8}  widths")
        tot_bad = 0
        for p in files:
            d = analyse(p)
            if d.get("empty"):
                print(f"{os.path.basename(p):<24}  EMPTY")
                continue
            tot_bad += d["bad"]
            flag = "  <-- MIXED" if len(d["widths"]) > 1 else ""
            print(f"{os.path.basename(p):<24}{d['n']:>7}{d['good']:>8}{d['bad']:>8}"
                  f"  {dict(d['widths'])}{flag}")
        print(f"\n{tot_bad} bad rows in total")
        sys.exit(0)

    print(f"repairing {len(files)} files")
    tot_dropped = 0
    for p in files:
        r = repair(p)
        tot_dropped += r.get("dropped", 0)
        if r.get("action") == "repaired":
            print(f"  {os.path.basename(p):<24}{r['before']:>7} -> {r['after']:<7}"
                  f" dropped {r['dropped']:<5} last {r['last']:%Y-%m-%d %H:%M}"
                  if r["last"] else f"  {os.path.basename(p)}  EMPTY AFTER REPAIR")
    print(f"\ndone: {tot_dropped} bad rows dropped")
