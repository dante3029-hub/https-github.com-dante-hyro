#!/usr/bin/env python3
"""Diagnose OI files: timestamp column, units, range, staleness.

The first version of this script read column 0 unconditionally. These files
have `idx,timestamp,open_interest`, so column 0 is a ROW NUMBER. Before the
column-alignment repair, the mis-appended rows happened to put a real
timestamp in column 0, so max() found it and the output looked correct by
accident. After the repair every row is properly 3-column and it reported
1970 for everything.

The diagnostic had the same class of bug it was written to catch. It now reads
the header and uses the named column.
"""
import csv
import datetime as dt
import glob
import os
import sys

MIN_MS, MAX_MS = 1_500_000_000_000, 4_000_000_000_000


def main(d="oi_data"):
    files = sorted(glob.glob(f"{d}/*_oi_1h.csv"))
    if not files:
        print(f"no files in {d}/")
        return 1
    now_ms = dt.datetime.now(dt.timezone.utc).timestamp() * 1000
    print(f"{'file':<24}{'rows':>7}  {'first':<17}{'last':<17}{'age h':>8}")
    stale, broken = [], []
    for p in files:
        try:
            with open(p) as f:
                rdr = csv.reader(f)
                header = next(rdr)
                if "timestamp" not in header:
                    print(f"{os.path.basename(p):<24}  NO timestamp COLUMN {header}")
                    broken.append(os.path.basename(p))
                    continue
                i = header.index("timestamp")
                ts = []
                for r in rdr:
                    if len(r) <= i:
                        continue
                    try:
                        v = int(float(r[i]))
                    except ValueError:
                        continue
                    if MIN_MS <= v <= MAX_MS:
                        ts.append(v)
        except Exception as e:
            print(f"{os.path.basename(p):<24}  UNREADABLE {e}")
            broken.append(os.path.basename(p))
            continue
        if not ts:
            print(f"{os.path.basename(p):<24}{0:>7}  no valid timestamps")
            broken.append(os.path.basename(p))
            continue
        lo, hi = min(ts), max(ts)
        age = (now_ms - hi) / 3_600_000
        f1 = dt.datetime.fromtimestamp(lo / 1000, dt.timezone.utc)
        f2 = dt.datetime.fromtimestamp(hi / 1000, dt.timezone.utc)
        flag = "  STALE" if age > 24 else ""
        print(f"{os.path.basename(p):<24}{len(ts):>7}  {f1:%Y-%m-%d %H:%M}  "
              f"{f2:%Y-%m-%d %H:%M}{age:>8.1f}{flag}")
        if age > 24:
            stale.append((os.path.basename(p), age))
    print()
    if broken:
        print(f"BROKEN: {broken}")
    if stale:
        print(f"STALE (>24h): {len(stale)} files, worst {max(s[1] for s in stale):.0f}h")
    if not broken and not stale:
        print("all OI files healthy")
    return 2 if broken else (1 if stale else 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "oi_data"))
