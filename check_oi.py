#!/usr/bin/env python3
"""Diagnose the OI files: timestamp units, ranges, duplicates."""
import glob, os, csv, datetime as dt

print(f"{'file':<22}{'rows':>7}{'first':>22}{'last':>22}{'unit':>8}")
bad_unit, bad_age = [], []
now_ms = dt.datetime.now(dt.timezone.utc).timestamp() * 1000
for p in sorted(glob.glob("oi_data/*_oi_1h.csv")):
    try:
        rows = list(csv.reader(open(p)))[1:]
        ts = [int(float(r[0])) for r in rows if r and r[0]]
    except Exception as e:
        print(f"{os.path.basename(p):<22} UNREADABLE {e}")
        continue
    if not ts:
        print(f"{os.path.basename(p):<22}      0  EMPTY")
        continue
    lo, hi = min(ts), max(ts)
    # milliseconds since epoch for a 2020s date is ~1.7e12; seconds is ~1.7e9
    unit = "ms" if hi > 1e11 else "SEC"
    if unit == "SEC":
        bad_unit.append(os.path.basename(p))
    f = dt.datetime.fromtimestamp(lo/(1000 if unit=="ms" else 1), dt.timezone.utc)
    l = dt.datetime.fromtimestamp(hi/(1000 if unit=="ms" else 1), dt.timezone.utc)
    age_h = (now_ms - hi*(1 if unit=="ms" else 1000))/3600000
    if age_h > 24: bad_age.append((os.path.basename(p), age_h))
    print(f"{os.path.basename(p):<22}{len(ts):>7}{f:%Y-%m-%d %H:%M:>22}"
          f"{l:%Y-%m-%d %H:%M:>22}{unit:>8}")
print()
if bad_unit:
    print(f"SECONDS not milliseconds: {bad_unit}")
if bad_age:
    print(f"stale (>24h): {len(bad_age)} files, worst {max(b[1] for b in bad_age):.0f}h")
