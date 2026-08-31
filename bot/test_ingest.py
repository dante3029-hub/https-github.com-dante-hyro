#!/usr/bin/env python3
"""
Tests for bot/ingest.py.

SCOPE LIMIT, STATED UP FRONT: these test the PARSERS and the CSV append logic
only. The HTTP layer is NOT tested and has never been executed against a live
endpoint, because both providers are geo-blocked from this machine
(api.bybit.com -> 403 CloudFront country block, fapi.binance.com -> 451).
Passing this file does NOT mean ingestion works. It means that IF a response
arrives with the documented shape, it is parsed and persisted correctly.
"""
from __future__ import annotations

import os
import sys
import tempfile

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from bot.ingest import (IngestError, append_klines_csv, last_open_time,
                        parse_binance_klines, parse_bybit_open_interest)

R = []


def check(name, ok, detail=""):
    R.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# A real Binance USDT-M /fapi/v1/klines row, 12 fields, values as strings.
BINANCE_ROW = [
    1672531200000, "16537.50", "16540.90", "16504.00", "16527.00", "5381.399",
    1672534799999, "88923087.83210", 31529, "2541.791", "42018274.11", "0",
]


def test_binance_kline_field_mapping():
    print("\ntest_binance_kline_field_mapping")
    out = parse_binance_klines([BINANCE_ROW])[0]
    check("open_time from field 0", out["open_time"] == 1672531200000)
    check("close from field 4, NOT field 6", out["close"] == 16527.00, f"{out['close']}")
    check("volume (base) from field 5", out["volume"] == 5381.399)
    check("quote_volume from field 7, NOT field 6",
          out["quote_volume"] == 88923087.83210, f"{out['quote_volume']}")
    check("trades from field 8", out["trades"] == 31529)
    check("taker_buy_base from field 9 -- the DELTA sleeve's only input",
          out["taker_buy_base"] == 2541.791, f"{out['taker_buy_base']}")
    check("taker_buy_base <= total volume (sanity)",
          out["taker_buy_base"] <= out["volume"])


def test_binance_matches_the_on_disk_panel_row():
    """The first row of the real panel file must round-trip through the parser
    unchanged -- this is what proves the field mapping is the panel's mapping
    and not merely self-consistent."""
    print("\ntest_binance_matches_the_on_disk_panel_row")
    out = parse_binance_klines([BINANCE_ROW])[0]
    # from clean_panel/hist/BTC_1h.csv line 2:
    # ,1672531200000,16537.50,16540.90,16504.00,16527.00,5381.399,88923087.83210,31529,2541.791
    panel = dict(open_time=1672531200000, open=16537.50, high=16540.90, low=16504.00,
                 close=16527.00, volume=5381.399, quote_volume=88923087.83210,
                 trades=31529, taker_buy_base=2541.791)
    check("parsed row equals the real BTC_1h.csv row exactly", out == panel,
          f"diff={ {k: (out[k], panel[k]) for k in panel if out[k] != panel[k]} }")


def test_malformed_kline_raises():
    print("\ntest_malformed_kline_raises")
    try:
        parse_binance_klines([[1, "2", "3"]])
        check("short row rejected", False, "no exception raised")
    except IngestError as e:
        check("short row rejected loudly, not silently truncated", True, str(e)[:60])


def test_bybit_oi_parsing():
    print("\ntest_bybit_oi_parsing")
    payload = {"retCode": 0, "retMsg": "OK", "result": {"list": [
        {"openInterest": "461134.0", "timestamp": "1672534800000"},
        {"openInterest": "460000.5", "timestamp": "1672531200000"}]}}
    out = parse_bybit_open_interest(payload)
    check("both rows parsed", len(out) == 2)
    check("rows sorted ascending by time", out[0]["open_time"] < out[1]["open_time"])
    check("string numerics coerced", out[0]["open_interest"] == 460000.5)


def test_bybit_error_code_is_not_swallowed():
    print("\ntest_bybit_error_code_is_not_swallowed")
    try:
        parse_bybit_open_interest({"retCode": 10001, "retMsg": "params error", "result": {}})
        check("non-zero retCode raises", False, "no exception")
    except IngestError as e:
        check("non-zero retCode raises instead of returning []", True, str(e)[:60])


def test_append_is_incremental_and_deduplicates():
    print("\ntest_append_is_incremental_and_deduplicates")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hist", "BTC_1h.csv")
        rows = parse_binance_klines([BINANCE_ROW])
        n1 = append_klines_csv(path, rows)
        check("first append writes the bar", n1 == 1)
        check("last_open_time reads it back", last_open_time(path) == 1672531200000)

        n2 = append_klines_csv(path, rows)
        check("re-appending the SAME bar writes nothing (no double-count)", n2 == 0)

        newer = [dict(rows[0], open_time=1672534800000)]
        n3 = append_klines_csv(path, newer)
        check("a strictly newer bar is appended", n3 == 1)
        check("last_open_time advances", last_open_time(path) == 1672534800000)

        older = [dict(rows[0], open_time=1672527600000)]
        n4 = append_klines_csv(path, older)
        check("an OLDER bar is refused (would corrupt ordering)", n4 == 0)

        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        check("file has header + exactly 2 data rows", len(lines) == 3, f"{len(lines)} lines")
        check("header matches the panel schema",
              lines[0].startswith(",open_time,open,high,low,close,volume,quote_volume,trades,taker_buy_base"),
              lines[0])


def test_last_open_time_on_missing_file():
    print("\ntest_last_open_time_on_missing_file")
    check("missing file returns None, does not raise",
          last_open_time("/nonexistent/nope.csv") is None)


if __name__ == "__main__":
    for fn in [test_binance_kline_field_mapping,
               test_binance_matches_the_on_disk_panel_row,
               test_malformed_kline_raises,
               test_bybit_oi_parsing,
               test_bybit_error_code_is_not_swallowed,
               test_append_is_incremental_and_deduplicates,
               test_last_open_time_on_missing_file]:
        fn()
    n = sum(1 for _, ok in R if ok)
    print("\n" + "=" * 64)
    print(f"PASSED {n} / {len(R)}")
    for name, ok in R:
        if not ok:
            print("  FAILED:", name)
    print("NOTE: parsers only. The HTTP layer is untested (providers geo-blocked).")
    sys.exit(0 if n == len(R) else 1)
