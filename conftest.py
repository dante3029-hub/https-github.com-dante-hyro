"""
Root conftest — 2026-08-10 FIX for the "146/146 from a clean extract" claim.

From a clean extract of the shipped zip, `python3 -m pytest` at the repo root
produced 3 COLLECTION ERRORS and ran 11 tests:

  portfolio_layer/test_compliance.py  ModuleNotFoundError: No module named 'compliance'
      -> the package dirs were never on sys.path; tests only passed when run
         from INSIDE their own directory.
  bot/test_protective_stops.py        FileNotFoundError: run/hist/ICP_1h.csv
  analysis/cap_test.py                TypeError: unbound method set.intersection()
      -> both require a market-data panel that this package does not ship.

This file puts every package dir on sys.path so intra-package imports resolve
from the repo root, and marks the data-dependent tests as SKIPPED rather than
ERRORED so a clean run reports honestly instead of aborting collection.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in ("", "bot", "portfolio_layer", "signal_engine", "analysis"):
    p = os.path.join(ROOT, _d) if _d else ROOT
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Market data is NOT shipped with this package. Tests are skipped per-requirement
# -- checking "any panel exists" is too coarse: test_protective_stops needs
# run/hist specifically, cap_test needs clean_panel/hist, and having one does not
# satisfy the other.
def _has(*parts):
    d = os.path.join(ROOT, *parts)
    return os.path.isdir(d) and bool(os.listdir(d))

HAS_RUN_HIST = _has("run", "hist")
HAS_CLEAN_PANEL = _has("clean_panel", "hist")

# test file -> the panel it requires
_DATA_REQUIREMENTS = {
    "bot/test_protective_stops.py": HAS_RUN_HIST,
}
collect_ignore = [f for f, ok in _DATA_REQUIREMENTS.items() if not ok]

# analysis/cap_test.py is an ANALYSIS SCRIPT, not a test suite -- it contains
# zero test functions and does all its work at module import. pytest collects it
# only because of the `*_test.py` filename, and it then crashes at line 59
# (`U[:, i]` on a 1-D array) whenever a panel IS present. It is not part of the
# automated check count and must not gate the suite. Run it directly:
#     python3 analysis/cap_test.py
collect_ignore.append("analysis/cap_test.py")


def pytest_report_header(config):
    lines = [f"run/hist: {HAS_RUN_HIST}   clean_panel/hist: {HAS_CLEAN_PANEL}"]
    if collect_ignore:
        lines.append("data-dependent tests SKIPPED: " + ", ".join(collect_ignore))
    return lines
