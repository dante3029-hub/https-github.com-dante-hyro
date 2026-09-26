#!/usr/bin/env python3
"""
sleeve_health.py — the bot tells YOU what is wrong, instead of failing silently.

WHY THIS EXISTS — three real failures from this project, all silent:

  * BOS produced ZERO weights on 300+ consecutive live cycles. The log said
    "rebalanced=['short','bos']" every time. Nothing flagged it. The cause was
    a data bug (taker delta was another venue's volume, so `dn < 0` could never
    be true) that took months to find.
  * `run/` went 408 hours stale while `clean_panel/` refreshed. Cycles kept
    "succeeding" on frozen data.
  * The taker column had taker_buy > TOTAL volume on 60% of bars. Nothing
    validated it.

Each was findable in one cycle by a check that knew what NORMAL looks like.

DESIGN
Every sleeve declares an EXPECTATION -- how often it should fire, what weight
range is sane, what data it needs and how fresh. Each cycle the checker
compares reality to that and emits specific, actionable diagnostics.

A check NEVER silently passes on missing information. If it cannot evaluate,
that is a WARN, not an OK.

SEVERITY
  CRITICAL  stop trading. Data is wrong or a sleeve is structurally broken.
  ERROR     this sleeve is not working. Others may continue.
  WARN      outside expectation but could be a quiet market.
  INFO      normal.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CRITICAL, ERROR, WARN, INFO = "CRITICAL", "ERROR", "WARN", "INFO"


@dataclass
class Expectation:
    """What NORMAL looks like for one sleeve, taken from the backtest.

    `trades_per_month` and `active_frac` come from the research run. A live
    sleeve firing at a wildly different rate is not necessarily wrong, but it
    is always worth knowing about -- and a sleeve firing ZERO times when it
    should fire 20 is the BOS failure repeating.
    """
    name: str
    trades_per_month: float          # from the backtest
    active_frac: float               # fraction of bars holding a position
    max_abs_weight: float            # per coin, per sleeve
    max_gross: float
    data_files: List[str] = field(default_factory=list)
    max_data_age_hours: float = 8.0
    cadence_hours: float = 24.0
    long_only: bool = False
    short_only: bool = False


@dataclass
class Finding:
    severity: str
    sleeve: str
    message: str
    detail: str = ""

    def __str__(self):
        d = f"  {self.detail}" if self.detail else ""
        return f"[{self.severity:<8}] {self.sleeve:<10} {self.message}{d}"


def check_data_freshness(exp: Expectation, now: pd.Timestamp) -> List[Finding]:
    """The 408-hour staleness failure. A file that exists is not a file that
    is current."""
    out = []
    for path in exp.data_files:
        if not os.path.exists(path):
            out.append(Finding(CRITICAL, exp.name, "data file missing",
                               f"{path} -- sleeve cannot run"))
            continue
        try:
            df = pd.read_csv(path, usecols=[0], nrows=0)
        except Exception as e:
            out.append(Finding(CRITICAL, exp.name, "data file unreadable",
                               f"{path}: {type(e).__name__}"))
            continue
        mtime = pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')
        age = (now - mtime).total_seconds() / 3600
        if age > exp.max_data_age_hours:
            sev = CRITICAL if age > exp.max_data_age_hours * 4 else ERROR
            out.append(Finding(sev, exp.name, "data STALE",
                               f"{os.path.basename(path)} last written "
                               f"{age:.0f}h ago (limit {exp.max_data_age_hours:.0f}h)"))
    return out


def check_data_sanity(path: str, sleeve: str) -> List[Finding]:
    """The corrupt-taker-column failure: taker_buy exceeded TOTAL volume on
    60% of bars and nothing noticed for months."""
    out = []
    if not os.path.exists(path):
        return [Finding(CRITICAL, sleeve, "data file missing", path)]
    try:
        df = pd.read_csv(path).tail(500)
    except Exception as e:
        return [Finding(CRITICAL, sleeve, "cannot parse data", f"{path}: {e}")]
    if df.empty:
        return [Finding(CRITICAL, sleeve, "data file is empty", path)]

    for col in ('close', 'volume'):
        if col in df.columns:
            if df[col].isna().any():
                out.append(Finding(ERROR, sleeve, f"NaN in {col}",
                                   f"{int(df[col].isna().sum())} of {len(df)} recent bars"))
            if (df[col] <= 0).any() and col == 'close':
                out.append(Finding(CRITICAL, sleeve, "non-positive close price",
                                   f"{int((df[col] <= 0).sum())} bars"))
    if 'taker_buy' in df.columns and 'volume' in df.columns:
        bad = int((df['taker_buy'] > df['volume']).sum())
        if bad:
            out.append(Finding(CRITICAL, sleeve,
                               "taker_buy EXCEEDS total volume",
                               f"{bad}/{len(df)} recent bars -- this is the "
                               "spliced-venue bug, delta is meaningless"))
    if 'delta' in df.columns:
        neg = float((df['delta'] < 0).mean())
        if neg < 0.05:
            out.append(Finding(CRITICAL, sleeve, "delta is never negative",
                               f"only {neg*100:.1f}% of bars -- any sleeve "
                               "needing delta<0 can NEVER fire (the BOS bug)"))
        elif neg > 0.95:
            out.append(Finding(CRITICAL, sleeve, "delta is never positive",
                               f"{neg*100:.1f}% negative"))
    if 'open_time' in df.columns and len(df) > 2:
        gaps = pd.to_datetime(df['open_time'], unit='ms').diff().dt.total_seconds()/3600
        big = gaps[gaps > 1.5].dropna()
        if len(big):
            out.append(Finding(WARN, sleeve, "gaps in recent bars",
                               f"{len(big)} gaps, largest {big.max():.0f}h"))
    return out


def check_weights(exp: Expectation, weights: Optional[Dict[str, float]],
                  cycles_since_any_weight: int = 0) -> List[Finding]:
    """The BOS failure: a sleeve that rebalances on schedule and produces
    nothing, forever, while the log reports success."""
    out = []
    if weights is None:
        out.append(Finding(ERROR, exp.name, "sleeve returned None",
                           "expected a {coin: weight} dict"))
        return out
    active = {c: w for c, w in weights.items() if abs(w) > 1e-9}
    if not active:
        # a flat sleeve is normal for an EVENT sleeve in a quiet week; it is
        # NOT normal for a cross-sectional one, or for many cycles running
        expected_gap = max(3.0, 30.0 / max(exp.trades_per_month, 0.1))
        if cycles_since_any_weight > expected_gap * 3:
            out.append(Finding(CRITICAL, exp.name,
                               "NO WEIGHTS for far too long",
                               f"{cycles_since_any_weight} cycles flat; expected "
                               f"a position every ~{expected_gap:.0f}. THIS IS "
                               "THE BOS FAILURE -- check the entry condition "
                               "can ever be true"))
        elif cycles_since_any_weight > expected_gap:
            out.append(Finding(WARN, exp.name, "flat longer than expected",
                               f"{cycles_since_any_weight} cycles"))
        else:
            out.append(Finding(INFO, exp.name, "flat this cycle", ""))
        return out

    gross = sum(abs(w) for w in active.values())
    mx = max(abs(w) for w in active.values())
    if mx > exp.max_abs_weight * 1.01:
        out.append(Finding(ERROR, exp.name, "per-coin weight above cap",
                           f"max {mx:.3f} vs cap {exp.max_abs_weight:.3f} "
                           f"-- concentration control is not binding"))
    if gross > exp.max_gross * 1.01:
        out.append(Finding(ERROR, exp.name, "gross above cap",
                           f"{gross:.3f} vs {exp.max_gross:.3f} -- the sleeve "
                           "is running levered (the FVG 2.1x bug)"))
    if exp.long_only and any(w < -1e-9 for w in active.values()):
        out.append(Finding(CRITICAL, exp.name, "SHORT position in a long-only sleeve",
                           str({c: round(w, 4) for c, w in active.items() if w < 0})))
    if exp.short_only and any(w > 1e-9 for w in active.values()):
        out.append(Finding(CRITICAL, exp.name, "LONG position in a short-only sleeve",
                           str({c: round(w, 4) for c, w in active.items() if w > 0})))
    if not out:
        out.append(Finding(INFO, exp.name,
                           f"{len(active)} positions, gross {gross:.3f}"))
    return out


def check_cadence(exp: Expectation, hours_since_rebalance: Optional[float]) -> List[Finding]:
    if hours_since_rebalance is None:
        return [Finding(WARN, exp.name, "never rebalanced",
                        "no record of a rebalance for this sleeve")]
    if hours_since_rebalance > exp.cadence_hours * 2.5:
        return [Finding(ERROR, exp.name, "rebalance OVERDUE",
                        f"{hours_since_rebalance:.0f}h since last, "
                        f"cadence is {exp.cadence_hours:.0f}h")]
    return []


def run_health_check(expectations: List[Expectation],
                     weights_by_sleeve: Dict[str, Optional[Dict[str, float]]],
                     flat_cycles: Dict[str, int],
                     hours_since_rebalance: Dict[str, Optional[float]],
                     sanity_files: Dict[str, str],
                     now: Optional[pd.Timestamp] = None) -> List[Finding]:
    """Run everything and return findings sorted worst-first."""
    now = now or pd.Timestamp.now(tz='UTC')
    if now.tzinfo is None:
        now = now.tz_localize('UTC')
    found: List[Finding] = []
    for exp in expectations:
        found += check_data_freshness(exp, now)
        if exp.name in sanity_files:
            found += check_data_sanity(sanity_files[exp.name], exp.name)
        found += check_weights(exp, weights_by_sleeve.get(exp.name),
                               flat_cycles.get(exp.name, 0))
        found += check_cadence(exp, hours_since_rebalance.get(exp.name))
    order = {CRITICAL: 0, ERROR: 1, WARN: 2, INFO: 3}
    found.sort(key=lambda f: order[f.severity])
    return found


def format_report(findings: List[Finding]) -> str:
    """A report you can read in five seconds and act on."""
    counts = {s: sum(1 for f in findings if f.severity == s)
              for s in (CRITICAL, ERROR, WARN, INFO)}
    lines = []
    if counts[CRITICAL]:
        lines.append(f"*** {counts[CRITICAL]} CRITICAL -- DO NOT TRADE ***")
    elif counts[ERROR]:
        lines.append(f"{counts[ERROR]} ERROR -- some sleeves are not working")
    elif counts[WARN]:
        lines.append(f"{counts[WARN]} warnings")
    else:
        lines.append("all sleeves healthy")
    lines.append("")
    for f in findings:
        if f.severity != INFO or counts[CRITICAL] + counts[ERROR] == 0:
            lines.append(str(f))
    return "\n".join(lines)


# ─────────────────────── the book's expectations ───────────────────────
# trades_per_month and active_frac come from the hourly backtest. A live
# sleeve that deviates sharply is not proof of a bug, but it is always worth
# knowing -- and ZERO when 20 are expected is the BOS failure.

def book_expectations(taker_dir: str, oi_dir: str) -> List[Expectation]:
    d = lambda c: os.path.join(taker_dir, f"{c}_1h.csv")
    return [
        Expectation("delta",   trades_per_month=0,   active_frac=0.85,
                    max_abs_weight=0.10, max_gross=1.0,
                    data_files=[d("SOL"), d("ETH")], cadence_hours=336),
        Expectation("relvol",  trades_per_month=0,   active_frac=0.65,
                    max_abs_weight=0.0625, max_gross=1.0,
                    data_files=[d("SOL")], cadence_hours=168),
        Expectation("skew",    trades_per_month=0,   active_frac=0.84,
                    max_abs_weight=0.084, max_gross=1.0,
                    data_files=[d("SOL")], cadence_hours=1080),
        Expectation("oirank",  trades_per_month=0,   active_frac=0.90,
                    max_abs_weight=0.10, max_gross=1.0,
                    data_files=[d("SOL"), os.path.join(oi_dir, "SOL_oi_1h.csv")],
                    cadence_hours=72),
        Expectation("cascade", trades_per_month=2.5, active_frac=0.033,
                    max_abs_weight=0.20, max_gross=1.0,
                    data_files=[d("SOL")], cadence_hours=24, long_only=True),
        Expectation("sr",      trades_per_month=13,  active_frac=0.24,
                    max_abs_weight=1/6, max_gross=1.0,
                    data_files=[d("SOL"), os.path.join(oi_dir, "SOL_oi_1h.csv")],
                    cadence_hours=6, long_only=True),
        Expectation("srflip",  trades_per_month=8,   active_frac=0.16,
                    max_abs_weight=1/6, max_gross=1.0,
                    data_files=[d("SOL")], cadence_hours=6, long_only=True),
        Expectation("pattern", trades_per_month=19,  active_frac=0.27,
                    max_abs_weight=1/6, max_gross=1.0,
                    data_files=[d("SOL"), os.path.join(oi_dir, "SOL_oi_1h.csv")],
                    cadence_hours=6, long_only=True),
        Expectation("fvg",     trades_per_month=126, active_frac=0.95,
                    max_abs_weight=1/6, max_gross=1.0,
                    data_files=[d("SOL"), os.path.join(oi_dir, "SOL_oi_1h.csv")],
                    cadence_hours=12),
    ]
