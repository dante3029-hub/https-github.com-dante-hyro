#!/usr/bin/env python3
"""
report.py — the bot reports back like a colleague, not a log file.

`alerts.py` shouts when something breaks. This is the other half: after every
cycle it says what it did, why, and what it is watching -- so you can read one
message and know the state of the book without opening a terminal.

    from report import CycleReport
    r = CycleReport(equity=..., day_start_equity=...)
    r.opened("fvg", "SOL", -1, 142.50, 148.20, "gap down, OI expanding")
    r.closed("sr", "LINK", 18.40, "15 bars elapsed", pnl_pct=+2.1)
    r.sleeve("delta", n=10, gross=0.31, next_due_h=312)
    r.note("oirank flat 3 cycles (expected roughly every 3)")
    r.post(webhook)

WHY IT IS SHAPED LIKE THIS

**Numbers without reasons are noise.** "opened SOL short" tells you nothing you
can act on; "opened SOL short, gap down with OI expanding, stop 4% away" tells
you whether the bot is doing what you think it does. Every entry carries its
trigger.

**Risk is stated in the firm's terms, every time.** Not "drawdown 2.1%" but
"$23,775 above the $180,000 floor" and "4.2% of the daily limit used". Those
are the two numbers that end the evaluation, so they appear whether or not
anything happened.

**Silence is reserved for alerts.** This posts every cycle by design -- it is a
status report, not an exception. That is why it goes to its OWN channel: mixing
routine reports into the alert channel trains you to scroll past both.

Discord caps a message at 2,000 characters, so long sections truncate with a
count rather than being dropped silently.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional


def _post(webhook: str, content: str) -> bool:
    """Discord rejects urllib's default User-Agent with a 403 -- curl works,
    the script does not, and the error says nothing useful."""
    if not webhook:
        return False
    try:
        req = urllib.request.Request(
            webhook, data=json.dumps({"content": content[:1990]}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "HyroBot (report, 1.0)"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status in (200, 204)
    except Exception:
        return False


def _px(v: float) -> str:
    """Format a price at a precision that suits its magnitude.

    A fixed 4dp renders 1000PEPE at $0.0000121 as "0.0000" -- the report would
    show a position with no usable price. Crypto spans eight orders of
    magnitude, so the precision has to follow the number.
    """
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:,.2f}"
    if a >= 1:
        return f"{v:,.4f}"
    if a >= 0.01:
        return f"{v:.5f}"
    if a >= 0.0001:
        return f"{v:.7f}"
    return f"{v:.9f}"


@dataclass
class CycleReport:
    equity: float
    day_start_equity: float
    floor: float = 180_000.0
    daily_limit: float = -10_000.0
    kill_switch: float = -5_000.0
    vol_target: float = 3_000.0
    as_of: Optional[dt.datetime] = None

    opens: List[str] = field(default_factory=list)
    closes: List[str] = field(default_factory=list)
    rebalances: List[str] = field(default_factory=list)
    sleeves: List[tuple] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    halted: bool = False
    halt_reason: str = ""

    # ── things that happened ─────────────────────────────────────────────
    def opened(self, sleeve: str, coin: str, side: int, price: float,
               stop: float, why: str = ""):
        d = "LONG " if side > 0 else "SHORT"
        risk = abs(stop - price) / price * 100 if price else 0
        line = f"`{sleeve:<7}` {d} **{coin}** @ {_px(price)} · stop {_px(stop)} ({risk:.1f}%)"
        if why:
            line += f" — {why}"
        self.opens.append(line)

    def closed(self, sleeve: str, coin: str, price: float, why: str,
               pnl_pct: Optional[float] = None):
        line = f"`{sleeve:<7}` closed **{coin}** @ {_px(price)} — {why}"
        if pnl_pct is not None:
            line += f" · {pnl_pct:+.1f}%"
        self.closes.append(line)

    def rebalanced(self, sleeve: str, n_long: int, n_short: int, n_changed: int):
        self.rebalances.append(
            f"`{sleeve:<7}` rebalanced · {n_long} long / {n_short} short · "
            f"{n_changed} position(s) changed")

    def sleeve(self, name: str, n: int, gross: float,
               next_due_h: Optional[float] = None):
        self.sleeves.append((name, n, gross, next_due_h))

    def note(self, text: str):
        self.notes.append(text)

    def problem(self, text: str):
        self.problems.append(text)

    # ── the message ──────────────────────────────────────────────────────
    def render(self) -> str:
        ts = (self.as_of or dt.datetime.now(dt.timezone.utc))
        pnl = self.equity - self.day_start_equity
        buffer_ = self.equity - self.floor
        buffer_pct = buffer_ / self.floor * 100 if self.floor else 0
        limit_used = abs(pnl) / abs(self.daily_limit) * 100 if pnl < 0 else 0

        L = []
        if self.halted:
            L.append(f"🔴 **CYCLE HALTED** — {self.halt_reason}")
            L.append("**No orders were placed.**")
            L.append("")

        L.append(f"**HyroTrader** · {ts:%d %b %H:%M} UTC")
        L.append(f"equity **${self.equity:,.0f}** · today **{pnl:+,.0f}** · "
                 f"sizing ${self.vol_target:,.0f}/day")
        L.append("")

        if self.opens or self.closes or self.rebalances:
            L.append("**what happened**")
            for x in self.opens[:6]:
                L.append("· " + x)
            if len(self.opens) > 6:
                L.append(f"· …and {len(self.opens)-6} more opens")
            for x in self.closes[:6]:
                L.append("· " + x)
            if len(self.closes) > 6:
                L.append(f"· …and {len(self.closes)-6} more closes")
            for x in self.rebalances:
                L.append("· " + x)
        else:
            L.append("**what happened** · nothing — no sleeve was due and no "
                     "event fired")
        L.append("")

        if self.sleeves:
            L.append("**the book**")
            tot_n = sum(s[1] for s in self.sleeves)
            tot_g = sum(s[2] for s in self.sleeves)
            for name, n, gross, due in sorted(self.sleeves, key=lambda s: -s[2]):
                if n == 0 and (due is None or due > 24):
                    continue                      # flat and not due: skip
                bit = f"`{name:<8}` {n:>2} pos · gross {gross:.2f}"
                if due is not None:
                    bit += (f" · due in {due/24:.0f}d" if due >= 24
                            else f" · due in {due:.0f}h")
                L.append(bit)
            L.append(f"**total** {tot_n} positions · gross {tot_g:.2f}")
            L.append("")

        # risk, stated in the firm's own terms, every cycle
        L.append("**risk**")
        L.append(f"· floor ${self.floor:,.0f} — **${buffer_:,.0f}** of room "
                 f"({buffer_pct:.1f}%)")
        if pnl < 0:
            L.append(f"· daily limit ${self.daily_limit:,.0f} — "
                     f"**{limit_used:.0f}%** used")
            if pnl <= self.kill_switch:
                L.append(f"· ⚠ kill switch ${self.kill_switch:,.0f} — **TRIPPED**, "
                         f"flat for the rest of the session")
            else:
                head = self.kill_switch - pnl
                L.append(f"· kill switch ${self.kill_switch:,.0f} — "
                         f"${abs(head):,.0f} away")
        else:
            L.append(f"· daily limit ${self.daily_limit:,.0f} — not approached")

        if self.problems:
            L.append("")
            L.append("**needs attention**")
            for p in self.problems[:5]:
                L.append("· ⚠ " + p)
        if self.notes:
            L.append("")
            L.append("**watching**")
            for n in self.notes[:4]:
                L.append("· " + n)
        return "\n".join(L)

    def post(self, webhook: Optional[str] = None) -> bool:
        return _post(webhook or os.environ.get("DISCORD_WEBHOOK_REPORT", ""),
                     self.render())


def from_cycle(report: dict, state, config, findings=None) -> CycleReport:
    """Build a CycleReport from the orchestrator's own report dict.

    Kept tolerant of missing keys on purpose: a reporting layer must never be
    the thing that breaks a trading cycle.
    """
    eq = getattr(state, "equity", 0.0)
    r = CycleReport(
        equity=eq,
        day_start_equity=getattr(state, "day_start_equity", eq),
        vol_target=(config.daily_vol_target(eq)
                    if hasattr(config, "daily_vol_target") else 3000.0),
        halted=bool(report.get("halted")),
        halt_reason="; ".join(report.get("flags", [])[:2]),
    )
    for sleeve, d in (report.get("sleeve_positions") or {}).items():
        r.sleeve(sleeve, len(d), sum(abs(v) for v in d.values()),
                 report.get("next_due", {}).get(sleeve))
    for f in (findings or []):
        if getattr(f, "severity", "") in ("CRITICAL", "ERROR"):
            r.problem(f"{f.sleeve}: {f.message}")
        elif getattr(f, "severity", "") == "WARN":
            r.note(f"{f.sleeve}: {f.message}")
    return r
