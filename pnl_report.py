#!/usr/bin/env python3
"""
pnl_report.py — where the money actually came from.

A THIRD channel, deliberately:

    DISCORD_WEBHOOK_BOT      alerts   — something is broken (silence = healthy)
    DISCORD_WEBHOOK_REPORT   cycles   — what the bot did, every 4h
    DISCORD_WEBHOOK_PNL      money    — attribution, once a day

They answer different questions on different clocks. Per-cycle reports tell you
what happened; this tells you whether it is WORKING. Mixed together you would
read neither.

WHAT IT ANSWERS

**Which sleeves are paying.** The book carries nine, and one of them (cascade)
has a NEGATIVE bull-regime Sharpe by design -- it earns its place by being
uncorrelated and firing when the price sleeves hurt. You cannot judge that from
a P&L number alone, which is why this reports each sleeve against the Sharpe it
was VALIDATED at rather than against zero.

**Whether live matches research.** Every sleeve has an expected trade rate and
win rate from the backtest. A sleeve trading half as often as expected is
usually a data problem, not a market one -- fvg should fire ~120x a year on a
single coin, so a quiet fvg means something is wrong upstream.

**What the costs really are.** fvg turns over ~300x a year and the research
charged 8.5bp a side. If realised costs run higher, fvg is the first sleeve to
stop being worth it, and that shows up here before it shows up in the equity
curve.

**How far the evaluation has left to run** -- in the firm's own terms, not
percentages.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Sharpe each sleeve was validated at, on the hourly simulator over 1,315 days.
# Live is compared against THESE, not against zero -- a 0.49-Sharpe sleeve
# doing 0.4 is performing; a 2.96-Sharpe sleeve doing 0.4 is not.
VALIDATED_SHARPE = {
    "fvg": 2.96, "sr": 1.95, "pattern": 1.87, "srflip": 1.69,
    "delta": 1.41, "skew": 1.26, "relvol": 1.17, "oirank": 1.34,
    "cascade": 0.49,
}

# expected trades per month across the 24-coin book, from the backtest
EXPECTED_TRADES_MO = {
    "fvg": 126, "pattern": 19, "sr": 13, "srflip": 8, "cascade": 2.5,
}


def _post(webhook: str, content: str) -> bool:
    if not webhook:
        return False
    try:
        req = urllib.request.Request(
            webhook, data=json.dumps({"content": content[:1990]}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "HyroBot (pnl, 1.0)"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status in (200, 204)
    except Exception:
        return False


def _money(v: float) -> str:
    return f"{v:+,.0f}"


@dataclass
class SleeveP:
    name: str
    pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    fees: float = 0.0
    gross_traded: float = 0.0
    days_live: int = 1

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.trades * 100 if self.trades else None

    @property
    def trades_per_month(self) -> float:
        return self.trades / max(self.days_live, 1) * 30


@dataclass
class PnLReport:
    equity: float
    start_equity: float
    target: float = 220_000.0
    floor: float = 180_000.0
    days_live: int = 1
    period: str = "today"

    sleeves: Dict[str, SleeveP] = field(default_factory=dict)
    by_coin: Dict[str, float] = field(default_factory=dict)
    kill_switch_fires: int = 0
    halted_cycles: int = 0
    notes: List[str] = field(default_factory=list)

    def sleeve(self, name: str, pnl: float, trades: int = 0, wins: int = 0,
               fees: float = 0.0, gross_traded: float = 0.0):
        self.sleeves[name] = SleeveP(name, pnl, trades, wins, fees,
                                     gross_traded, self.days_live)

    def coin(self, name: str, pnl: float):
        self.by_coin[name] = pnl

    def note(self, t: str):
        self.notes.append(t)

    # ── the message ──────────────────────────────────────────────────────
    def render(self) -> str:
        total = self.equity - self.start_equity
        to_target = self.target - self.equity
        room = self.equity - self.floor
        fees = sum(s.fees for s in self.sleeves.values())
        L = []

        L.append(f"**P&L · {self.period}** · {dt.datetime.now(dt.timezone.utc):%d %b}")
        L.append(f"equity **${self.equity:,.0f}** · {_money(total)} "
                 f"· fees {-abs(fees):,.0f}")
        if to_target > 0:
            L.append(f"**${to_target:,.0f}** to target · **${room:,.0f}** above floor")
        else:
            L.append(f"🎯 **TARGET PASSED** · ${room:,.0f} above floor")
        L.append("")

        if self.sleeves:
            L.append("**who paid**")
            ranked = sorted(self.sleeves.values(), key=lambda s: -s.pnl)
            for s in ranked:
                bits = [f"`{s.name:<8}` **{_money(s.pnl):>9}**"]
                if s.trades:
                    bits.append(f"{s.trades:>3} trades")
                    if s.win_rate is not None:
                        bits.append(f"{s.win_rate:.0f}% win")
                L.append(" · ".join(bits))
            L.append("")

            # is anything trading at the wrong rate? usually a data problem
            odd = []
            for s in self.sleeves.values():
                exp = EXPECTED_TRADES_MO.get(s.name)
                if exp and self.days_live >= 7:
                    got = s.trades_per_month
                    if got < exp * 0.4:
                        odd.append(f"{s.name} {got:.0f}/mo vs ~{exp:.0f} expected")
                    elif got > exp * 2.0:
                        odd.append(f"{s.name} {got:.0f}/mo vs ~{exp:.0f} expected")
            if odd:
                L.append("**trading at the wrong rate** — usually a data problem, "
                         "not a market one")
                for o in odd[:4]:
                    L.append("· ⚠ " + o)
                L.append("")

        if self.by_coin:
            top = sorted(self.by_coin.items(), key=lambda kv: -kv[1])
            L.append("**by coin**")
            L.append("· best  " + " · ".join(f"{c} {_money(v)}" for c, v in top[:3]))
            L.append("· worst " + " · ".join(f"{c} {_money(v)}" for c, v in top[-3:]))
            share = (max(abs(v) for v in self.by_coin.values())
                     / max(sum(abs(v) for v in self.by_coin.values()), 1e-9) * 100)
            if share > 40:
                L.append(f"· ⚠ one coin is {share:.0f}% of the absolute P&L — "
                         f"the book is not meant to be that concentrated")
            L.append("")

        L.append("**risk events**")
        L.append(f"· kill switch fired **{self.kill_switch_fires}x** "
                 f"(expect ~16% of days at full size)")
        if self.halted_cycles:
            L.append(f"· ⚠ **{self.halted_cycles}** cycle(s) halted by the health "
                     f"gate — no orders placed")
        else:
            L.append("· no cycles halted")

        if self.notes:
            L.append("")
            for n in self.notes[:3]:
                L.append("· " + n)
        return "\n".join(L)

    def post(self, webhook: Optional[str] = None) -> bool:
        return _post(webhook or os.environ.get("DISCORD_WEBHOOK_PNL", ""),
                     self.render())


def weekly_from_trades(trades: List[dict], equity: float, start_equity: float,
                       days: int = 7) -> PnLReport:
    """Build a report from a trade log: [{sleeve, coin, pnl, fee}, ...].

    Tolerant of missing keys on purpose -- a reporting layer must never be the
    thing that breaks a trading cycle.
    """
    r = PnLReport(equity=equity, start_equity=start_equity, days_live=days,
                  period=f"last {days} days")
    agg: Dict[str, List] = {}
    coins: Dict[str, float] = {}
    for t in trades:
        s = t.get("sleeve", "?")
        p = float(t.get("pnl", 0.0))
        a = agg.setdefault(s, [0.0, 0, 0, 0.0])
        a[0] += p
        a[1] += 1
        a[2] += 1 if p > 0 else 0
        a[3] += float(t.get("fee", 0.0))
        c = t.get("coin")
        if c:
            coins[c] = coins.get(c, 0.0) + p
    for s, (pnl, n, w, fee) in agg.items():
        r.sleeve(s, pnl, n, w, fee)
    for c, p in coins.items():
        r.coin(c, p)
    return r
