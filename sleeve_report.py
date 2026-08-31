#!/usr/bin/env python3
"""
sleeve_report.py — per-sleeve Discord reporting.

WHY: `discord_msg()` only ever sent one-line crash/breach strings. There was no
way to follow WHAT each sleeve holds, WHEN it next acts, or WHY it did nothing
this cycle. On a book that rebalances weekly and fires events every 4h, "nothing
happened" and "something is broken" look identical without this.

CADENCES (bot/config.py):
    main    72h     flow  168h (weekly)
    delta  168h     relvol 168h (weekly)
    short     4h    bos      4h   (event-driven: check every 4h, fire on signal)

So on a normal day the honest answer to "when will it trade?" is: the weekly
sleeves act once every 7 days, and short/bos act only when a setup appears --
roughly a third of days for BOS. Silence is the expected state, which is exactly
why the next-due times need reporting.

USAGE
    from sleeve_report import post_cycle_report
    post_cycle_report(report, state, equity)          # after bot.run_cycle()

    python3 sleeve_report.py            # standalone: print current status
    python3 sleeve_report.py --post     # ...and send it to Discord
"""
from __future__ import annotations

import datetime as dt
import os
import sys

ROOT = os.environ.get("HYRO_WORKSPACE", os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "bot"), os.path.join(ROOT, "existing_botcode", "botcode")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

CADENCE_HOURS = {"main": 72, "flow": 168, "delta": 168,
                 "relvol": 168, "short": 4, "bos": 4}
EVENT_SLEEVES = {"short", "bos"}          # fire on signal, not on a clock


def _fmt_due(last_iso: str | None, hours: float, now: dt.datetime) -> str:
    if not last_iso:
        return "DUE NOW (never run)"
    try:
        last = dt.datetime.fromisoformat(last_iso)
    except ValueError:
        return "DUE NOW (unparseable timestamp)"
    if last.tzinfo is None:
        last = last.replace(tzinfo=dt.timezone.utc)
    due = last + dt.timedelta(hours=hours)
    delta = (due - now).total_seconds() / 3600.0
    if delta <= 0:
        return "DUE NOW"
    if delta < 48:
        return f"in {delta:.1f}h"
    return f"in {delta/24:.1f}d"


def build_report(state, equity: float, report: dict | None = None,
                 now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    targets = getattr(state, "last_dollar_targets", {}) or {}
    last_reb = getattr(state, "last_rebalance", {}) or {}

    lines = [f"**HyroTrader cycle** · {now.strftime('%Y-%m-%d %H:%M')} UTC",
             f"equity **${equity:,.2f}**"]

    if report:
        gross = report.get("gross_notional", 0.0)
        lines.append(f"gross notional **${gross:,.0f}** · "
                     f"rebalanced: {report.get('rebalanced_sleeves') or 'none'}")

    lines.append("")
    lines.append("```")
    lines.append(f"{'sleeve':<8}{'legs':>5}{'gross $':>11}{'next action':>16}")
    for sleeve in ("main", "flow", "delta", "relvol", "short", "bos"):
        book = targets.get(sleeve, {}) or {}
        legs = sum(1 for v in book.values() if v)
        gross = sum(abs(v) for v in book.values())
        if sleeve in EVENT_SLEEVES:
            nxt = f"checks {CADENCE_HOURS[sleeve]}h"
        else:
            nxt = _fmt_due(last_reb.get(sleeve), CADENCE_HOURS[sleeve], now)
        lines.append(f"{sleeve:<8}{legs:>5}{gross:>11,.0f}{nxt:>16}")
    lines.append("```")

    # positions, largest first -- this is what you'd see on the exchange
    merged: dict[str, float] = {}
    for book in targets.values():
        for coin, d in (book or {}).items():
            if d:
                merged[coin] = merged.get(coin, 0.0) + d
    if merged:
        lines.append("**net targets** (per coin, all sleeves netted)")
        lines.append("```")
        for coin, d in sorted(merged.items(), key=lambda kv: -abs(kv[1])):
            side = "LONG " if d > 0 else "SHORT"
            lines.append(f"{coin:<10}{side} ${abs(d):>9,.0f}")
        lines.append("```")
    else:
        lines.append("_no open targets — all sleeves flat_")

    if report:
        flags = report.get("flags", []) or []
        # Cap/flat flags are routine on this book; surface the ones that mean
        # something went WRONG rather than the ones that mean a limit worked.
        serious = [f for f in flags if any(k in f.lower() for k in
                   ("breach", "kill switch", "tripped", "failed", "busted", "halted"))]
        if serious:
            lines.append("**⚠ ALERTS**")
            for f in serious[:6]:
                lines.append(f"• {f}")
        elif flags:
            lines.append(f"_{len(flags)} routine flags (caps/cadence) — no alerts_")
    return "\n".join(lines)


def post_cycle_report(report: dict | None, state, equity: float) -> None:
    """Never raises — a reporting failure must not affect trading."""
    try:
        from bot.alerts import discord_msg
        discord_msg(build_report(state, equity, report))
    except Exception as e:                                        # noqa: BLE001
        print(f"sleeve_report: post failed ({e})")


def main() -> int:
    import json
    sp = os.path.join(ROOT, "bot_runtime", "bot_state.json")
    if not os.path.exists(sp):
        print(f"no state file at {sp}")
        return 1
    raw = json.load(open(sp))

    class _S:  # lightweight view over the JSON, avoids importing BotState
        last_dollar_targets = raw.get("last_dollar_targets", {})
        last_rebalance = raw.get("last_rebalance", {})

    text = build_report(_S(), float(raw.get("equity", 0.0)))
    print(text.replace("```", "").replace("**", ""))
    if "--post" in sys.argv:
        try:
            from bot.alerts import discord_msg
            discord_msg(text)
            print("\n[posted to Discord]")
        except Exception as e:                                    # noqa: BLE001
            print(f"\n[post failed: {e}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
