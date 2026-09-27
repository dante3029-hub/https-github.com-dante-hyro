#!/usr/bin/env python3
"""
alerts.py — the bot tells you what it did, and shouts when something is wrong.

Reuses the Discord webhooks already working for the scanner. Set one for the
bot rather than sharing a signal channel -- you want alerts to stand out, not
scroll past between pattern grids:

    export DISCORD_WEBHOOK_BOT='https://discord.com/api/webhooks/...'

THREE LEVELS, deliberately different in volume

  heartbeat   once per cycle, ONLY if something changed. A bot that posts
              every cycle trains you to ignore it.
  alert       something needs attention but trading continues
  critical    trading has STOPPED. Always sent, always @-mentions.

WHY THIS EXISTS
Every failure on this project was silent. BOS produced zero weights for 300+
cycles while the log said "rebalanced=['short','bos']". run/ went 408 hours
stale while cycles kept "succeeding". Nobody was watching the log, because
nothing ever asked them to.

The rule here is the inverse: SILENCE MEANS HEALTHY. If the bot is alive and
nothing has changed, it says nothing. Anything in the channel is worth reading.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

HEARTBEAT, ALERT, CRITICAL = "heartbeat", "alert", "critical"

_EMOJI = {HEARTBEAT: "\U0001F7E2", ALERT: "\U0001F7E1", CRITICAL: "\U0001F534"}


def _post(webhook: str, content: str, retries: int = 3) -> bool:
    """Discord rejects urllib's default User-Agent with a 403 -- curl works and
    the script does not. That cost an evening once."""
    if not webhook:
        return False
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                webhook,
                data=json.dumps({"content": content[:1900]}).encode(),
                headers={"Content-Type": "application/json",
                         "User-Agent": "HyroBot (alerts, 1.0)"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status in (200, 204)
        except urllib.error.HTTPError as e:
            if e.code == 429:                       # rate limited
                time.sleep(2 * (attempt + 1))
                continue
            return False
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return False


class Alerter:
    """Posts to Discord, and remembers what it already said.

    `state` is a plain dict owned by the bot's state file, so de-duplication
    survives restarts -- otherwise a crash-loop would spam the channel with
    the same message every cycle.
    """

    def __init__(self, state: dict, webhook: Optional[str] = None,
                 mention: str = ""):
        self.state = state
        self.state.setdefault("sent", {})
        self.webhook = webhook or os.environ.get("DISCORD_WEBHOOK_BOT", "")
        self.mention = mention          # e.g. "<@1234567890>" to ping yourself

    # ── de-duplication ───────────────────────────────────────────────────
    def _should_send(self, key: str, repeat_after_h: float) -> bool:
        last = self.state["sent"].get(key)
        if last is None:
            return True
        return (time.time() - last) / 3600.0 >= repeat_after_h

    def _mark(self, key: str):
        self.state["sent"][key] = time.time()

    # ── the three levels ─────────────────────────────────────────────────
    def critical(self, title: str, detail: str = "", key: Optional[str] = None):
        """Trading has stopped. Always sent; repeats every 4h until resolved."""
        k = key or f"crit:{title}"
        if not self._should_send(k, 4.0):
            return
        msg = f"{_EMOJI[CRITICAL]} **TRADING HALTED** {self.mention}\n**{title}**"
        if detail:
            msg += f"\n```{detail[:1200]}```"
        if _post(self.webhook, msg):
            self._mark(k)

    def alert(self, title: str, detail: str = "", key: Optional[str] = None):
        """Needs attention, trading continues. Repeats at most every 12h."""
        k = key or f"alert:{title}"
        if not self._should_send(k, 12.0):
            return
        msg = f"{_EMOJI[ALERT]} **{title}**"
        if detail:
            msg += f"\n{detail[:1200]}"
        if _post(self.webhook, msg):
            self._mark(k)

    def cleared(self, key: str, title: str):
        """Something that WAS critical is now fine. Worth saying once --
        otherwise you never learn whether a problem resolved itself."""
        if key in self.state["sent"]:
            _post(self.webhook, f"\U00002705 **resolved** -- {title}")
            del self.state["sent"][key]

    # ── the per-cycle summary ────────────────────────────────────────────
    def cycle(self, *, equity: float, daily_pnl: float, opened: List[str],
              closed: List[str], gross: float, vol_target: float,
              halted: bool = False, halt_reason: str = "") -> None:
        """Post ONLY if something happened. Silence means healthy."""
        if halted:
            self.critical("cycle halted", halt_reason, key="crit:halted")
            return
        if not opened and not closed:
            return                       # nothing changed: say nothing
        lines = [f"{_EMOJI[HEARTBEAT]} equity **${equity:,.0f}**  "
                 f"today **{daily_pnl:+,.0f}**  gross {gross:.2f}  "
                 f"size ${vol_target:,.0f}/day"]
        for t in opened:
            lines.append(f"  OPEN  {t}")
        for t in closed:
            lines.append(f"  CLOSE {t}")
        _post(self.webhook, "\n".join(lines))

    # ── wiring for the health check ──────────────────────────────────────
    def from_findings(self, findings) -> bool:
        """Turn sleeve_health findings into alerts. Returns True if any
        CRITICAL was found, so the caller can halt on the same signal."""
        crit = [f for f in findings if f.severity == "CRITICAL"]
        errs = [f for f in findings if f.severity == "ERROR"]
        if crit:
            detail = "\n".join(f"{f.sleeve}: {f.message}" +
                               (f" -- {f.detail}" if f.detail else "")
                               for f in crit[:8])
            self.critical(f"{len(crit)} CRITICAL health findings", detail,
                          key="crit:health")
        else:
            self.cleared("crit:health", "health checks passing again")
        for f in errs[:5]:
            self.alert(f"{f.sleeve}: {f.message}", f.detail,
                       key=f"alert:{f.sleeve}:{f.message}")
        return bool(crit)

    # ── the one nobody thinks of until it happens ────────────────────────
    def watchdog(self, last_cycle_ts: Optional[float], max_gap_h: float = 2.0):
        """A bot that has STOPPED sends no alerts -- which looks exactly like
        a healthy quiet period. Call this from a SEPARATE cron entry so the
        alarm does not depend on the thing it is watching.
        """
        if last_cycle_ts is None:
            self.critical("no cycle has ever run", key="crit:watchdog")
            return
        gap = (time.time() - last_cycle_ts) / 3600.0
        if gap > max_gap_h:
            self.critical("BOT IS NOT RUNNING",
                          f"last cycle was {gap:.1f}h ago (limit {max_gap_h}h)",
                          key="crit:watchdog")
        else:
            self.cleared("crit:watchdog", "bot is running again")


def alert_from_doctor(exit_code: int, output: str,
                      webhook: Optional[str] = None) -> None:
    """Post doctor.py's result. Wire it into cron:

        python3 doctor.py > /tmp/doc.txt 2>&1; \
        python3 -c "import alerts,sys; \
          alerts.alert_from_doctor($?, open('/tmp/doc.txt').read())"
    """
    a = Alerter({}, webhook)
    if exit_code >= 2:
        bad = [l for l in output.splitlines() if l.startswith(("XX", " X"))]
        a.critical("doctor says DO NOT TRADE", "\n".join(bad[:10]),
                   key="crit:doctor")
    elif exit_code == 1:
        warn = [l for l in output.splitlines() if l.startswith(" !")]
        a.alert("doctor warnings", "\n".join(warn[:6]), key="alert:doctor")
