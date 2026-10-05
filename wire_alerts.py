#!/usr/bin/env python3
"""
wire_alerts.py — make the ALERTS channel actually fire.

THE GAP
`DISCORD_WEBHOOK_URL` -> reports channel, posted at step 5 of every cycle.
`DISCORD_WEBHOOK_BOT` -> alerts channel, set in .env, and NOTHING CALLS IT.

Everything that went wrong this week would have shown up there:
  * the doctor gate blocked 4 days of cycles (exit 1 on warnings + `&&`)
  * a sed wrote an empty webhook, so reports silently stopped
  * the risk caps looked unapplied because the report showed saved state

All of it looked exactly like a quiet market.

WHAT THIS ADDS
  1. run_cycle.sh posts to the ALERTS channel when a cycle fails or is gated
  2. a watchdog cron that shouts if no cycle has completed in 5 hours --
     a bot that has STOPPED sends no alerts, which is indistinguishable from
     a bot with nothing to say

    python3 wire_alerts.py --check
    python3 wire_alerts.py
"""
import os
import shutil
import sys

RC = "run_cycle.sh"

ALERT_FN = '''
# ---------------------------------------------------------------- alerting
# Posts to the ALERTS channel (DISCORD_WEBHOOK_BOT), which is separate from the
# reports channel on purpose: routine reports train you to scroll past alerts.
alert() {
    local msg="$1"
    local url="${DISCORD_WEBHOOK_BOT:-}"
    [ -z "$url" ] && return 0
    curl -s -m 10 -H "Content-Type: application/json" \\
        -d "{\\"content\\": $(printf '%s' "$msg" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()[:1900]))')}" \\
        "$url" > /dev/null 2>&1 || true
}

# Any non-zero exit from here on gets announced. Without this a failed cycle
# is as silent as a healthy one.
on_fail() {
    local code=$?
    [ $code -ne 0 ] && alert "🔴 **HyroTrader cycle FAILED** (exit $code)
host: $(hostname -s)  ·  $(date -u '+%Y-%m-%d %H:%M') UTC
last log lines:
\\`\\`\\`$(tail -6 /root/cron.log 2>/dev/null | cut -c1-300)\\`\\`\\`"
    exit $code
}
trap on_fail EXIT
'''

WATCHDOG = '''#!/bin/bash
# watchdog.sh -- shouts if no cycle has completed recently.
#
# A bot that has STOPPED sends no alerts, which looks exactly like a bot with
# nothing to say. This runs from its OWN cron entry so the alarm does not
# depend on the thing it is watching.
#
# The doctor gate silently blocked four days of cycles. This is what would
# have caught it on day one.
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a
LOG=/root/cron.log
MAX_H=5

[ -f "$LOG" ] || exit 0
AGE=$(( ( $(date +%s) - $(stat -c %Y "$LOG") ) / 3600 ))
[ "$AGE" -lt "$MAX_H" ] && exit 0

MSG="🔴 **HyroTrader is NOT RUNNING**
no cycle has completed in ${AGE}h (limit ${MAX_H}h)
host: $(hostname -s)  ·  $(date -u '+%Y-%m-%d %H:%M') UTC
check:  crontab -l | grep run_cycle   ·   tail -20 /root/cron.log"

[ -n "${DISCORD_WEBHOOK_BOT:-}" ] && curl -s -m 10 -H "Content-Type: application/json" \\
    -d "$(python3 -c "import json,os; print(json.dumps({'content': os.environ['MSG']}))" MSG="$MSG")" \\
    "$DISCORD_WEBHOOK_BOT" > /dev/null 2>&1
exit 0
'''

if __name__ == "__main__":
    if not os.path.exists(RC):
        print(f"{RC} not found -- run from the repo root")
        sys.exit(1)
    s = open(RC).read()
    if "DISCORD_WEBHOOK_BOT" in s:
        print("run_cycle.sh already wired")
    elif "--check" in sys.argv:
        print("ready to patch run_cycle.sh and write watchdog.sh")
        sys.exit(0)
    else:
        # insert after the .env sourcing block so the webhook is available
        anchor = 'set -a; . "$HYRO_WORKSPACE/.env"; set +a'
        if anchor not in s:
            print("could not find the .env sourcing line -- not patching")
            sys.exit(1)
        i = s.index(anchor) + len(anchor)
        # step past the closing fi of that if-block
        j = s.find("\nfi\n", i)
        j = (j + 4) if j != -1 else (i + 1)
        shutil.copy2(RC, RC + ".bak_alerts")
        open(RC, "w").write(s[:j] + ALERT_FN + s[j:])
        print(f"patched {RC}  (backup: {RC}.bak_alerts)")

    if "--check" not in sys.argv:
        with open("watchdog.sh", "w") as f:
            f.write(WATCHDOG)
        os.chmod("watchdog.sh", 0o755)
        print("wrote watchdog.sh")
        print("\nadd the watchdog cron (separate entry, on purpose):")
        print("  (crontab -l; echo '23 * * * * cd ~/bot_hyrotrader_v1 && "
              "./watchdog.sh') | crontab -")
        print("\ntest both:")
        print("  ./watchdog.sh                     # silent if a cycle ran recently")
        print("  touch -d '9 hours ago' /root/cron.log && ./watchdog.sh")
