#!/usr/bin/env bash
# =============================================================================
# run_cycle.sh — the scheduled entry point. Cron calls THIS, not bot.main.
#
# WHY THIS EXISTS
#   bot/main.py runs ONE cycle and exits. Nothing refreshes the data panels,
#   nothing rolls the trading day, and nothing reports. Left to cron alone the
#   bot would score week-old CSVs against a day-start equity that never moves.
#
#   Order matters and is not arbitrary:
#     1. refresh panels   -- signals must see today's bars
#     2. verify freshness -- refuse to trade on a stale panel
#     3. roll the session -- start_new_day() so daily baselines/flags reset
#     4. run the cycle
#     5. report           -- per-sleeve state to Discord
#
# USAGE
#     ./run_cycle.sh              # dry run (default, places NO orders)
#     ./run_cycle.sh --execute    # places real orders
#
# CRON (every 4h, aligned to the BOS/short check cadence):
#     0 */4 * * * /root/bot_hyrotrader_v1/run_cycle.sh >> /root/cron.log 2>&1
# =============================================================================
set -uo pipefail

export HYRO_WORKSPACE="${HYRO_WORKSPACE:-/root/bot_hyrotrader_v1}"
export BYBIT_USE_DEMO="${BYBIT_USE_DEMO:-true}"
cd "$HYRO_WORKSPACE" || { echo "FATAL: cannot cd to $HYRO_WORKSPACE"; exit 2; }

# Credentials are NOT baked into this script. Put them in a 0600 file so they
# are not world-readable and not in shell history / .bashrc.
if [ -f "$HYRO_WORKSPACE/.env" ]; then
    set -a; . "$HYRO_WORKSPACE/.env"; set +a
fi

EXECUTE=""
[ "${1:-}" = "--execute" ] && EXECUTE="--execute"
STAMP() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "===== $(STAMP) run_cycle.sh start ${EXECUTE:-(dry run)} ====="

# ---------------------------------------------------------------- 1. refresh
echo "[1/5] refreshing panels"
if ! python3 live_refresh.py --universe b; then
    echo "FATAL: panel refresh failed -- NOT running a cycle on stale data"
    python3 - <<'PY' 2>/dev/null || true
from bot.alerts import discord_msg
discord_msg("🔴 **HyroTrader** panel refresh FAILED — cycle skipped, no trading on stale data.")
PY
    exit 1
fi

# ---------------------------------------------------------------- 2. verify
echo "[2/5] verifying freshness"
python3 - <<'PY' || exit 1
import os, sys
sys.path.insert(0, os.environ["HYRO_WORKSPACE"])
from live_refresh import panel_staleness, MAX_STALE_HOURS
newest, age = panel_staleness(os.path.join(os.environ["HYRO_WORKSPACE"], "clean_panel"))
print(f"      clean_panel newest={newest} age={age:.0f}h (max {MAX_STALE_HOURS}h)")
if newest is None or age > MAX_STALE_HOURS:
    print("FATAL: panel stale after refresh -- aborting")
    sys.exit(1)
PY

# ------------------------------------------------------- 3. roll the session
# The orchestrator only calls start_new_session()/start_new_day() when the
# stored session_date differs from today. That works -- PROVIDED equity is
# marked to the live wallet first, otherwise the new day's baseline is seeded
# from a stale figure and the daily-loss check measures against a phantom.
echo "[3/5] rolling session / marking equity"
python3 - <<'PY' || echo "      WARN: equity mark failed, cycle will use last known"
import os, sys, json, datetime as dt
ROOT = os.environ["HYRO_WORKSPACE"]
for p in (ROOT, os.path.join(ROOT, "bot"), os.path.join(ROOT, "existing_botcode", "botcode")):
    if os.path.isdir(p) and p not in sys.path: sys.path.insert(0, p)
key, sec = os.environ.get("BYBIT_API_KEY"), os.environ.get("BYBIT_API_SECRET")
if not (key and sec):
    print("      no credentials -- skipping live equity mark"); raise SystemExit
import importlib.util as u
sp = u.spec_from_file_location("hyro_cfg", os.path.join(ROOT, "bot", "config.py"))
cfg = u.module_from_spec(sp); sp.loader.exec_module(cfg)
from bot.exchange_client import HardenedBybitClient, ExchangeMode
mode = ExchangeMode.DEMO if cfg.BYBIT_USE_DEMO else ExchangeMode.MAINNET
eq = float(HardenedBybitClient(key, sec, mode=mode).get_wallet_balance())
fp = os.path.join(ROOT, "bot_runtime", "bot_state.json")
st = json.load(open(fp)) if os.path.exists(fp) else {}
today = dt.datetime.now(dt.timezone.utc).date().isoformat()
prev = st.get("session_date")
st["equity"] = eq
if prev != today:
    # NEW TRADING DAY: baseline must be TODAY's opening equity, and the sticky
    # daily flags must clear. Without this the -$3,000 kill switch and the
    # $10,000 daily limit stay tripped forever once they fire once.
    st["day_start_equity"] = eq
    rs = st.get("risk_state") or {}
    rs.update(kill_switch_tripped_today=False, daily_loss_limit_tripped_today=False,
              session_realized_pnl=0.0, equity=eq)
    st["risk_state"] = rs
    cs = st.get("compliance_state")
    if isinstance(cs, dict):
        cs["day_start_equity"] = eq; cs["equity"] = eq
        cs["daily_loss_breached"] = False       # daily flag resets; max-DD does NOT
        st["compliance_state"] = cs
    print(f"      NEW DAY {today}: baseline reset to ${eq:,.2f}")
else:
    print(f"      same day {today}: equity marked ${eq:,.2f}, baseline unchanged")
json.dump(st, open(fp, "w"), indent=2)
PY

# ---------------------------------------------------------------- 4. cycle
echo "[4/5] running cycle"
python3 -u -m bot.main $EXECUTE
RC=$?
echo "      exit=$RC"

# ---------------------------------------------------------------- 5. report
echo "[5/5] reporting"
python3 sleeve_report.py --post || echo "      WARN: report failed"

echo "===== $(STAMP) done rc=$RC ====="
exit $RC
