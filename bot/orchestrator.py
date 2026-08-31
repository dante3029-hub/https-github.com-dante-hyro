"""
Main orchestrator wiring:

    DataFeed
      -> signal_engine snapshot (Main/Flow/DELTA/RELVOL weights, Phase 2)
      -> event_sleeves (Short/BOS slot-based weights, event-driven)
      -> sleeve_history (trailing return histories for vol-scaling)
      -> portfolio_layer.portfolio.size_portfolio() (Phase 3: multipliers,
         leverage, dollar targets)
      -> risk_overlay.RiskState (kill switch / throttle -- already inside
         size_portfolio via risk_state.target_exposure_multiplier())
      -> compliance.ComplianceState (firm pass/fail/bust tracking, separate
         axis from risk_overlay)
      -> diff dollar targets vs current exchange positions
      -> execute via exchange_client (SKIPPED, only logged, if dry_run=True)
      -> persist state
      -> alert

Respects per-sleeve cadence gating: a sleeve's raw weights from THIS cycle's
snapshot are only actually applied to trading (and last_rebalance stamped)
if enough time has passed since its last rebalance. Between rebalances the
sleeve's dollar target is held at its last computed value -- carried
forward from bot_state.last_dollar_targets, not silently zeroed and not
freshly recomputed every cycle just because a cheap signal fetch happened.
"""
import os
import sys
import datetime as dt
import logging
from typing import Optional

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from bot import config
from bot import clock
from bot.state import BotState
from bot.data_feed import DataFeed, MarketSnapshot
from bot.alerts import discord_msg
from bot.reconciliation import reconcile
from bot.execution import build_execution_plan, execute_plan
from bot import execution
from bot import event_sleeves
from portfolio_layer.portfolio import size_portfolio, SLEEVE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("Orchestrator")

CADENCE_HOURS = {
    "main": config.MAIN_CADENCE_HOURS,
    "flow": config.FLOW_CADENCE_HOURS,
    "delta": config.DELTA_CADENCE_HOURS,
    "relvol": config.RELVOL_CADENCE_HOURS,
    "short": config.SHORT_CHECK_HOURS,
    "bos": config.BOS_CHECK_HOURS,
}


def _due(last_iso: Optional[str], cadence_hours: float, now: dt.datetime) -> bool:
    """
    Is `sleeve` due for a rebalance?

    Both sides are forced through `clock` so that a NAIVE timestamp written by
    an older build of this bot (or hand-edited into the state file) cannot
    raise `TypeError: can't subtract offset-naive and offset-aware datetimes`
    against an aware `now`. That was a real crash-on-upgrade path.
    """
    last = clock.parse_iso(last_iso)
    if last is None:
        return True
    return (clock.ensure_utc(now) - last).total_seconds() >= cadence_hours * 3600


class HyroTraderBot:
    def __init__(self, data_feed: DataFeed, exchange_client=None, dry_run: bool = True,
                 state_path: str = config.STATE_FILE, paper_trade: bool = False):
        """
        Three distinct modes, deliberately not collapsed into one flag:

          dry_run=True                  -- no exchange contact whatsoever.
                                           Targets computed and logged only.
          dry_run=False, paper_trade=True  -- READS the real exchange (positions,
                                           marks, instrument filters) and builds
                                           a real, fully-quantized order plan,
                                           but sends nothing. This is the
                                           burn-in mode.
          dry_run=False, paper_trade=False -- sends real orders.

        The middle mode is the one that matters: it is the only way to validate
        dollar->contract conversion, quantization, and the position diff against
        real instrument filters without risking capital.
        """
        self.data_feed = data_feed
        self.exchange_client = exchange_client
        self.dry_run = dry_run
        self.paper_trade = paper_trade
        self.state_path = state_path
        self.state = BotState.load(state_path)
        if not dry_run and exchange_client is None:
            raise ValueError("dry_run=False requires an exchange_client -- refusing to trade live with no client")
        logger.info(f"HyroTraderBot initialized dry_run={dry_run} paper_trade={paper_trade} "
                    f"feed={type(data_feed).__name__} state_path={state_path}")

    # ------------------------------------------------------------ one full cycle
    def run_cycle(self, as_of_date: Optional[dt.date] = None, now: Optional[dt.datetime] = None) -> dict:
        # all time handling goes through bot.clock -- aware UTC, single
        # daily-rollover definition shared with compliance/risk. See clock.py.
        now = clock.ensure_utc(now) if now is not None else clock.now_utc()
        session_date = as_of_date or clock.trading_day(now)
        report = {"cycle": self.state.cycle_count, "as_of": str(session_date), "flags": []}

        # 1. session bookkeeping
        risk_state = self.state.get_risk_state()
        if risk_state.session_date != session_date:
            risk_state.start_new_session(session_date)
            self.state.set_risk_state(risk_state)
        # 2026-08-19: seed a FRESH compliance state's equity/day-start from the
        # account's actual balance, NOT config.initial_balance. initial_balance is
        # the PHASE-START reference the fixed $180,000 floor is measured from; it is
        # not what the account is worth today. Seeding day_start_equity from it on an
        # account at $183,792 produced a phantom -$16,208 intraday loss that tripped
        # the daily limit on every cycle and forced the book flat.
        compliance_state = self.state.get_compliance_state(
            config.HYRO_COMPLIANCE_CONFIG, current_equity=self.state.equity)
        if compliance_state.current_date != session_date:
            compliance_state.start_new_day(session_date)

        # 1b. refresh the live equity mark and feed it into risk_overlay's OWN
        # kill switch / ORIG_THROTTLE drawdown throttle -- this MUST happen
        # before sizing (step 4 below), because size_portfolio() consults
        # risk_state.target_exposure_multiplier() synchronously and that
        # multiplier depends on risk_state.equity/peak_equity and the
        # kill_switch_tripped_today / daily_loss_limit_tripped_today flags.
        #
        # BUG FOUND 2026-08-10: prior to this fix, nothing in this live cycle
        # ever called risk_state.update_equity() or
        # risk_state.on_intraday_pnl_update() -- those were only ever called
        # from unit tests and burn_in.py's manual trip simulation. That meant
        # risk_state.equity/peak_equity never moved off their session-start
        # values: current_drawdown() was permanently 0.0 (throttle
        # permanently 1.0x, i.e. the 4%/8% drawdown de-risking never
        # engaged), and the -$3,000 kill switch could never trip live. Every
        # backtested "with kill switch" Monte Carlo figure produced this
        # session assumed this wiring existed; it did not. This block is the
        # fix. It is deliberately placed BEFORE step 4 so a kill-switch trip
        # this cycle actually zeroes THIS cycle's sizing, not just the next
        # one.
        #
        # running_session_pnl_dollars is computed the same way compliance.py
        # computes its own daily-loss floor: current equity minus the
        # session's start-of-day equity (compliance_state.day_start_equity,
        # already rolled forward by start_new_day() above). This is
        # deliberately the SAME baseline the firm's own $10k daily-loss
        # tracker uses, so both trackers agree on what "this session's P&L"
        # means. risk_overlay.RiskState remains a separate state object from
        # ComplianceState (different reset timing, different consequence --
        # session flatten vs account bust) -- only the day-start reference
        # point is shared.
        if not self.dry_run:
            try:
                self.state.equity = self.exchange_client.get_wallet_balance()  # returns USDT walletBalance as float
            except Exception as e:
                report["flags"].append(f"get_wallet_balance() failed: {e} -- using last known equity, "
                                        f"NOT silently assuming no change")
        equity_mark = self.state.equity  # in dry-run this is only ever updated by an external mark-to-market step
        running_session_pnl_dollars = equity_mark - compliance_state.day_start_equity
        risk_state.update_equity(equity_mark)
        risk_kill_switch_or_limit_tripped = risk_state.on_intraday_pnl_update(running_session_pnl_dollars)
        self.state.set_risk_state(risk_state)
        if risk_kill_switch_or_limit_tripped:
            report["flags"].append(
                f"risk_overlay kill switch TRIPPED this cycle -- running_session_pnl="
                f"${running_session_pnl_dollars:,.2f} vs day_start_equity=${compliance_state.day_start_equity:,.2f}. "
                f"account_multiplier forced to 0.0 for the rest of this session."
            )
        if risk_state.daily_loss_limit_tripped_today:   # 2026-08-10: was elif --
            # a gap past -$10,000 trips BOTH; the elif hid the firm-limit breach
            report["flags"].append(
                f"risk_overlay daily loss limit TRIPPED this cycle -- running_session_pnl="
                f"${running_session_pnl_dollars:,.2f}. account_multiplier forced to 0.0 for the rest of this session."
            )

        # 2. pull market snapshot (signals + event-sleeve slot updates)
        snap: MarketSnapshot = self.data_feed.get_snapshot(
            as_of_date, self.state.short_tracker_state, self.state.bos_tracker_state
        )
        report["data_source"] = snap.data_source
        # note: ShortSleeveTracker/BOSSleeveTracker mutate the state dicts we
        # passed in (self.state.short_tracker_state / bos_tracker_state) IN
        # PLACE (see event_sleeves.py docstring) -- no reassignment needed here.

        # 3. cadence gating -- decide which sleeves' raw weights get applied this cycle
        raw_weights_this_cycle = {
            "main": snap.signal_snapshot.main_weights,
            "flow": snap.signal_snapshot.flow_weights,
            "delta": snap.signal_snapshot.delta_weights,
            "relvol": snap.signal_snapshot.relvol_weights,
            "short": snap.short_weights,
            "bos": snap.bos_weights,
        }
        effective_raw_weights = {}
        report["rebalanced_sleeves"] = []
        for sleeve in SLEEVE_NAMES:
            due = _due(self.state.last_rebalance.get(sleeve), CADENCE_HOURS[sleeve], now)
            if due:
                effective_raw_weights[sleeve] = raw_weights_this_cycle[sleeve]
                self.state.last_rebalance[sleeve] = clock.to_iso(now)
                report["rebalanced_sleeves"].append(sleeve)
            else:
                prev = self.state.last_dollar_targets.get(sleeve, {})
                if prev:
                    report["flags"].append(f"sleeve '{sleeve}' not due for rebalance -- holding prior dollar targets")
                effective_raw_weights[sleeve] = None  # signal to size_portfolio's caller: hold prior, see below

        # 4. size the portfolio (Phase 3) -- only for sleeves due this cycle; others carry forward
        sized_input_weights = {
            k: (v if v is not None else {}) for k, v in effective_raw_weights.items()
        }
        sized = size_portfolio(
            sleeve_raw_weights=sized_input_weights,
            sleeve_return_history={k: v for k, v in snap.sleeve_histories.items() if k in SLEEVE_NAMES},
            risk_state=risk_state,
            window=config.AB_BLEND_WINDOW,
            L=config.LEVERAGE_L,
            max_sleeve_multiplier=config.MAX_SLEEVE_MULTIPLIER,
            stop_fracs=event_sleeves.stop_fracs_from_state(
                self.state.short_tracker_state, self.state.bos_tracker_state),
            max_loss_per_trade=config.MAX_LOSS_PER_TRADE_,
            low_cap_coins=config.LOW_CAP_COINS,
            low_cap_exposure_limit=config.LOW_CAP_EXPOSURE_LIMIT_,
            aggregate_max_loss=config.AGGREGATE_MAX_LOSS_,
        )
        report["flags"].extend(sized.flags)

        # carry forward dollar targets for sleeves not due this cycle.
        #
        # CRITICAL ORDERING NOTE: carry-forward is deliberately gated on
        # account_multiplier != 0. `size_portfolio()` already multiplies every
        # sleeve by account_multiplier, so when the kill switch or daily loss
        # limit trips it correctly returns ZERO dollars for every sleeve --
        # but an unconditional carry-forward then RESTORED the previous
        # cycle's non-zero targets for every sleeve that wasn't due to
        # rebalance, silently re-arming the book at the exact moment it is
        # required to be flat. Measured impact of that bug before this fix:
        # kill switch tripped at -$3,100 intraday, account_multiplier=0.0,
        # yet combined gross notional came back $45,460.50 instead of $0.00.
        # A tripped kill switch MUST dominate cadence. Do not reorder this.
        killed = (sized.account_multiplier == 0.0)
        final_dollar_targets = dict(sized.dollar_targets)
        if killed:
            final_dollar_targets = {s: {} for s in SLEEVE_NAMES}
            report["flags"].append(
                "account_multiplier=0.0 -- kill switch / daily loss limit tripped: ALL sleeves forced "
                "flat and cadence carry-forward SUPPRESSED (stale targets are not re-applied)."
            )
        else:
            for sleeve in SLEEVE_NAMES:
                if effective_raw_weights[sleeve] is None:
                    final_dollar_targets[sleeve] = self.state.last_dollar_targets.get(sleeve, {})

        # 5. compliance check against current equity mark (equity_mark and
        # the wallet-balance refresh already happened in step 1b above --
        # single fetch per cycle, shared by both risk_overlay and compliance)
        compliance_status = compliance_state.update_equity(equity_mark, as_of=now)
        self.state.set_compliance_state(compliance_state)
        report["compliance"] = compliance_status
        if compliance_status["busted"]:
            # A compliance bust is terminal for the account, so it must also
            # force the book flat -- previously this only logged a flag and
            # left the sized targets untouched, meaning the bot would keep
            # holding (and rebalancing into) positions on a busted account.
            final_dollar_targets = {s: {} for s in SLEEVE_NAMES}
            killed = True
            report["flags"].append("COMPLIANCE BREACH -- account busted per HyroTrader rules. ALL sleeves "
                                    "forced flat. This is separate from the risk_overlay kill switch and "
                                    "is NOT self-clearing on the next session.")
            discord_msg(f"[HyroTraderBot] COMPLIANCE BREACH at {clock.to_iso(now)}: {compliance_status}")

        # 6. persist the FINAL (post-flatten) targets, never the pre-flatten ones
        self.state.last_dollar_targets = final_dollar_targets
        report["forced_flat"] = killed

        # 7. aggregate per-coin dollar targets across sleeves (independent/additive model)
        combined_targets: dict = {}
        for sleeve_dollars in final_dollar_targets.values():
            for coin, d in sleeve_dollars.items():
                combined_targets[coin] = combined_targets.get(coin, 0.0) + d
        report["combined_dollar_targets"] = combined_targets
        report["gross_notional"] = sum(abs(v) for v in combined_targets.values())
        report["sleeve_multipliers"] = sized.sleeve_multipliers
        report["account_multiplier"] = sized.account_multiplier

        # 8. execution (skipped/logged only in dry-run)
        if self.dry_run:
            report["execution"] = "SKIPPED (dry_run=True) -- targets computed and logged only, no orders placed"
            logger.info(f"[DRY RUN] cycle {self.state.cycle_count}: gross_notional=${report['gross_notional']:,.0f} "
                        f"nonzero_coins={sum(1 for v in combined_targets.values() if abs(v) > 1)}")
        else:
            tracked = {
                config.to_bybit_symbol(coin): {"side": "Buy" if d > 0 else "Sell", "qty": abs(d)}
                for coin, d in combined_targets.items() if abs(d) > 1
            }
            recon = reconcile(self.exchange_client, tracked)
            report["reconciliation"] = {
                "externally_closed": len(recon.externally_closed),
                "orphans": len(recon.orphans),
                "matched": len(recon.matched),
            }
            if recon.orphans:
                discord_msg(f"[HyroTraderBot] {len(recon.orphans)} ORPHAN position(s) found on exchange -- "
                            f"not auto-adopted, needs manual review.")

            # ---- real execution: diff dollar targets against live positions ----
            # `killed` forces an unconditional flatten; the plan builder ignores
            # the no-trade band in that case so nothing can be left on the book.
            plan = build_execution_plan(
                self.exchange_client,
                combined_targets,
                force_flat=killed,
            )
            report["execution_plan"] = plan.summary()
            report["execution_orders"] = [
                {"symbol": o.symbol, "side": o.side, "qty": o.qty_str,
                 "usd": round(o.usd_delta, 2), "reduceOnly": o.reduce_only,
                 "reason": o.reason}
                for o in plan.orders
            ]
            if plan.errors:
                logger.error(f"execution planning errors: {plan.errors}")
                discord_msg(f"[HyroTraderBot] {len(plan.errors)} symbol(s) failed execution "
                            f"planning this cycle: {plan.errors[:5]}")

            exec_result = execute_plan(
                self.exchange_client, plan,
                dry_run=self.paper_trade,
                stop_loss_frac=config.LIVE_STOP_LOSS_FRAC,
            )
            report["execution"] = exec_result
            n_sent = len(exec_result["sent"])
            n_failed = len(exec_result["failed"])
            if n_failed:
                discord_msg(f"[HyroTraderBot] {n_failed} ORDER(S) FAILED this cycle -- "
                            f"book may be partially rebalanced: {exec_result['failed'][:5]}")
            logger.info(f"cycle {self.state.cycle_count}: {n_sent} order(s) "
                        f"{'planned (paper)' if self.paper_trade else 'sent'}, {n_failed} failed, "
                        f"gross traded ${plan.gross_traded_usd:,.0f}"
                        + (" [FORCED FLAT]" if killed else ""))

        # 8b. PROTECTIVE STOP SYNC -- runs EVERY cycle, including cycles where
        #     no sleeve was due to rebalance and no order was placed.
        #
        #     Bug fixed 2026-08-09: the SHORT sleeve's chandelier trail and the
        #     BOS sleeve's 2-ATR stop / 3-ATR TP were computed inside the
        #     trackers and never sent to the venue. _attach_stops() only fires
        #     for symbols in an execution plan and only sets the static 60%
        #     stop, which is correct for the four cross-sectional sleeves and
        #     wrong for the two event sleeves. Live positions therefore sat on
        #     a loose entry stop while the bot's internal state believed the
        #     trail had tightened. The backtest exits on the trail; the live
        #     bot would not have.
        #
        #     Gated on forced-flat: if the kill switch just flattened the book
        #     there are no positions to protect and every amend would 404.
        short_sl = event_sleeves.short_protective_stops(self.state.short_tracker_state)
        bos_sl, bos_tp = event_sleeves.bos_protective_levels(self.state.bos_tracker_state)
        stops_by_symbol = {config.to_bybit_symbol(c): v
                           for c, v in list(short_sl.items()) + list(bos_sl.items())}
        tps_by_symbol = {config.to_bybit_symbol(c): v for c, v in bos_tp.items()}
        if stops_by_symbol or tps_by_symbol:
            if killed:
                report["protective_stops"] = (
                    "SKIPPED -- book force-flattened this cycle, no positions to protect")
            else:
                sync = execution.sync_protective_stops(
                    self.exchange_client, stops_by_symbol,
                    dry_run=self.dry_run or self.paper_trade,
                    tps_by_symbol=tps_by_symbol,
                )
                report["protective_stops"] = sync
                n_fail = len(sync["stops_failed"]) + len(sync["tps_failed"])
                if n_fail and not (self.dry_run or self.paper_trade):
                    discord_msg(f"[HyroTraderBot] {n_fail} protective-stop amend(s) FAILED -- "
                                f"event-sleeve position(s) may be running on a stale stop: "
                                f"{(sync['stops_failed'] + sync['tps_failed'])[:5]}")
                logger.info(f"cycle {self.state.cycle_count}: protective stops synced "
                            f"{len(sync['stops_synced'])} ok / {n_fail} failed")
        else:
            report["protective_stops"] = "none open"

        # 9. persist + audit trail
        self.state.cycle_count += 1
        self.state.equity_curve.append({"date": str(session_date), "equity": equity_mark})
        self.state.save(self.state_path)

        return report
