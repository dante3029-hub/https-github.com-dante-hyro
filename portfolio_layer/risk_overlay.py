"""
Leverage overlay, ORIG_THROTTLE drawdown-based exposure cut, and the -$3,000
intraday kill switch -- restructured from mc_hourly_ks.py's vectorized batch
Monte Carlo simulator (mc_285_hourly) into a live, state-carrying tracker
that processes one account-equity update at a time.

Threshold logic (throttle bands, kill switch truncation) is reused UNCHANGED
from mc_hourly_ks.py -- this module does not invent new risk math, only
reshapes the same math for live, incremental use instead of a pre-generated
batch of simulated days.
"""
import datetime as dt
from dataclasses import dataclass, field

L_DEFAULT = 1.70
ORIG_THROTTLE = [(0.04, 1.0), (0.08, 0.5), (1.0, 0.30)]   # from mc_hourly_ks.py -- do not diverge
KILL_SWITCH_DOLLARS = 3_000.0                              # locked final parameter (BUILD_PLAN.md Phase 0)
DAILY_LOSS_LIMIT = 10_000.0                                 # HyroTrader daily loss limit


def throttle_multiplier(drawdown_frac: float, throttle=ORIG_THROTTLE) -> float:
    """
    drawdown_frac: 1 - equity/peak_equity (fraction, >= 0).
    Reproduces mc_hourly_ks.py's exact band logic:
        dd < 0.04         -> 1.00  (full exposure)
        0.04 <= dd < 0.08  -> 0.50  (half exposure)
        dd >= 0.08         -> 0.30  (30% exposure)
    Implemented identically to the original (iterate thresholds in reverse,
    take the tightest match) so behavior at the exact boundary values matches.
    """
    mult = throttle[-1][1]
    for cutoff, m in reversed(throttle):
        if drawdown_frac < cutoff:
            mult = m
    return mult


@dataclass
class RiskState:
    """Carries the account-level state the throttle and kill switch need
    across calls: running peak equity, current session's flatten/blocked
    status, and the daily-loss-limit trip flag."""
    equity: float
    peak_equity: float
    session_date: "dt.date | None" = None
    kill_switch_tripped_today: bool = False
    daily_loss_limit_tripped_today: bool = False
    session_realized_pnl: float = 0.0

    @classmethod
    def new(cls, starting_equity: float):
        return cls(equity=starting_equity, peak_equity=starting_equity)

    def start_new_session(self, session_date: dt.date):
        """Call once at the start of each trading day/session. Resets the
        kill-switch and daily-loss-limit flags -- both are explicitly a
        PER-SESSION block, not a permanent one, per the locked -$3,000
        kill-switch spec (\"flatten and block re-entry UNTIL NEXT SESSION\")."""
        self.session_date = session_date
        self.kill_switch_tripped_today = False
        self.daily_loss_limit_tripped_today = False
        self.session_realized_pnl = 0.0

    def current_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)

    def target_exposure_multiplier(self, L: float = L_DEFAULT) -> float:
        """
        Combined multiplier to apply to the UNLEVERED, un-throttled target
        weight vector: L * throttle(drawdown). Returns 0.0 if the kill
        switch or daily loss limit has already tripped this session (flat,
        blocked from re-entry until start_new_session() is called again).
        """
        if self.kill_switch_tripped_today or self.daily_loss_limit_tripped_today:
            return 0.0
        return L * throttle_multiplier(self.current_drawdown())

    def on_intraday_pnl_update(self, running_session_pnl_dollars: float) -> bool:
        """
        Call on every fresh intraday mark (e.g. every fill or every price
        tick) with the CUMULATIVE realized+unrealized P&L for the current
        session (since start_new_session()). Returns True the moment the
        kill switch trips (caller should flatten all sleeve positions
        immediately when this returns True) -- mirrors mc_hourly_ks.py's
        `crossed = dollar_path <= -kill_switch` truncation, applied
        incrementally instead of over a precomputed hourly path.
        """
        self.session_realized_pnl = running_session_pnl_dollars

        # 2026-08-10 FIX. Both thresholds are evaluated BEFORE returning.
        # Previously the kill-switch branch returned early, so a single mark
        # that gapped straight past -$10,000 set kill_switch_tripped_today but
        # left daily_loss_limit_tripped_today False. Trading was still safe
        # (target_exposure_multiplier() returns 0.0 on either flag), but the
        # orchestrator's elif then logged only the -$3,000 kill switch and
        # NEVER recorded that the firm's hard $10,000 daily limit had been
        # breached -- you would hear about it from HyroTrader, not your logs.
        newly_killed = False
        if not self.kill_switch_tripped_today and running_session_pnl_dollars <= -KILL_SWITCH_DOLLARS:
            self.kill_switch_tripped_today = True
            newly_killed = True
        if not self.daily_loss_limit_tripped_today and running_session_pnl_dollars <= -DAILY_LOSS_LIMIT:
            self.daily_loss_limit_tripped_today = True
        return newly_killed

    def update_equity(self, new_equity: float):
        """Call at end of day / after settlement to roll peak_equity forward,
        exactly matching mc_hourly_ks.py's `peak = np.maximum(peak, eq[:, k+1])`."""
        self.equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
