"""
Live compliance wiring for HyroTrader's actual account rules -- checks LIVE
account equity against the firm's real max-drawdown floor and daily-drawdown
limit, independent of and in addition to risk_overlay.py's own internal
kill-switch/throttle (which manages the strategy's OWN risk budget, not the
firm's pass/fail/bust criteria). This module answers a different question
than risk_overlay.py: not "should we cut exposure to protect our own P&L"
but "are we about to get disqualified/busted by the prop firm's rules".

RULES MODELED (see STRATEGY.md #44/#44a for full sourcing):
  - Max drawdown floor: TWO possible mechanics, selected via `drawdown_type`:
      STATIC   -- floor is FIXED at phase_start_balance * (1 - max_drawdown_pct)
                  for the entire phase and never moves, regardless of how high
                  equity runs. This is HyroTrader's paid "Swing" drawdown
                  upgrade.
      TRAILING -- floor ratchets UP (never down) as equity makes new
                  all-time highs, staying a fixed dollar distance below the
                  highest peak ever reached. This is HyroTrader's default
                  (unpurchased) drawdown mode.
    CONFIRMED 2026-08-10 (user statement): this account has the Swing/static
    upgrade -- `drawdown_type` therefore now DEFAULTS TO STATIC below. This
    is a verbal user confirmation, not an independently verified dashboard
    screenshot or HyroTrader support reply -- treat as user-asserted fact,
    not independently re-confirmed by this codebase. If the account is ever
    downgraded back to standard/trailing, switch `drawdown_type` back to
    TRAILING before going live, or every number below is wrong in the
    unsafe direction (it would understate real drawdown-floor bust risk).
    The TRAILING branch is kept fully implemented and tested, not deleted,
    for exactly that reason.
  - Daily loss limit: $10,000 (STRATEGY.md #44, DAILY_LOSS_LIMIT in
    risk_overlay.py) -- measured against the START-OF-DAY balance. Unrelated
    to drawdown_type; applies identically under both.
  - Balance/target reset at Phase 2 start: CONFIRMED (STRATEGY.md #44a point
    1) -- Phase 2's $200k baseline and profit target restart from the
    ORIGINAL $200k, not Phase 1's ending equity.

WHY THE MC VALIDATION NUMBERS DID NOT NEED TO BE RE-RUN:
  `mc_hourly_ks.py` (the engine behind the reported 98.63% pass / 0.14% bust
  / 46.0-day median figures in STRATEGY.md #44) checks bust with a FIXED
  constant `bust_lvl=180_000`, never a peak-tracking floor -- i.e. it already
  modeled a STATIC floor, not TRAILING. Reproduced directly against the
  cached hourly path on 2026-08-10: pass 98.63%+/-0.05, bust 0.14%+/-0.01,
  median 46.0 (IID); pass 98.75%+/-0.05, bust 0.07%+/-0.02, median 46.4+/-0.49
  (block) -- exact match to the recorded figures. So those numbers already
  assumed the drawdown mechanic this account actually has; the STATIC
  default added below makes the LIVE compliance code match what was
  already validated, closing a real (and previously undetected) mismatch
  between the validated model and the deployed guardrail.

DrawdownFloorMode (RESET vs CARRY_OVER) below ONLY has any effect under
TRAILING -- under STATIC there is no ratcheted floor to carry over or reset,
so Phase 2 simply reopens at phase_start_balance * (1 - max_drawdown_pct)
regardless of which DrawdownFloorMode is set. This also makes the
previously-flagged "100% instant-bust" Phase-1-to-Phase-2 contradiction in
STRATEGY.md #44a moot for a static-drawdown account: there is no inherited
floor level to be too close to the new phase's starting equity in the first
place. STRATEGY.md #44a's caveat still applies verbatim if this account is
ever on TRAILING (unpurchased) terms.
"""
import datetime as dt
from dataclasses import dataclass, field
from enum import Enum


class DrawdownType(Enum):
    """Which drawdown-floor mechanic the account is actually enrolled in.
    STATIC = paid Swing upgrade (fixed floor, never ratchets).
    TRAILING = default HyroTrader terms (floor ratchets up with new highs).
    """
    STATIC = "static"
    TRAILING = "trailing"


class DrawdownFloorMode(Enum):
    """
    How the max-drawdown floor behaves across a Phase 1 -> Phase 2
    transition WHEN drawdown_type == TRAILING. Has no effect under STATIC
    (see module docstring). UNRESOLVED per STRATEGY.md #44a for the
    TRAILING case -- see module docstring. Pick explicitly; do not assume
    either is correct without confirming with HyroTrader support.
    """
    RESET = "reset"              # floor ratchet restarts fresh at Phase 2 start: floor = 90% of the
                                   # new $200k baseline, then ratchets up only off Phase-2-era peaks.
                                   # Consistent with the CONFIRMED balance/target reset, but NOT
                                   # independently confirmed for the drawdown floor specifically.
    CARRY_OVER = "carry_over"    # the exact floor DOLLAR LEVEL reached by the end of Phase 1 persists
                                   # unchanged into Phase 2 (even though the balance/target itself
                                   # resets to $200k) and continues to ratchet only if/when equity
                                   # exceeds the ALL-TIME peak (tracked across both phases). Matches
                                   # the third-party review's literal claim; per STRATEGY.md #44a this
                                   # combined with the confirmed balance reset can put a Phase-2 start
                                   # uncomfortably close to (or past) the inherited floor depending on
                                   # how high Phase 1's peak ran -- flagged as operationally risky, use
                                   # only if HyroTrader support confirms this is really how their
                                   # system works.


@dataclass
class HyroTraderComplianceConfig:
    """All the account-rule numbers this module checks against. Values below
    are the CONFIRMED Phase 0 rules for the $200k account (STRATEGY.md #44).
    drawdown_floor_mode is the UNRESOLVED flag -- see DrawdownFloorMode."""
    initial_balance: float = 200_000.0
    max_drawdown_pct: float = 0.10          # 10% of phase-start balance = the fixed $ distance
    daily_loss_limit_dollars: float = 10_000.0
    drawdown_type: DrawdownType = DrawdownType.STATIC   # CONFIRMED 2026-08-10 (user statement):
                                              # paid Swing/static upgrade on this account. See module
                                              # docstring -- switch to TRAILING if that ever changes.
    drawdown_floor_mode: DrawdownFloorMode = DrawdownFloorMode.RESET  # only matters if drawdown_type
                                              # is switched to TRAILING; no effect under STATIC.
    min_trading_days: int = 4                # user-confirmed for this account; STRATEGY.md #44a flags
                                              # a 10/5-day discrepancy vs several public reviews -- not
                                              # binding given all modeled time-to-pass numbers exceed it,
                                              # but re-verify before relying on it operationally.


@dataclass
class ComplianceState:
    """
    Tracks LIVE account equity (not backtest equity) against HyroTrader's
    actual pass/fail/bust rules, phase by phase. This is deliberately a
    SEPARATE state object from risk_overlay.RiskState -- that module protects
    the strategy's own risk budget; this one tracks the firm's disqualifying
    conditions, which have a different floor mechanic, different reset
    timing, and different consequences (account termination, not just a
    session flatten).
    """
    config: HyroTraderComplianceConfig
    phase: int = 1                                    # 1 or 2
    equity: float = 0.0
    phase_start_balance: float = 0.0                    # balance level at the start of the CURRENT phase
    peak_equity: float = 0.0                            # all-time peak used to ratchet the floor
    dd_floor_dollars: float = 0.0                       # explicit ratcheting floor state (never decreases)
    day_start_equity: float = 0.0
    current_date: "dt.date | None" = None
    trading_days_count: int = 0
    max_dd_breached: bool = False
    daily_loss_breached: bool = False
    breach_log: list = field(default_factory=list)

    @classmethod
    def start_phase1(cls, config: HyroTraderComplianceConfig,
                     current_equity: float | None = None):
        """
        2026-08-19 FIX. `initial_balance` was doing two incompatible jobs:
          (a) the PHASE-START balance the fixed floor is measured from -- for
              HyroTrader static DD this is $200,000, and the floor is $180,000
              no matter what the account is worth today;
          (b) the SESSION opening equity used for the daily-loss check, which
              must be TODAY'S ACTUAL BALANCE.
        Seeding (b) from (a) on an account at $183,792 produced a phantom
        -$16,208 intraday loss and tripped the daily limit on every cycle.
        The floor still comes from config; equity/day-start now come from the
        live wallet when supplied.
        """
        phase_start = config.initial_balance          # floor reference -- FIXED
        floor = phase_start * (1 - config.max_drawdown_pct)
        eq = current_equity if current_equity is not None else phase_start
        return cls(config=config, phase=1, equity=eq, phase_start_balance=phase_start,
                    peak_equity=max(eq, phase_start), dd_floor_dollars=floor,
                    day_start_equity=eq)

    def _dollar_cushion(self) -> float:
        """The fixed $ distance the floor trails behind the peak, set once
        per phase off that phase's own starting balance."""
        return self.phase_start_balance * self.config.max_drawdown_pct

    def daily_loss_floor_dollars(self) -> float:
        return self.day_start_equity - self.config.daily_loss_limit_dollars

    def start_new_day(self, session_date: dt.date):
        if self.current_date is not None and session_date != self.current_date:
            self.trading_days_count += 1
        self.current_date = session_date
        self.day_start_equity = self.equity

    def update_equity(self, new_equity: float, as_of: "dt.datetime | None" = None) -> dict:
        """
        Call on every live equity mark. Ratchets peak_equity and
        dd_floor_dollars upward as new highs are made, then checks the
        current equity against both the (ratcheted) max-DD floor and the
        daily-loss floor. Returns a dict of the current compliance status;
        raises no exceptions -- breaches are reported as flags, the caller
        (e.g. an execution loop) decides what to do (flatten, halt, alert).
        """
        self.equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            if self.config.drawdown_type == DrawdownType.TRAILING:
                candidate_floor = self.peak_equity - self._dollar_cushion()
                self.dd_floor_dollars = max(self.dd_floor_dollars, candidate_floor)
            # STATIC: peak_equity is still tracked (useful for reporting /
            # diagnostics) but dd_floor_dollars deliberately never moves --
            # it stays at the value start_phase1()/advance_to_phase2() set.

        daily_floor = self.daily_loss_floor_dollars()

        newly_breached = []
        if not self.max_dd_breached and new_equity <= self.dd_floor_dollars:
            self.max_dd_breached = True
            newly_breached.append("max_drawdown")
        if not self.daily_loss_breached and new_equity <= daily_floor:
            self.daily_loss_breached = True
            newly_breached.append("daily_loss_limit")

        for kind in newly_breached:
            self.breach_log.append(dict(kind=kind, equity=new_equity,
                                         dd_floor=self.dd_floor_dollars, daily_floor=daily_floor,
                                         # aware UTC: utcnow() is naive + deprecated in 3.12+,
                                         # and a naive breach timestamp compares wrongly against
                                         # the aware timestamps the orchestrator now produces.
                                         at=as_of or dt.datetime.now(dt.timezone.utc),
                                         phase=self.phase))

        return dict(
            phase=self.phase,
            equity=new_equity,
            peak_equity=self.peak_equity,
            max_dd_floor=self.dd_floor_dollars,
            daily_loss_floor=daily_floor,
            distance_to_max_dd_floor=new_equity - self.dd_floor_dollars,
            distance_to_daily_floor=new_equity - daily_floor,
            max_dd_breached=self.max_dd_breached,
            daily_loss_breached=self.daily_loss_breached,
            busted=self.max_dd_breached or self.daily_loss_breached,
            newly_breached=newly_breached,
            trading_days_count=self.trading_days_count,
            min_trading_days_met=self.trading_days_count >= self.config.min_trading_days,
            drawdown_type=self.config.drawdown_type.value,
            drawdown_floor_mode=self.config.drawdown_floor_mode.value,
        )

    def advance_to_phase2(self):
        """
        Call once when Phase 1 is confirmed passed. Applies the CONFIRMED
        balance/target reset (STRATEGY.md #44a point 1: Phase 2 restarts
        from the original $200k baseline, not Phase 1's ending equity).

        Floor/peak handling depends on drawdown_floor_mode:
          Under drawdown_type == STATIC, drawdown_floor_mode is irrelevant:
        there is no ratcheted floor to carry over, so peak_equity and
        dd_floor_dollars both always reset fresh off the new $200k baseline.

        Under drawdown_type == TRAILING, drawdown_floor_mode decides:
          RESET:      peak_equity and dd_floor_dollars both reset fresh off
                      the new $200k baseline -- Phase 1's peak is forgotten.
          CARRY_OVER: dd_floor_dollars and peak_equity BOTH persist unchanged
                      from Phase 1 -- the floor stays exactly where Phase 1
                      left it, and can only ratchet further if Phase 2
                      equity exceeds Phase 1's own peak.
        Only `equity`, `phase_start_balance`, and `day_start_equity` reset to
        $200k in all cases (that piece is the CONFIRMED part).
        """
        self.phase = 2
        self.equity = self.config.initial_balance
        self.phase_start_balance = self.config.initial_balance
        self.day_start_equity = self.config.initial_balance
        self.max_dd_breached = False
        self.daily_loss_breached = False
        self.trading_days_count = 0
        if self.config.drawdown_type == DrawdownType.STATIC or \
                self.config.drawdown_floor_mode == DrawdownFloorMode.RESET:
            self.peak_equity = self.config.initial_balance
            self.dd_floor_dollars = self.config.initial_balance * (1 - self.config.max_drawdown_pct)
        # TRAILING + CARRY_OVER: peak_equity and dd_floor_dollars deliberately left untouched.
