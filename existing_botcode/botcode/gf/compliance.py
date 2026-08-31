"""
compliance.py — Hyro prop firm rules enforcement
  - Static max DD (90% of initial)
  - Swing daily DD (fixed at start-of-day equity - 5% of initial)
  - Cooldown after losses
  - Circuit breaker on intraday loss
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DrawdownState:
    high_water_mark: float = 0.0
    max_dd_floor: float = 0.0
    day_start_equity: float = 0.0
    daily_floor: float = 0.0
    current_date: str = ""
    today_pnl: float = 0.0
    last_loss_bar: int = -100
    day_frozen: bool = False


class ComplianceEngine:
    def __init__(self, initial_balance, max_dd_pct=10.0, daily_dd_pct=5.0, swing_dd=True,
                 dd_safety_buffer=85.0, daily_safety_buffer=80.0,
                 cooldown_bars=5, circuit_breaker_pct=2.0):
        self.initial_balance = initial_balance
        self.max_dd_pct = max_dd_pct
        self.daily_dd_pct = daily_dd_pct
        self.swing_dd = swing_dd
        self.dd_safety_buffer = dd_safety_buffer / 100.0
        self.daily_safety_buffer = daily_safety_buffer / 100.0
        self.cooldown_bars = cooldown_bars
        self.circuit_breaker_pct = circuit_breaker_pct
        self._equity_cache = initial_balance

        self.state = DrawdownState(
            high_water_mark=initial_balance,
            max_dd_floor=initial_balance * 0.90,  # STATIC at 90% of initial — never moves
            day_start_equity=initial_balance,
            daily_floor=initial_balance - (initial_balance * daily_dd_pct / 100),
        )

    def update_equity(self, equity: float, worst_case_equity: float, current_date: str):
        # New day → reset swing DD floor (fixed for the day)
        if current_date != self.state.current_date:
            self.state.current_date = current_date
            self.state.day_start_equity = equity
            self.state.daily_floor = equity - (self.initial_balance * self.daily_dd_pct / 100)
            self.state.today_pnl = 0.0
            self.state.day_frozen = False
            self.state.last_loss_bar = -100  # Reset cooldown on new day

        # HWM tracks for reporting only — max DD is STATIC at 90% of initial
        if equity > self.state.high_water_mark:
            self.state.high_water_mark = equity

        check_equity = min(equity, worst_case_equity)
        breached = False
        breach_type = None

        if check_equity <= self.state.max_dd_floor:
            breached = True
            breach_type = "MAX_DRAWDOWN"
        if check_equity <= self.state.daily_floor:
            breached = True
            breach_type = "DAILY_DRAWDOWN"

        max_dd_limit = self.initial_balance * self.max_dd_pct / 100
        dd_used_pct = ((self.initial_balance - check_equity) / max_dd_limit * 100) if max_dd_limit > 0 else 0
        if dd_used_pct < 0:
            dd_used_pct = 0

        swing_dd_limit = self.initial_balance * self.daily_dd_pct / 100
        daily_dd_used_pct = ((self.state.day_start_equity - equity) / swing_dd_limit * 100) if swing_dd_limit > 0 else 0
        if daily_dd_used_pct < 0:
            daily_dd_used_pct = 0

        daily_loss_pct = (self.state.day_start_equity - equity) / self.initial_balance * 100
        breaker_tripped_now = False
        if daily_loss_pct >= self.circuit_breaker_pct:
            if not self.state.day_frozen:
                breaker_tripped_now = True   # first bar the breaker fires this day
            self.state.day_frozen = True

        self._equity_cache = equity

        return {
            'breached': breached, 'breach_type': breach_type,
            'dd_used_pct': dd_used_pct, 'daily_dd_used_pct': daily_dd_used_pct,
            'equity': equity, 'hwm': self.state.high_water_mark,
            'day_frozen': self.state.day_frozen, 'breaker_tripped_now': breaker_tripped_now,
        }

    def can_open_position(self, current_bar: int):
        if current_bar - self.state.last_loss_bar < self.cooldown_bars:
            return {'allowed': False, 'reason': f'Cooldown ({self.cooldown_bars - (current_bar - self.state.last_loss_bar)} bars left)'}
        if self.state.day_frozen:
            return {'allowed': False, 'reason': 'Circuit breaker frozen for today'}

        swing_limit = self.initial_balance * self.daily_dd_pct / 100
        swing_used = (self.state.day_start_equity - self._equity_cache) / swing_limit if swing_limit > 0 else 0
        if swing_used >= self.daily_safety_buffer:
            return {'allowed': False, 'reason': f'Swing DD safety ({swing_used*100:.0f}%)'}

        max_limit = self.initial_balance * self.max_dd_pct / 100
        max_used = (self.initial_balance - self._equity_cache) / max_limit if max_limit > 0 else 0
        if max_used >= self.dd_safety_buffer:
            return {'allowed': False, 'reason': f'Max DD safety ({max_used*100:.0f}%)'}

        return {'allowed': True, 'reason': 'OK'}

    def record_loss(self, current_bar: int):
        self.state.last_loss_bar = current_bar

    def record_pnl(self, pnl: float):
        self.state.today_pnl += pnl

    def reset_for_cashout(self):
        """After cashout, equity returns to initial balance — reset DD tracking."""
        self.state.high_water_mark = self.initial_balance
        self.state.max_dd_floor = self.initial_balance * 0.90
        self.state.day_start_equity = self.initial_balance
        self.state.daily_floor = self.initial_balance - (self.initial_balance * self.daily_dd_pct / 100)
        self.state.today_pnl = 0.0
        self.state.day_frozen = False
