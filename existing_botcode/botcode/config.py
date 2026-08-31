"""
HyroTrader Bot — Configuration
Mode: EVAL (sprint to 10% target, then switch to FUNDED)

EVAL CONFIG (current):
    r=0.8%, SL=1.0x, TP1=1.5x, TP2=3.5x
    NO ADX, TOD filter ON (block 16-19 UTC)
    EMA(8/21), ST trail OFF
    15 coins, 2 positions, 5-bar confirm, 5-bar cooldown

FUNDED CONFIG (switch after passing both eval steps):
    r=0.6%, SL=1.0x, TP1=1.5x, TP2=7.0x
    ADX>15, TOD filter ON
    EMA(8/21), ST trail ON
    15 coins, 2 positions, 5-bar confirm, 5-bar cooldown
    3-day cashout, 80% split
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ─── API CREDENTIALS ───
API_KEY = "REDACTED"
API_SECRET = "REDACTED"
BYBIT_API_KEY = "REDACTED"
BYBIT_API_SECRET = "REDACTED"
BYBIT_TESTNET = True

# ─── MODE: 'eval_p1' | 'eval_p2' | 'funded' ───
CURRENT_MODE = "eval_p1"


@dataclass
class AccountConfig:
    initial_balance: float = 200_000.0
    challenge_type: str = "2-step"
    phase1_profit_target_pct: float = 10.0
    phase2_profit_target_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    daily_drawdown_pct: float = 5.0
    swing_daily_dd: bool = True            # $29 upgrade — REQUIRED
    phase1_min_trading_days: int = 10
    phase2_min_trading_days: int = 5
    consistency_rule_pct: float = 40.0
    profit_split: float = 0.80
    cashout_frequency_days: int = 3        # 3-day cashout (was 5)


@dataclass
class ComplianceConfig:
    stop_loss_deadline_seconds: int = 240
    max_risk_per_trade_pct: float = 3.0
    max_exposure_pct: float = 25.0
    daily_reset_utc_hour: int = 0
    daily_dd_safety_buffer_pct: float = 80.0
    max_dd_safety_buffer_pct: float = 85.0
    # TOD filter: block entries during these UTC hours
    block_hours_utc: List[int] = field(default_factory=lambda: [16, 17, 18, 19])


# ─── 15 COINS (added FIL based on backtest analysis) ───
COIN_CONFIGS: Dict[str, Tuple[int, float]] = {
    "ORDIUSDT":   (28, 1.5),
    "ONDOUSDT":   (14, 1.5),
    "ICPUSDT":    (14, 1.5),
    "APTUSDT":    (10, 2.0),
    "AVAXUSDT":   (28, 1.5),
    "LINKUSDT":   (14, 1.5),
    "FILUSDT":    (21, 3.5),
    "ARBUSDT":    (21, 2.5),
    "WLDUSDT":    (14, 1.5),
}


@dataclass
class StrategyConfig:
    symbols: List[str] = field(default_factory=lambda: list(COIN_CONFIGS.keys()))
    timeframe: str = "60"
    confirmation_bars: int = 5
    cooldown_bars: int = 5

    ema_fast_period: int = 8
    ema_slow_period: int = 21
    adx_period: int = 14
    adx_threshold_funded: float = 18.0
    adx_enabled_eval: bool = True
    adx_enabled_funded: bool = True

    # ─── EVAL CONFIG (sprint to 10%) ───
    eval_risk_per_trade_pct: float = 1.0
    eval_sl_atr_multiplier: float = 1.5
    eval_tp1_atr_multiplier: float = 1.5
    eval_tp2_atr_multiplier: float = 7.0
    eval_circuit_breaker_pct: float = 3.0
    eval_use_supertrend_trailing: bool = True
    eval_trail_atr_multiplier: float = 0.0   # post-TP1 ATR trail (0=off) - using ST line instead

    # ─── FUNDED CONFIG (long-term safety) ───
    funded_risk_per_trade_pct: float = 0.60
    funded_sl_atr_multiplier: float = 1.0
    funded_tp1_atr_multiplier: float = 1.5
    funded_tp2_atr_multiplier: float = 7.0
    funded_circuit_breaker_pct: float = 3.0
    funded_use_supertrend_trailing: bool = True

    partial_tp1_pct: float = 75.0
    partial_tp2_pct: float = 50.0
    max_concurrent_positions: int = 2
    leverage: int = 10
    kline_limit: int = 200

    def get_coin_config(self, symbol):
        return COIN_CONFIGS.get(symbol, (21, 2.0))

    def get_risk(self, mode):
        return self.eval_risk_per_trade_pct if mode.startswith("eval") else self.funded_risk_per_trade_pct

    def get_sl(self, mode):
        return self.eval_sl_atr_multiplier if mode.startswith("eval") else self.funded_sl_atr_multiplier

    def get_tp1(self, mode):
        return self.eval_tp1_atr_multiplier if mode.startswith("eval") else self.funded_tp1_atr_multiplier

    def get_tp2(self, mode):
        return self.eval_tp2_atr_multiplier if mode.startswith("eval") else self.funded_tp2_atr_multiplier

    def use_adx(self, mode):
        return self.adx_enabled_funded if mode == "funded" else self.adx_enabled_eval

    def get_adx_threshold(self, mode):
        return self.adx_threshold_funded

    def use_st_trailing(self, mode):
        return self.funded_use_supertrend_trailing if mode == "funded" else self.eval_use_supertrend_trailing

    def get_circuit_breaker(self, mode):
        return self.funded_circuit_breaker_pct if mode == "funded" else self.eval_circuit_breaker_pct


@dataclass
class ModeConfig:
    mode: str = CURRENT_MODE


@dataclass
class MonitoringConfig:
    log_file: str = "hyro_bot.log"
    log_level: str = "INFO"
    state_file: str = "bot_state.json"
    positions_file: str = "positions.json"      # NEW: persist positions across restarts
    discord_webhook: str = "https://discord.com/api/webhooks/REDACTED"
    status_webhook: str = "https://discord.com/api/webhooks/REDACTED"


ACCOUNT = AccountConfig()
COMPLIANCE = ComplianceConfig()
STRATEGY = StrategyConfig()
MODE = ModeConfig()
MONITORING = MonitoringConfig()
