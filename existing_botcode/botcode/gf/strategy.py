import logging
logger = logging.getLogger("hyrobot")
"""
strategy.py — Clean, single source of truth for all indicators.

Key fixes:
  - ADX is actually computed (was always 0 in v1)
  - Hold counter logic identical between initialize() and feed_new_candle()
  - Single-bar incremental updates only (caller controls when to feed)
  - Returns_supertrend_trailing_sl returns current_sl (was returning current_s typo)
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Signal:
    symbol: str
    side: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    atr: float
    risk_usd: float
    qty: float
    confidence: str = "MEDIUM"


class SupertrendEngine:
    def __init__(self, atr_period: int, multiplier: float):
        self.atr_period = atr_period
        self.multiplier = multiplier
        self._tr_history = []
        self._atr = None
        self._upper = None
        self._lower = None
        self._trend = 0
        self._prev_trend = 0
        self._prev_close = None
        self._ready = False
        self.signal: Optional[str] = None
        self.trend = 0

    def feed(self, h: float, l: float, c: float):
        """Feed a single closed candle. Returns nothing — read self.trend, self.signal."""
        if self._prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))
        self._tr_history.append(tr)
        n = len(self._tr_history)

        if n < self.atr_period:
            self._prev_close = c
            self.signal = None
            return

        if n == self.atr_period:
            self._atr = sum(self._tr_history) / self.atr_period
        else:
            alpha = 1.0 / self.atr_period
            self._atr = self._atr * (1.0 - alpha) + tr * alpha

        hl2 = (h + l) / 2.0
        bu = hl2 + self.multiplier * self._atr
        bl = hl2 - self.multiplier * self._atr

        if not self._ready:
            self._upper = bu
            self._lower = bl
            self._trend = 1
            self._prev_trend = 1
            self._ready = True
            self._prev_close = c
            self.trend = 1
            self.signal = None
            return

        pc = self._prev_close
        if bu < self._upper or pc > self._upper:
            self._upper = bu
        if bl > self._lower or pc < self._lower:
            self._lower = bl

        self._prev_trend = self._trend
        if self._trend == 1:
            if c < self._lower:
                self._trend = -1
        else:
            if c > self._upper:
                self._trend = 1
        self.trend = self._trend

        if self._trend == 1 and self._prev_trend == -1:
            self.signal = "long"
        elif self._trend == -1 and self._prev_trend == 1:
            self.signal = "short"
        else:
            self.signal = None
        self._prev_close = c


class ADXCalculator:
    """Wilder's ADX, incremental."""
    def __init__(self, period: int = 14):
        self.period = period
        self._tr_smooth = 0.0
        self._plus_dm_smooth = 0.0
        self._minus_dm_smooth = 0.0
        self._adx = 0.0
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._dx_history = []
        self._initialized = False
        self._sums = {'tr': 0.0, 'plus_dm': 0.0, 'minus_dm': 0.0, 'count': 0}

    @property
    def value(self) -> float:
        return self._adx

    def feed(self, h: float, l: float, c: float):
        if self._prev_high is None:
            self._prev_high, self._prev_low, self._prev_close = h, l, c
            return

        up = h - self._prev_high
        dn = self._prev_low - l
        plus_dm = up if up > dn and up > 0 else 0
        minus_dm = dn if dn > up and dn > 0 else 0
        tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))

        if not self._initialized:
            self._sums['tr'] += tr
            self._sums['plus_dm'] += plus_dm
            self._sums['minus_dm'] += minus_dm
            self._sums['count'] += 1
            if self._sums['count'] >= self.period:
                self._tr_smooth = self._sums['tr']
                self._plus_dm_smooth = self._sums['plus_dm']
                self._minus_dm_smooth = self._sums['minus_dm']
                self._initialized = True
        else:
            self._tr_smooth = self._tr_smooth - self._tr_smooth/self.period + tr
            self._plus_dm_smooth = self._plus_dm_smooth - self._plus_dm_smooth/self.period + plus_dm
            self._minus_dm_smooth = self._minus_dm_smooth - self._minus_dm_smooth/self.period + minus_dm

            if self._tr_smooth > 0:
                plus_di = self._plus_dm_smooth / self._tr_smooth * 100
                minus_di = self._minus_dm_smooth / self._tr_smooth * 100
                if plus_di + minus_di > 0:
                    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
                    self._dx_history.append(dx)
                    if len(self._dx_history) >= self.period:
                        if self._adx == 0:
                            self._adx = sum(self._dx_history[-self.period:]) / self.period
                        else:
                            self._adx = (self._adx * (self.period - 1) + dx) / self.period

        self._prev_high, self._prev_low, self._prev_close = h, l, c


class StrategyEngine:
    def __init__(self, config):
        self.config = config
        self.engines = {}
        self.adx_calcs = {}
        self.direction_hold = {}
        self.ema_fast_val = {}
        self.ema_slow_val = {}
        self.last_atr = {}
        self.last_close = {}
        self.last_st = {}

    def _calc_initial_ema(self, closes, period):
        valid = [c for c in closes if not np.isnan(c)]
        if len(valid) < period:
            return 0.0
        ema = sum(valid[:period]) / period
        mult = 2.0 / (period + 1)
        for c in valid[period:]:
            ema = (c - ema) * mult + ema
        return ema

    def initialize(self, symbol: str, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
        """One-shot initialization from historical bars. Replays each bar through engines."""
        atr_period, atr_mult = self.config.get_coin_config(symbol)
        eng = SupertrendEngine(atr_period, atr_mult)
        adx_calc = ADXCalculator(self.config.adx_period)

        hold = 0
        last_dir = 0

        for i in range(len(closes)):
            if np.isnan(closes[i]) or np.isnan(highs[i]) or np.isnan(lows[i]):
                continue

            eng.feed(highs[i], lows[i], closes[i])
            adx_calc.feed(highs[i], lows[i], closes[i])

            if eng._ready:
                # IDENTICAL logic to feed_new_candle below
                if eng.trend != last_dir and last_dir != 0:
                    hold = 1
                else:
                    hold += 1
                last_dir = eng.trend

        self.engines[symbol] = eng
        self.adx_calcs[symbol] = adx_calc
        self.direction_hold[symbol] = hold
        self.last_atr[symbol] = eng._atr if eng._atr else 0
        self.last_close[symbol] = float(closes[-1]) if not np.isnan(closes[-1]) else 0
        self.last_st[symbol] = eng._lower if eng.trend == 1 else eng._upper

        ef = self._calc_initial_ema(closes, self.config.ema_fast_period)
        es = self._calc_initial_ema(closes, self.config.ema_slow_period)
        self.ema_fast_val[symbol] = ef
        self.ema_slow_val[symbol] = es

    def feed_new_candle(self, symbol: str, h: float, l: float, c: float):
        """Feed ONE new closed candle. Caller is responsible for not feeding the same bar twice."""
        if symbol not in self.engines:
            return

        eng = self.engines[symbol]
        adx_calc = self.adx_calcs[symbol]

        prev_trend = eng.trend
        eng.feed(h, l, c)
        adx_calc.feed(h, l, c)

        # IDENTICAL hold logic to initialize()
        if eng.trend != prev_trend and prev_trend != 0:
            self.direction_hold[symbol] = 1
        else:
            self.direction_hold[symbol] = self.direction_hold.get(symbol, 0) + 1

        self.last_atr[symbol] = eng._atr if eng._atr else 0
        self.last_close[symbol] = c
        self.last_st[symbol] = eng._lower if eng.trend == 1 else eng._upper

        # Incremental EMA updates
        if symbol in self.ema_fast_val and self.ema_fast_val[symbol] > 0:
            m = 2.0 / (self.config.ema_fast_period + 1)
            self.ema_fast_val[symbol] = (c - self.ema_fast_val[symbol]) * m + self.ema_fast_val[symbol]
        if symbol in self.ema_slow_val and self.ema_slow_val[symbol] > 0:
            m = 2.0 / (self.config.ema_slow_period + 1)
            self.ema_slow_val[symbol] = (c - self.ema_slow_val[symbol]) * m + self.ema_slow_val[symbol]

    def check_entry(self, symbol: str, equity: float, account_balance: float, mode: str) -> Optional[Signal]:
        """Returns Signal if all entry conditions met, else None."""
        if symbol not in self.engines:
            return None

        eng = self.engines[symbol]
        if not eng._ready:
            return None

        close = self.last_close.get(symbol, 0)
        atr = self.last_atr.get(symbol, 0)
        if close == 0 or atr == 0:
            return None

        # Confirmation: exact bar number
        hold = self.direction_hold.get(symbol, 0)
        if not (self.config.confirmation_bars <= hold <= self.config.confirmation_bars + 1):
            return None

        direction = eng.trend
        side = "BUY" if direction == 1 else "SELL"

        # EMA filter
        ef = self.ema_fast_val.get(symbol, 0)
        es = self.ema_slow_val.get(symbol, 0)
        if ef == 0 or es == 0:
            return None
        if side == "BUY" and ef <= es:
            return None
        if side == "SELL" and ef >= es:
            return None

        # ADX filter (only if mode requires it)
        if self.config.use_adx(mode):
            adx_calc = self.adx_calcs.get(symbol)
            if adx_calc is not None:
                threshold = self.config.get_adx_threshold(mode)
                if adx_calc.value > 0 and adx_calc.value <= threshold:
                    return None

        # Position sizing
        risk_pct = self.config.get_risk(mode)
        sl_mult = self.config.get_sl(mode)
        tp1_mult = self.config.get_tp1(mode)
        tp2_mult = self.config.get_tp2(mode)

        sl_dist = atr * sl_mult
        if sl_dist == 0:
            return None

        risk_usd = min(equity * risk_pct / 100, account_balance * 0.03, 1700)
        qty = risk_usd / sl_dist
        # Hard notional cap: never exceed 50% of equity
        notional = qty * close
        if notional > equity * 0.5:
            qty = (equity * 0.5) / close
            logger.info(f"Notional capped for {symbol}: ${notional:.0f} -> ${qty*close:.0f}")

        if side == "BUY":
            sl = close - sl_dist
            tp1 = close + atr * tp1_mult
            tp2 = close + atr * tp2_mult
        else:
            sl = close + sl_dist
            tp1 = close - atr * tp1_mult
            tp2 = close - atr * tp2_mult

        return Signal(
            symbol=symbol, side=side, entry_price=close,
            sl_price=sl, tp1_price=tp1, tp2_price=tp2,
            atr=atr, risk_usd=risk_usd, qty=qty, confidence="MEDIUM"
        )

    def get_supertrend_trailing_sl(self, symbol: str, side: str, current_sl: float) -> float:
        """Returns updated SL based on Supertrend line, never tightens beyond current_sl."""
        st = self.last_st.get(symbol)
        if st is None or st == 0:
            return current_sl
        if side == "BUY" and st > current_sl:
            return st
        if side == "SELL" and st < current_sl:
            return st
        return current_sl
