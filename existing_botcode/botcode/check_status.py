"""
check_status.py — Show what bot will see on the next bar.

Uses the SAME initialize() and SAME hold logic as main.py, so output
should match exactly what the running bot has in memory.
"""
import numpy as np
from config import STRATEGY, COMPLIANCE
from strategy import StrategyEngine
from bybit_client import BybitClient
import config

client = BybitClient(config.BYBIT_API_KEY, config.BYBIT_API_SECRET, config.BYBIT_TESTNET)
strategy = StrategyEngine(STRATEGY)

print(f"{'Symbol':<14} {'Trend':<6} {'Hold':>5} {'EMA':>6} {'ADX':>6} {'Status':>10}")
print("-" * 60)

for symbol in STRATEGY.symbols:
    try:
        klines = client.get_klines(symbol, STRATEGY.timeframe, STRATEGY.kline_limit)
        if not klines or len(klines) < 50:
            print(f"{symbol:<14} INSUFFICIENT DATA")
            continue

        # Same as main.py: drop the unclosed candle
        closed = klines[:-1]
        h = np.array([k['high'] for k in closed])
        l = np.array([k['low'] for k in closed])
        c = np.array([k['close'] for k in closed])

        strategy.initialize(symbol, h, l, c)

        eng = strategy.engines.get(symbol)
        hold = strategy.direction_hold.get(symbol, 0)
        ef = strategy.ema_fast_val.get(symbol, 0)
        es = strategy.ema_slow_val.get(symbol, 0)
        adx = strategy.adx_calcs[symbol].value if symbol in strategy.adx_calcs else 0
        trend = "BULL" if eng.trend == 1 else "BEAR"

        ema_ok = "OK" if (eng.trend == 1 and ef > es) or (eng.trend == -1 and ef < es) else "NO"

        if hold == STRATEGY.confirmation_bars and ema_ok == "OK":
            status = "READY"
        elif hold < STRATEGY.confirmation_bars:
            status = f"WAIT({hold})"
        else:
            status = f"OLD({hold})"

        print(f"{symbol:<14} {trend:<6} {hold:>5} {ema_ok:>6} {adx:>5.1f} {status:>10}")

    except Exception as e:
        print(f"{symbol:<14} ERROR: {e}")

print()
print(f"Mode: {config.CURRENT_MODE}")
print(f"TOD blocked hours UTC: {COMPLIANCE.block_hours_utc}")
print(f"Confirmation bars: {STRATEGY.confirmation_bars}")
print(f"Max positions: {STRATEGY.max_concurrent_positions}")
