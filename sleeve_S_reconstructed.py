#!/usr/bin/env python3
"""
SLEEVE S -- RECONSTRUCTION, NOT VERIFICATION.

short_engine_backtest.py (the file sleeve_S() in the reference implementation
imports) does not exist in any uploaded file. Every rule below is built from
the PROSE description in OPTION-1-FINAL-SPEC.pdf Section 2 only. Every
threshold not given a specific number in the prose (swing-low lookback,
Ichimoku periods, delta z-score cutoff, "OI falling over last 3 bars"
definition) is an ASSUMPTION I made, marked below. Nothing here should be
read as reproducing the spec's claimed 0.78 solo Sharpe -- it is an
independent build of what the prose describes, on real data, and whatever
number comes out is what that reconstruction actually does.

Spec rules (verbatim from §2):
  regime:  close < EMA200 AND EMA50 < EMA200                  (bear only)
  trigger: break of prior swing low (BOS) OR Ichimoku cloud breakdown
  confirm: delta z-score below threshold (aggressive selling)
  confirm: OI falling over last 3 bars
  exit:    chandelier trail (slm 1.5, chand 2.0 ATR) -- NO take-profit
  max 6 concurrent shorts, short only, ~33% of days deployed (episodic)

ASSUMPTIONS made to make this runnable (none of these numbers are in the
prose spec -- flagged explicitly, sensitivity-tested below):
  - swing-low lookback for BOS: 20 bars (4h bars -> ~80h / 3.3 days) [ASSUMED]
  - Ichimoku: tenkan=9, kijun=26, senkou_b=52 (standard defaults) [ASSUMED]
  - delta z-score window: 48 bars (8 days of 4h bars), threshold: z < -1.0 [ASSUMED]
  - "OI falling over last 3 bars": OI[t] < OI[t-1] < OI[t-2] < OI[t-3] [ASSUMED interpretation]
  - ATR period: 14 bars [ASSUMED, industry standard for chandelier]
  - position sizing: equal risk per slot, 1/6 of sleeve capital per concurrent short [ASSUMED]
  - entry timing: signal bar close, enter next bar open [ASSUMED]

Timeframe: 4h, resampled from the 1h OHLC in run/hist (spec explicitly says "4h").
OI source: run/oi_raw/{COIN}_oi_1h.csv (actual open-interest series, NOT funding
rate -- oi_data/{COIN}_funding.csv is funding, a different file in the same zip;
using the correct one here).
"""
import numpy as np, pandas as pd, glob, os

HIST_DIR   = 'run/hist'
TAKER_DIR  = 'run/taker'
OI_DIR     = 'run/oi_raw'
EXCLUDE    = ("BTC", "ETHBTC")

SWING_LOOKBACK   = 20     # [ASSUMED]
TENKAN, KIJUN, SENKOU_B = 9, 26, 52   # [ASSUMED, standard Ichimoku]
DELTA_Z_WINDOW   = 48     # [ASSUMED]
DELTA_Z_THRESH   = -1.0   # [ASSUMED]
OI_FALL_BARS     = 3      # from spec prose ("last 3 bars")
ATR_PERIOD       = 14     # [ASSUMED]
CHAND_MULT       = 2.0    # from spec ("chand 2.0 ATR")
SLM_MULT         = 1.5    # from spec ("slm 1.5") -- initial stop, chandelier trail governs exit
MAX_CONCURRENT   = 6       # from spec
FEE              = 0.00055 + 0.0003

def load_4h(coin):
    p = f"{HIST_DIR}/{coin}_1h.csv"
    df = pd.read_csv(p)
    ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
    df['dt'] = pd.to_datetime(df[ts_col], unit='ms', utc=True)
    df = df.set_index('dt')[['open','high','low','close','volume']]
    r = df.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    return r

def load_oi_4h(coin):
    p = f"{OI_DIR}/{coin}_oi_1h.csv"
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('dt')[['open_interest']]
    return df.resample('4h').last().dropna()

def load_taker_4h(coin):
    p = f"{TAKER_DIR}/{coin}_taker_1h.csv"
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('dt')[['delta']]
    return df.resample('4h').sum()

def atr(df, period=ATR_PERIOD):
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def ichimoku_cloud(df):
    h, l = df['high'], df['low']
    tenkan = (h.rolling(TENKAN).max() + l.rolling(TENKAN).min())/2
    kijun  = (h.rolling(KIJUN).max()  + l.rolling(KIJUN).min())/2
    span_a = ((tenkan + kijun)/2).shift(KIJUN)
    span_b = ((h.rolling(SENKOU_B).max() + l.rolling(SENKOU_B).min())/2).shift(KIJUN)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    return cloud_bottom

def build_signals(coin):
    """Returns a DataFrame with columns: close, short_ok, atr, for backtest engine.
    short_ok[t] = True if regime+trigger+confirms are satisfied at bar close t
    (signal known at close of t, tradeable from open of t+1 -- causal)."""
    px = load_4h(coin)
    if len(px) < SENKOU_B + KIJUN + 20:
        return None
    oi = load_oi_4h(coin)
    tk = load_taker_4h(coin)
    if oi is None or tk is None:
        return None
    df = px.copy()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema50']  = df['close'].ewm(span=50,  adjust=False).mean()
    df['atr']    = atr(df)
    df['swing_low'] = df['low'].shift(1).rolling(SWING_LOOKBACK).min()
    df['cloud_bottom'] = ichimoku_cloud(df)
    df = df.join(oi.reindex(df.index, method='ffill'), how='left')
    df = df.join(tk.reindex(df.index, method='ffill'), how='left')
    df['delta_z'] = (df['delta'] - df['delta'].rolling(DELTA_Z_WINDOW).mean()) / df['delta'].rolling(DELTA_Z_WINDOW).std()
    df['oi_falling'] = df['open_interest'] < df['open_interest'].shift(OI_FALL_BARS)
    for k in range(1, OI_FALL_BARS):
        df['oi_falling'] &= df['open_interest'].shift(k) < df['open_interest'].shift(k+1)

    regime  = (df['close'] < df['ema200']) & (df['ema50'] < df['ema200'])
    trigger = (df['close'] < df['swing_low']) | (df['close'] < df['cloud_bottom'])
    confirm = (df['delta_z'] < DELTA_Z_THRESH) & df['oi_falling']
    df['short_ok'] = regime & trigger & confirm
    return df[['close','high','low','atr','short_ok']].dropna(subset=['atr'])

def backtest_sleeve_S(coins, start=None, end=None):
    """Event-driven backtest: at most MAX_CONCURRENT concurrent shorts,
    chandelier exit (highest high since entry - CHAND_MULT*ATR), no take-profit.
    Returns a daily P&L series (fraction of sleeve capital, dense across all days
    -- zero-filled on non-trading days per spec's own equal-risk blending rule)."""
    all_sig = {}
    for c in coins:
        s = build_signals(c)
        if s is not None:
            if start: s = s[s.index >= start]
            if end:   s = s[s.index <= end]
            all_sig[c] = s
    if not all_sig:
        return pd.Series(dtype=float)

    idx = sorted(set().union(*[s.index for s in all_sig.values()]))
    open_positions = {}   # coin -> dict(entry_price, highest_high, entry_time)
    pnl_4h = pd.Series(0.0, index=idx)
    per_slot_risk = 1.0 / MAX_CONCURRENT

    for t in idx:
        step_pnl = 0.0
        to_close = []
        for c, pos in open_positions.items():
            if t not in all_sig[c].index: continue
            row = all_sig[c].loc[t]
            pos['lowest_close_high'] = min(pos.get('lowest_close_high', row['high']), row['high'])
            chand_stop = pos['lowest_close_high'] + CHAND_MULT * row['atr']
            bar_ret = (pos['prev_close'] - row['close']) / pos['entry_price']
            step_pnl += per_slot_risk * bar_ret
            pos['prev_close'] = row['close']
            if row['high'] >= chand_stop:
                to_close.append(c)
        for c in to_close:
            del open_positions[c]

        if len(open_positions) < MAX_CONCURRENT:
            for c, s in all_sig.items():
                if len(open_positions) >= MAX_CONCURRENT: break
                if c in open_positions: continue
                if t not in s.index: continue
                row = s.loc[t]
                if row['short_ok']:
                    open_positions[c] = {'entry_price': row['close'], 'prev_close': row['close'],
                                          'lowest_close_high': row['high']}
                    step_pnl -= per_slot_risk * FEE
        pnl_4h[t] = step_pnl

    daily = pnl_4h.groupby(pnl_4h.index.date).sum()
    daily.index = pd.to_datetime(daily.index)
    return daily

if __name__ == '__main__':
    coins = sorted(os.path.basename(f).replace('_1h.csv','')
                    for f in glob.glob(f"{HIST_DIR}/*_1h.csv"))
    coins = [c for c in coins if c not in EXCLUDE]
    print(f"reconstructing sleeve S on {len(coins)} coins (4h bars)...")
    daily = backtest_sleeve_S(coins, start='2025-01-01', end='2026-07-25')
    print(f"days: {len(daily)}, days with nonzero pnl (deployed): {(daily!=0).sum()} "
          f"({(daily!=0).mean()*100:.0f}%)")
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0.0
    print(f"sleeve S (reconstructed) solo Sharpe, clean 18mo window: {sh:.3f}")
    daily.to_csv('sleeve_S_daily.csv', header=['pnl'])
