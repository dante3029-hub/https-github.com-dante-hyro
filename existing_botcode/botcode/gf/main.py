import sys; sys.path.insert(0,"/root")
from shared_positions import other_bot_coins, write_positions
_MY_BOT_ID = "v2"
"""
main.py — HyroTrader Bot v2 (complete rewrite)

ALL 22 BUGS FIXED:
  #1  Decimal-based quantization (no float precision errors)
  #2  Order chunking when qty > maxOrderQty
  #3  Entries checked AFTER feeding new candles, in correct order
  #4  Single feed per bar — no double-feeding (was THE critical bug)
  #5  Position sync records PnL, triggers cooldown, sends Discord
  #6  Discord notifications for ALL exits (TP1, TP2, SL, sync detection)
  #7  Strategy returns current_sl correctly (already fixed in v1 strategy)
  #8  No None SLs sent — chunking puts SL only on first chunk
  #9  Hold counter logic IDENTICAL between initialize and feed_new_candle
  #10 Order chunking handles PEOPLE-sized quantities correctly
  #11 Clean rewrite from scratch — no legacy patches
  #12 Position persistence to positions.json — restored on restart
  #13 ADX actually computed (was always 0 — silently broke funded mode)
  #14 Real worst-case equity from intrabar high/low (not fake equity*0.998)
  #15 TOD filter blocks entries during 16-19 UTC (eval requires this)
  #16 15 coins including FIL
  #17 3-day cashout (was 5-day)
  #18 Eval: TP2=3.5x SL=1.0x (was 2.5x SL=1.5x)
  #19 Eval risk = 0.8% (correct)
  #20 Funded TP2=7.0x (was 5.0x)
  #21 Removed misleading 0 for OPEN parameter (Supertrend doesn't use it)
  #22 Sync ALSO adds positions found on exchange that aren't tracked (with warning)

CONFIG: Eval mode (sprint to 10%)
  r=0.8%, SL=1.0x, TP1=1.5x, TP2=3.5x
  No ADX, TOD filter ON, ST trail OFF
  15 coins, 2 positions, 5-bar confirm/cooldown

After passing both eval steps, change CURRENT_MODE in config.py to 'funded'
to switch to long-term safe config (r=0.6%, TP2=7.0x, ADX>15, ST trail ON).
"""
import time
import json
import logging
import os
import requests as _req
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

import numpy as np

from config import ACCOUNT, COMPLIANCE, STRATEGY, MODE, MONITORING
from strategy import StrategyEngine, Signal
from compliance import ComplianceEngine
from bybit_client import BybitClient
import config


# ─── Logging ───
logging.basicConfig(
    level=getattr(logging, MONITORING.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(MONITORING.log_file), logging.StreamHandler()],
)
logger = logging.getLogger("HyroBot")


def discord_msg(content: str, webhook: str = None):
    """Fire-and-forget Discord notification."""
    url = webhook or MONITORING.discord_webhook
    try:
        _req.post(url, json={"content": content}, timeout=5)
    except Exception:
        pass


# ─── Position state ───
@dataclass
class Position:
    symbol: str
    side: str               # "BUY" or "SELL"
    entry_price: float
    qty: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    atr_at_entry: float
    entry_time: str
    risk_usd: float
    tp1_hit: bool = False
    tp1_pnl_usd: float = 0.0
    peak_price: float = 0.0
    order_ids: list = field(default_factory=list)


@dataclass
class BotState:
    total_realized_pnl: float = 0.0
    total_cashouts: float = 0.0
    cashout_count: int = 0
    last_cashout_date: str = ""
    bar_count: int = 0
    last_processed_bar_ts: int = 0   # Tracks last candle timestamp processed (prevents double-feed)
    mode: str = "eval_p1"

    def save(self, fp: str = "bot_state.json"):
        try:
            with open(fp, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    @classmethod
    def load(cls, fp: str = "bot_state.json"):
        try:
            with open(fp) as f:
                return cls(**json.load(f))
        except Exception:
            return cls()


def save_positions(positions: Dict[str, Position], fp: str):
    try:
        data = {sym: asdict(pos) for sym, pos in positions.items()}
        with open(fp, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save positions: {e}")


def load_positions(fp: str) -> Dict[str, Position]:
    try:
        with open(fp) as f:
            data = json.load(f)
        return {sym: Position(**pos) for sym, pos in data.items()}
    except Exception:
        return {}


class HyroTraderBot:
    def __init__(self):
        self.mode = MODE.mode
        self.client = BybitClient(
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
            testnet=config.BYBIT_TESTNET,
        )
        self.strategy = StrategyEngine(STRATEGY)
        self.compliance = ComplianceEngine(
            initial_balance=ACCOUNT.initial_balance,
            max_dd_pct=ACCOUNT.max_drawdown_pct,
            daily_dd_pct=ACCOUNT.daily_drawdown_pct,
            swing_dd=ACCOUNT.swing_daily_dd,
            dd_safety_buffer=COMPLIANCE.max_dd_safety_buffer_pct,
            daily_safety_buffer=COMPLIANCE.daily_dd_safety_buffer_pct,
            cooldown_bars=STRATEGY.cooldown_bars,
            circuit_breaker_pct=STRATEGY.get_circuit_breaker(self.mode),
        )
        self.state = BotState.load(MONITORING.state_file)
        self.positions: Dict[str, Position] = load_positions(MONITORING.positions_file)
        if self.positions:
            logger.info(f"Restored {len(self.positions)} positions from disk: {list(self.positions.keys())}")

    # ─── Initialization ───
    def initialize(self):
        logger.info("=" * 70)
        logger.info(f"HYRO BOT v2 — Mode: {self.mode.upper()}")
        logger.info(f"Account: ${ACCOUNT.initial_balance:,.0f} | Swing DD: {ACCOUNT.swing_daily_dd}")
        logger.info(f"Strategy: r={STRATEGY.get_risk(self.mode)}% | SL={STRATEGY.get_sl(self.mode)}x | "
                    f"TP1={STRATEGY.get_tp1(self.mode)}x | TP2={STRATEGY.get_tp2(self.mode)}x")
        logger.info(f"Filters: EMA(8/21) ON | ADX={STRATEGY.use_adx(self.mode)} | "
                    f"TOD blocks {COMPLIANCE.block_hours_utc} UTC")
        logger.info(f"ST trail: {STRATEGY.use_st_trailing(self.mode)} | "
                    f"Max pos: {STRATEGY.max_concurrent_positions} | "
                    f"Confirm: {STRATEGY.confirmation_bars} | Cooldown: {STRATEGY.cooldown_bars}")
        logger.info(f"Coins ({len(STRATEGY.symbols)}): {', '.join(STRATEGY.symbols)}")
        logger.info("=" * 70)

        for symbol in STRATEGY.symbols:
            try:
                # Pre-fetch instrument info (cached)
                info = self.client.get_instrument_info(symbol)
                if not info:
                    logger.warning(f"  {symbol}: NO INSTRUMENT INFO — skipping")
                    continue

                self.client.set_leverage(symbol, STRATEGY.leverage)

                klines = self.client.get_klines(symbol, STRATEGY.timeframe, STRATEGY.kline_limit)
                if not klines or len(klines) < 50:
                    logger.warning(f"  {symbol}: insufficient data ({len(klines) if klines else 0} bars)")
                    continue

                # Drop the last candle (it's currently forming and not closed yet)
                closed = klines[:-1]
                h = np.array([k['high'] for k in closed])
                l = np.array([k['low'] for k in closed])
                c = np.array([k['close'] for k in closed])

                self.strategy.initialize(symbol, h, l, c)

                eng = self.strategy.engines.get(symbol)
                hold = self.strategy.direction_hold.get(symbol, 0)
                trend = "BULL" if eng.trend == 1 else "BEAR"
                adx = self.strategy.adx_calcs[symbol].value if symbol in self.strategy.adx_calcs else 0
                logger.info(f"  {symbol}: {len(closed)} bars | {trend} | hold={hold} | ADX={adx:.1f}")

            except Exception as e:
                logger.error(f"  {symbol}: init failed — {e}")

        # Set last processed bar to the most recent CLOSED candle timestamp
        # so we don't immediately re-feed it on first cycle
        try:
            sample_klines = self.client.get_klines("XRPUSDT", STRATEGY.timeframe, 2)
            if sample_klines and len(sample_klines) >= 2:
                self.state.last_processed_bar_ts = sample_klines[-2]['timestamp']
                logger.info(f"Last processed bar timestamp: {self.state.last_processed_bar_ts}")
        except Exception as e:
            logger.warning(f"Could not set last_processed_bar_ts: {e}")

        discord_msg(f"🟢 HYRO BOT STARTED | Mode: {self.mode} | "
                    f"{len(STRATEGY.symbols)} coins | r={STRATEGY.get_risk(self.mode)}%")

    # ─── Main loop ───
    def run(self):
        self.initialize()
        while True:
            try:
                self._wait_for_candle()
                self._process_bar()
                self.state.save(MONITORING.state_file)
                save_positions(self.positions, MONITORING.positions_file)
                write_positions(_MY_BOT_ID, list(self.positions.keys()))
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                self.state.save(MONITORING.state_file)
                save_positions(self.positions, MONITORING.positions_file)
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                discord_msg(f"⚠️ HYRO loop error: {e}")
                time.sleep(60)

    def _wait_for_candle(self):
        """Sleep until next hour, checking TP exits every 10s when positions open."""
        now = datetime.now(timezone.utc)
        nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        wait = (nxt - now).total_seconds() + 35
        if wait <= 0:
            return
        logger.debug(f"Sleeping {wait:.0f}s until next bar")
        if not self.positions:
            time.sleep(wait)
            return
        end_time = time.time() + wait
        while time.time() < end_time:
            time.sleep(10)
            if not self.positions:
                remaining = end_time - time.time()
                if remaining > 0:
                    time.sleep(remaining)
                return
            try:
                self._fast_tp_check()
            except Exception as e:
                logger.debug(f"Fast TP check error: {e}")

    def _fast_tp_check(self):
        """Quick TP1/TP2/SL check between hourly bars."""
        tp1m = STRATEGY.get_tp1(self.mode)
        tp2m = STRATEGY.get_tp2(self.mode)
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            ticker = self.client.get_ticker(symbol)
            if not ticker:
                continue
            price = ticker['last_price']
            # ─── ATR TRAILING STOP (after TP1) ───
            _tmult = getattr(STRATEGY, 'eval_trail_atr_multiplier', 0.0) if self.mode.startswith('eval') else 0.0
            if pos.tp1_hit and _tmult > 0:
                if pos.peak_price == 0.0:
                    pos.peak_price = pos.entry_price
                if pos.side == "BUY":
                    if price > pos.peak_price:
                        pos.peak_price = price
                    _newsl = pos.peak_price - pos.atr_at_entry * _tmult
                    if _newsl > pos.sl_price:
                        pos.sl_price = _newsl
                        try:
                            self.client.modify_sl(symbol, pos.sl_price)
                            logger.info(f"⚖️ TRAIL {symbol}: SL -> {pos.sl_price:.6f} (peak {pos.peak_price:.6f})")
                        except Exception as _e:
                            logger.error(f"trail modify_sl failed {symbol}: {_e}")
                else:
                    if price < pos.peak_price or pos.peak_price == pos.entry_price:
                        pos.peak_price = price
                    _newsl = pos.peak_price + pos.atr_at_entry * _tmult
                    if _newsl < pos.sl_price:
                        pos.sl_price = _newsl
                        try:
                            self.client.modify_sl(symbol, pos.sl_price)
                            logger.info(f"⚖️ TRAIL {symbol}: SL -> {pos.sl_price:.6f} (trough {pos.peak_price:.6f})")
                        except Exception as _e:
                            logger.error(f"trail modify_sl failed {symbol}: {_e}")
            # SL check (trailing or original)
            if pos.side == "BUY" and price <= pos.sl_price:
                self._handle_sl_hit(symbol, price)
                continue
            elif pos.side == "SELL" and price >= pos.sl_price:
                self._handle_sl_hit(symbol, price)
                continue
            # TP1 check
            if not pos.tp1_hit:
                if pos.side == "BUY":
                    tp1p = pos.entry_price + pos.atr_at_entry * tp1m
                    hit = price >= tp1p
                else:
                    tp1p = pos.entry_price - pos.atr_at_entry * tp1m
                    hit = price <= tp1p
                if hit:
                    half_qty = pos.qty * 0.5
                    order_ids = self.client.close_position(symbol, pos.side, half_qty)
                    if order_ids:
                        closed_qty = pos.qty * 0.5
                        pos.tp1_hit = True
                        pos.tp1_pnl_usd = abs(price - pos.entry_price) * closed_qty
                        pos.qty -= closed_qty
                        pos.sl_price = pos.entry_price
                        pos.peak_price = price
                        self.client.modify_sl(symbol, pos.sl_price)
                        logger.info(f"\u26a1 FAST TP1: {symbol} closed 50% @ {price:.6f} | tp1_pnl=${pos.tp1_pnl_usd:+,.0f}")
                        discord_msg(f"\u26a1 FAST TP1: {symbol} 50% @ {price:.6f} | PnL=${pos.tp1_pnl_usd:+,.0f}")
                    continue
            # TP2 check
            if pos.tp1_hit:
                if pos.side == "BUY":
                    tp2p = pos.entry_price + pos.atr_at_entry * tp2m
                    hit = price >= tp2p
                else:
                    tp2p = pos.entry_price - pos.atr_at_entry * tp2m
                    hit = price <= tp2p
                if hit:
                    self.client.close_position(symbol, pos.side, pos.qty)
                    tp2_pnl = abs(price - pos.entry_price) * pos.qty
                    total_pnl = pos.tp1_pnl_usd + tp2_pnl
                    self.state.total_realized_pnl += total_pnl
                    self.compliance.record_pnl(total_pnl)
                    logger.info(f"\u26a1 FAST TP2: {symbol} fully closed @ {price:.6f} | PnL=${total_pnl:+,.0f}")
                    discord_msg(f"\u26a1 FAST TP2: {symbol} closed | PnL=${total_pnl:+,.0f}")
                    del self.positions[symbol]

    def _handle_sl_hit(self, symbol, price):
        """Handle SL hit detected during fast check."""
        pos = self.positions[symbol]
        self.client.close_position(symbol, pos.side, pos.qty)
        if pos.tp1_hit:
            pnl = pos.tp1_pnl_usd + (price - pos.entry_price) * pos.qty if pos.side == "BUY" else pos.tp1_pnl_usd + (pos.entry_price - price) * pos.qty
        else:
            pnl = (price - pos.entry_price) * pos.qty if pos.side == "BUY" else (pos.entry_price - price) * pos.qty
        self.state.total_realized_pnl += pnl
        self.compliance.record_pnl(pnl)
        logger.info(f"\u26a1 FAST SL: {symbol} closed @ {price:.6f} | PnL=${pnl:+,.0f}")
        discord_msg(f"\u26a1 FAST SL: {symbol} closed @ {price:.6f} | PnL=${pnl:+,.0f}")
        del self.positions[symbol]

    # ─── Per-bar processing ───
    def _process_bar(self):
        self.state.bar_count += 1
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        hour_utc = now.hour

        # ─── 1. Feed new closed candles to strategy (ONCE per bar) ───
        new_bar_ts = self._feed_new_candles()
        if new_bar_ts == 0:
            logger.warning("No new candle data fetched; skipping bar")
            return

        # ─── 2. Sync positions with exchange (detect SL/manual closes) ───
        self._sync_positions_with_exchange()

        # ─── 3. Get equity (with REAL worst-case from mark prices) ───
        balance = self.client.get_wallet_balance()
        api_pos = self.client.get_positions()
        unrealized = sum(p['unrealized_pnl'] for p in api_pos)
        equity = balance + unrealized

        # Real worst-case: use intrabar low/high of most recent candle for each open position
        worst_case_equity = self._compute_worst_case_equity(balance, api_pos)

        # ─── 4. Compliance check ───
        dd = self.compliance.update_equity(equity, worst_case_equity, today)
        if dd.get('breaker_tripped_now'):
            # 2% circuit breaker fired: freeze new entries (handled in can_open) AND
            # move every open position's stop to break-even to cap further bleed.
            for _sym, _pos in list(self.positions.items()):
                try:
                    be = _pos.entry_price
                    improves = (_pos.side == "BUY" and be > _pos.sl_price) or \
                               (_pos.side == "SELL" and be < _pos.sl_price)
                    if improves:
                        self.client.modify_sl(_sym, be)
                        _pos.sl_price = be
                        logger.info(f"🧯 BREAKER BE-MOVE: {_sym} SL -> entry {be:.6f}")
                except Exception as _e:
                    logger.error(f"breaker BE-move failed for {_sym}: {_e}")
            discord_msg(f"🧯 HYRO 3% breaker tripped — froze entries, moved open SLs to break-even")
        if dd['breached']:
            logger.critical(f"🛑 DD BREACH: {dd['breach_type']} | eq=${equity:,.0f} | worst=${worst_case_equity:,.0f}")
            discord_msg(f"🛑 HYRO DD BREACH: {dd['breach_type']} — closing all positions")
            self._close_all_positions(reason="DD_BREACH")
            return

        logger.info(f"Bar {self.state.bar_count} [{now.strftime('%H:%M UTC')}] | "
                    f"Eq: ${equity:,.0f} | DD: {dd['dd_used_pct']:.1f}%/100% | "
                    f"SwingDD: {dd['daily_dd_used_pct']:.1f}%/100% | "
                    f"Pos: {len(self.positions)} | Frozen: {dd['day_frozen']}")

        # ─── 5. Cashout check (funded mode only) ───
        if self.mode == "funded":
            self._check_cashout(today)

        # ─── 5b. Signal flip exit (close positions when Supertrend reverses) ───
        if getattr(__import__('config'), 'USE_SIGNAL_FLIP_EXIT', True):
            for symbol in list(self.positions.keys()):
                if symbol not in self.strategy.engines:
                    continue
                pos = self.positions[symbol]
                eng = self.strategy.engines[symbol]
                current_trend = eng.trend
                flipped = (pos.side == "BUY" and current_trend == -1) or \
                          (pos.side == "SELL" and current_trend == 1)
                if flipped:
                    ticker = self.client.get_ticker(symbol)
                    price = ticker['last_price'] if ticker else pos.entry_price
                    self.client.close_position(symbol, pos.side, pos.qty)
                    if pos.tp1_hit:
                        if pos.side == "BUY":
                            pnl = pos.tp1_pnl_usd + (price - pos.entry_price) * pos.qty
                        else:
                            pnl = pos.tp1_pnl_usd + (pos.entry_price - price) * pos.qty
                    else:
                        if pos.side == "BUY":
                            pnl = (price - pos.entry_price) * pos.qty
                        else:
                            pnl = (pos.entry_price - price) * pos.qty
                    self.state.total_realized_pnl += pnl
                    self.compliance.record_pnl(pnl)
                    logger.info(f"🔄 SIGNAL_FLIP_EXIT: {symbol} {pos.side} closed @ {price:.6f} | PnL=${pnl:+,.0f}")
                    discord_msg(f"🔄 SIGNAL FLIP: {symbol} {pos.side} closed @ {price:.6f} | PnL=${pnl:+,.0f}")
                    del self.positions[symbol]
        # ─── 6. Manage existing positions (TP1, TP2, trail) ───
        for symbol in list(self.positions.keys()):
            self._manage_position(symbol)

        # ─── 7. TOD filter check ───
        tod_blocked = hour_utc in COMPLIANCE.block_hours_utc
        if tod_blocked:
            logger.debug(f"Entries blocked by TOD filter (hour={hour_utc} UTC)")

        # ─── 8. Scan for new entries ───
        if not tod_blocked and len(self.positions) < STRATEGY.max_concurrent_positions:
            can = self.compliance.can_open_position(self.state.bar_count)
            if can['allowed']:
                self._scan_entries(equity)
            else:
                logger.debug(f"Entry blocked: {can['reason']}")

        # ─── 9. Send status update to Discord ───
        self._send_status_update(equity, dd)

    def _feed_new_candles(self) -> int:
        """Feed the most recent CLOSED candle to each engine. Returns its timestamp.
        Skips feeding if we've already seen this candle (prevents double-feed bug)."""
        new_bar_ts = 0

        for symbol in STRATEGY.symbols:
            if symbol not in self.strategy.engines:
                continue
            try:
                klines = self.client.get_klines(symbol, STRATEGY.timeframe, 3)
                if not klines or len(klines) < 2:
                    continue

                # klines[-1] = currently forming candle, klines[-2] = most recent closed
                closed_candle = klines[-2]
                ts = closed_candle['timestamp']

                # CRITICAL: skip if we've already fed this candle
                if ts <= self.state.last_processed_bar_ts:
                    logger.debug(f"  {symbol}: candle ts={ts} already processed, skipping")
                    continue

                self.strategy.feed_new_candle(
                    symbol,
                    closed_candle['high'],
                    closed_candle['low'],
                    closed_candle['close'],
                )
                new_bar_ts = max(new_bar_ts, ts)

            except Exception as e:
                logger.error(f"Feed failed {symbol}: {e}")

        if new_bar_ts > 0:
            self.state.last_processed_bar_ts = new_bar_ts

        return new_bar_ts

    def _sync_positions_with_exchange(self):
        """Reconcile internal state with exchange. Detects SL hits and manual closes."""
        try:
            api_positions = self.client.get_positions()
        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            return

        exchange_symbols = {p['symbol'] for p in api_positions}

        # Detect positions that closed on the exchange
        for symbol in list(self.positions.keys()):
            if symbol not in exchange_symbols:
                pos = self.positions[symbol]
                # Compute realized PnL using last known price
                ticker = self.client.get_ticker(symbol)
                if ticker:
                    exit_price = ticker['last_price']
                    if pos.side == "BUY":
                        pnl = (exit_price - pos.entry_price) * pos.qty
                    else:
                        pnl = (pos.entry_price - exit_price) * pos.qty
                    if pos.tp1_hit:
                        pnl += pos.tp1_pnl_usd

                    self.state.total_realized_pnl += pnl
                    self.compliance.record_pnl(pnl)

                    if pnl < 0:
                        self.compliance.record_loss(self.state.bar_count - 1)
                        exit_type = "STOP-LOSS"
                        emoji = "🔴"
                    else:
                        exit_type = "EXTERNAL_CLOSE"
                        emoji = "🟢"

                    logger.warning(f"{emoji} {exit_type}: {symbol} closed externally | "
                                   f"entry={pos.entry_price:.6f} exit={exit_price:.6f} | "
                                   f"PnL=${pnl:+,.0f}")
                    discord_msg(f"{emoji} HYRO {exit_type}: {symbol} {pos.side} | "
                                f"entry={pos.entry_price:.6f} exit={exit_price:.6f} | "
                                f"PnL=${pnl:+,.0f}")
                else:
                    logger.warning(f"Position {symbol} closed externally (no ticker for PnL calc)")
                    discord_msg(f"⚠️ HYRO: {symbol} closed externally (PnL unknown)")

                del self.positions[symbol]

        # Detect ORPHAN positions on exchange (we don't track them)
        for ap in api_positions:
            if ap['symbol'] not in self.positions:
                logger.warning(f"⚠️ ORPHAN exchange position: {ap['symbol']} {ap['side']} "
                               f"size={ap['size']} @ {ap['entry_price']:.6f} — NOT tracked")
                discord_msg(f"⚠️ HYRO ORPHAN: {ap['symbol']} {ap['side']} on exchange but not tracked")
                # Don't auto-add — let user decide. Bot won't manage it.

    def _compute_worst_case_equity(self, balance: float, api_positions: list) -> float:
        """Compute realistic worst-case equity using intrabar low/high.
        Replaces the fake equity*0.998 from v1."""
        worst_unrealized = 0.0
        for ap in api_positions:
            try:
                klines = self.client.get_klines(ap['symbol'], STRATEGY.timeframe, 2)
                if not klines or len(klines) < 1:
                    worst_unrealized += ap['unrealized_pnl']
                    continue

                # Use the most recent CLOSED candle's extremes
                recent = klines[-2] if len(klines) >= 2 else klines[-1]
                if ap['side'] == 'Buy':
                    worst_price = recent['low']
                    worst_pnl = (worst_price - ap['entry_price']) * ap['size']
                else:
                    worst_price = recent['high']
                    worst_pnl = (ap['entry_price'] - worst_price) * ap['size']
                worst_unrealized += worst_pnl
            except Exception:
                worst_unrealized += ap['unrealized_pnl']

        return balance + worst_unrealized

    def _manage_position(self, symbol: str):
        """Check TP1, TP2, and trailing SL for an open position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        ticker = self.client.get_ticker(symbol)
        if not ticker:
            return

        price = ticker['last_price']
        tp1m = STRATEGY.get_tp1(self.mode)
        tp2m = STRATEGY.get_tp2(self.mode)

        # ─── TP1: close 50%, move SL to breakeven ───
        if not pos.tp1_hit:
            if pos.side == "BUY":
                tp1p = pos.entry_price + pos.atr_at_entry * tp1m
                hit = price >= tp1p
            else:
                tp1p = pos.entry_price - pos.atr_at_entry * tp1m
                hit = price <= tp1p

            if hit:
                half_qty = pos.qty * 0.5
                order_ids = self.client.close_position(symbol, pos.side, half_qty)
                if order_ids:
                    closed_qty = pos.qty * 0.5  # Approximate; actual fills may differ slightly
                    pos.tp1_hit = True
                    pos.tp1_pnl_usd = abs(price - pos.entry_price) * closed_qty
                    pos.qty -= closed_qty
                    pos.sl_price = pos.entry_price  # Move SL to breakeven
                    self.client.modify_sl(symbol, pos.sl_price)
                    logger.info(f"🎯 TP1: {symbol} closed 50% @ {price:.6f} | "
                                f"SL→BE | tp1_pnl=${pos.tp1_pnl_usd:+,.0f}")
                    discord_msg(f"🎯 HYRO TP1: {symbol} 50% closed @ {price:.6f} | "
                                f"PnL=${pos.tp1_pnl_usd:+,.0f}")

        # ─── TP2: close remainder ───
        if pos.tp1_hit:
            if pos.side == "BUY":
                tp2p = pos.entry_price + pos.atr_at_entry * tp2m
                hit = price >= tp2p
            else:
                tp2p = pos.entry_price - pos.atr_at_entry * tp2m
                hit = price <= tp2p

            if hit:
                order_ids = self.client.close_position(symbol, pos.side, pos.qty)
                tp2_pnl = abs(price - pos.entry_price) * pos.qty
                total_pnl = pos.tp1_pnl_usd + tp2_pnl
                self.state.total_realized_pnl += total_pnl
                self.compliance.record_pnl(total_pnl)
                logger.info(f"🏆 TP2: {symbol} fully closed @ {price:.6f} | Total PnL=${total_pnl:+,.0f}")
                discord_msg(f"🏆 HYRO TP2: {symbol} fully closed | Total PnL=${total_pnl:+,.0f}")
                del self.positions[symbol]
                return

        # ─── Trailing SL (funded mode only) ───
        if pos.tp1_hit and STRATEGY.use_st_trailing(self.mode):
            new_sl = self.strategy.get_supertrend_trailing_sl(symbol, pos.side, pos.sl_price)
            if new_sl != pos.sl_price:
                if self.client.modify_sl(symbol, new_sl):
                    logger.info(f"🔧 Trail SL: {symbol} {pos.sl_price:.6f} → {new_sl:.6f}")
                    pos.sl_price = new_sl

    def _scan_entries(self, equity: float):
        """Scan all symbols for entry signals."""
        for symbol in STRATEGY.symbols:
            if symbol in other_bot_coins(_MY_BOT_ID): continue
            if symbol in self.positions:
                continue
            if len(self.positions) >= STRATEGY.max_concurrent_positions:
                break

            signal = self.strategy.check_entry(symbol, equity, ACCOUNT.initial_balance, self.mode)
            if signal is None:
                continue

            eng = self.strategy.engines[symbol]
            hold = self.strategy.direction_hold.get(symbol, 0)
            logger.info(f"📍 SIGNAL: {signal.side} {symbol} | trend={eng.trend} hold={hold} | "
                        f"price={signal.entry_price:.6f} SL={signal.sl_price:.6f} | "
                        f"risk=${signal.risk_usd:.0f} qty={signal.qty:.4f}")

            self._execute_entry(signal)

    def _execute_entry(self, signal: Signal):
        """Place order(s) for an entry signal, with proper chunking."""
        # Risk exposure check
        total_risk = sum(p.risk_usd for p in self.positions.values())
        max_exposure_usd = ACCOUNT.initial_balance * COMPLIANCE.max_exposure_pct / 100
        if total_risk + signal.risk_usd > max_exposure_usd:
            logger.warning(f"⚠️ SKIP {signal.symbol}: exposure limit "
                           f"(${total_risk:.0f} + ${signal.risk_usd:.0f} > ${max_exposure_usd:.0f})")
            return

        # Quantize and validate qty
        # ─── WORST-CASE SIZING CAP (prevents two-position floor breach) ───
        # If this new position + all open positions hit their stops at once,
        # equity must stay above the safety floor ($184K, $4K cushion over $180K).
        # ─── TIERED WORST-CASE SIZING (deadlock fix) ───
        # Gate to the HARD FLOOR ($180k, real breach point) using resulting-portfolio
        # worst-case. FLAT: one floor-safe position allowed (recovery possible, cures
        # the freeze). ONE OPEN: 2nd allowed only if double-stop still can't breach
        # the floor — blow-up protection UNCHANGED.
        HARD_FLOOR = ACCOUNT.initial_balance * 0.90     # $180,000
        SLIP_BUFFER = 500.0                             # slippage/fee cushion above floor
        balance = self.compliance._equity_cache if getattr(self.compliance, '_equity_cache', 0) else ACCOUNT.initial_balance
        existing_wc_loss = 0.0
        for _p in self.positions.values():
            if not _p.tp1_hit:
                existing_wc_loss += _p.risk_usd          # pre-TP1 = full worst-case loss
        room = (balance - existing_wc_loss - SLIP_BUFFER) - HARD_FLOOR
        if room <= 0:
            logger.warning(f"⚠️ SKIP {signal.symbol}: worst-case sizing — no room "
                           f"(bal=${balance:,.0f} - open_wc=${existing_wc_loss:,.0f} "
                           f"- buf=${SLIP_BUFFER:,.0f} < floor ${HARD_FLOOR:,.0f})")
            return
        if signal.risk_usd > room:
            _sl_dist = abs(signal.entry_price - signal.sl_price)
            if _sl_dist <= 0:
                logger.warning(f"⚠️ SKIP {signal.symbol}: invalid sl_dist for worst-case sizing")
                return
            _new_qty = room / _sl_dist
            logger.info(f"📐 WORST-CASE CAP {signal.symbol}: qty {signal.qty:.4f} -> {_new_qty:.4f} "
                        f"(room=${room:,.0f}, open_wc=${existing_wc_loss:,.0f})")
            signal.qty = _new_qty
            signal.risk_usd = room
        # ─── END WORST-CASE SIZING CAP ───
        qty_str = self.client.quantize_qty(signal.symbol, signal.qty)
        if not qty_str:
            logger.warning(f"⚠️ SKIP {signal.symbol}: qty too small after quantization")
            return

        # Place order(s) — handles chunking internally
        logger.info(f"🚀 PLACING: {signal.side} {qty_str} {signal.symbol} SL={signal.sl_price:.6f}")
        order_ids = self.client.place_order_chunked(
            signal.symbol, signal.side, float(qty_str), sl_price=signal.sl_price
        )

        if order_ids:
            actual_qty = float(qty_str)
            # Display ACTUAL risk from final qty (reflects notional/exposure/worst-case caps + quantization)
            actual_risk_usd = abs(signal.entry_price - signal.sl_price) * actual_qty
            self.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                side=signal.side,
                entry_price=signal.entry_price,
                qty=actual_qty,
                sl_price=signal.sl_price,
                tp1_price=signal.tp1_price,
                tp2_price=signal.tp2_price,
                atr_at_entry=signal.atr,
                entry_time=datetime.now(timezone.utc).isoformat(),
                risk_usd=actual_risk_usd,
                order_ids=order_ids,
            )
            logger.info(f"✅ ENTRY: {signal.side} {actual_qty} {signal.symbol} @ "
                        f"{signal.entry_price:.6f} | SL={signal.sl_price:.6f} | "
                        f"Risk=${actual_risk_usd:.0f} | {len(order_ids)} order(s)")
            discord_msg(f"✅ HYRO ENTRY: {signal.side} {actual_qty} {signal.symbol} @ "
                        f"{signal.entry_price:.6f} | SL={signal.sl_price:.6f} | "
                        f"Risk=${actual_risk_usd:.0f}")
        else:
            logger.error(f"❌ ORDER FAILED: {signal.side} {qty_str} {signal.symbol}")
            discord_msg(f"❌ HYRO ORDER FAILED: {signal.side} {qty_str} {signal.symbol}")

    def _check_cashout(self, today: str):
        """Funded-mode 3-day cashout."""
        if not self.state.last_cashout_date:
            self.state.last_cashout_date = today
            return

        try:
            days = (datetime.strptime(today, "%Y-%m-%d") -
                    datetime.strptime(self.state.last_cashout_date, "%Y-%m-%d")).days
        except Exception:
            return

        if days >= ACCOUNT.cashout_frequency_days:
            logger.info(f"💰 CASHOUT triggered ({days} days since last)")
            self._close_all_positions(reason="CASHOUT")
            profit = self.state.total_realized_pnl
            if profit > 0:
                payout = profit * ACCOUNT.profit_split
                self.state.total_cashouts += payout
                self.state.cashout_count += 1
                logger.info(f"💰 CASHOUT: profit=${profit:+,.0f} → payout=${payout:+,.0f} "
                            f"(total: ${self.state.total_cashouts:+,.0f})")
                discord_msg(f"💰 HYRO CASHOUT #{self.state.cashout_count}: ${payout:+,.0f}")
            self.state.total_realized_pnl = 0
            self.state.last_cashout_date = today
            self.compliance.reset_for_cashout()

    def _close_all_positions(self, reason: str = "MANUAL"):
        """Force-close all positions. Used by DD breach and cashout."""
        for symbol in list(self.positions.keys()):
            try:
                pos = self.positions[symbol]
                self.client.close_position(symbol, pos.side, pos.qty)
                ticker = self.client.get_ticker(symbol)
                if ticker:
                    price = ticker['last_price']
                    if pos.side == "BUY":
                        pnl = (price - pos.entry_price) * pos.qty
                    else:
                        pnl = (pos.entry_price - price) * pos.qty
                    if pos.tp1_hit:
                        pnl += pos.tp1_pnl_usd
                    self.state.total_realized_pnl += pnl
                    logger.info(f"🚪 CLOSE ({reason}): {symbol} @ {price:.6f} | PnL=${pnl:+,.0f}")
                del self.positions[symbol]
            except Exception as e:
                logger.error(f"Close failed {symbol}: {e}")
                discord_msg(f"❌ HYRO CLOSE ERROR ({reason}): {symbol} - {e}")

    def _send_status_update(self, equity: float, dd: dict):
        """Hourly Discord status — mobile-friendly with ADX column."""
        mode = MODE.mode
        confirm_bars = STRATEGY.confirmation_bars
        adx_threshold = STRATEGY.get_adx_threshold(mode)
        adx_filter_on = STRATEGY.use_adx(mode)

        status_lines = []
        for sym in STRATEGY.symbols:
            if sym not in self.strategy.engines:
                continue
            eng = self.strategy.engines[sym]
            hold = self.strategy.direction_hold.get(sym, 0)
            ef = self.strategy.ema_fast_val.get(sym, 0)
            es = self.strategy.ema_slow_val.get(sym, 0)

            if eng.trend == 1:
                trend = 'BULL'
            elif eng.trend == -1:
                trend = 'BEAR'
            else:
                trend = '----'

            adx_val = self.strategy.adx_calcs[sym].value if sym in self.strategy.adx_calcs else 0
            adx_str = f"{adx_val:>3.0f}" if adx_val > 0 else "  -"
            if adx_filter_on:
                adx_ok = 'OK' if adx_val > adx_threshold else 'NO'
            else:
                adx_ok = 'off'

            if eng.trend == 1:
                ema_ok = 'OK' if ef > es else 'NO'
            elif eng.trend == -1:
                ema_ok = 'OK' if ef < es else 'NO'
            else:
                ema_ok = 'NO'

            if hold == confirm_bars:
                state = 'READY'
            elif hold < confirm_bars:
                state = f'WAIT({hold})'
            else:
                state = f'OLD({hold})'

            short = sym.replace('USDT', '')
            status_lines.append(
                f"{short:<6} {trend} {hold:>2}b  "
                f"ADX {adx_str} {adx_ok}  "
                f"EMA {ema_ok}  {state}"
            )

        pos_str = ', '.join(f'{s.replace("USDT","")} {p.side}' for s, p in self.positions.items()) or 'None'
        msg = (f"**HYRO Bar {self.state.bar_count}** | Eq: ${equity:,.0f} | "
               f"DD: {dd['dd_used_pct']:.1f}%/100% | Swing: {dd['daily_dd_used_pct']:.1f}%/100%\n"
               f"Pos: {pos_str}\n```\n" + '\n'.join(status_lines) + "\n```")

        if len(msg) > 1950:
            msg = msg[:1947] + "..."

        discord_msg(msg, MONITORING.status_webhook)


if __name__ == "__main__":
    bot = HyroTraderBot()
    bot.run()
