#!/usr/bin/env python3
"""
A faithful in-memory stand-in for the Bybit v5 client.

Why this exists: this sandbox cannot reach Bybit at all. api.bybit.com returns
HTTP 403 with the body

    "The Amazon CloudFront distribution is configured to block access from
     your country"

for every endpoint including unauthenticated ones (/v5/market/time,
/v5/market/kline). It is a CloudFront geographic block on the egress IP, not a
credential problem and not an outage -- api.github.com returns 200 from the
same host, and api.binance.com returns 451. No API key would change this.

So the live execution path is exercised against this mock instead. The mock
deliberately reproduces the behaviours that break naive clients:

  * qtyStep / minOrderQty quantization, including symbols where the step is
    1.0 (whole contracts) and where it is 0.001
  * orders below minOrderQty are rejected, exactly as quantize_qty returning
    None must be handled
  * positions come back with STRING numerics, as Bybit sends them
  * side is capitalised "Buy"/"Sell"
  * a symbol that raises on order placement, to prove one bad symbol cannot
    abort the cycle

This is a test double. It proves the logic is correct. It does NOT prove
connectivity, authentication, rate limits, or real fills -- only a run against
the real testnet from a permitted jurisdiction can do that.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")

logger = logging.getLogger("MockExchange")

# symbol -> (mark price, qtyStep, minOrderQty)
DEFAULT_INSTRUMENTS: Dict[str, tuple] = {
    "BTCUSDT":    (61_000.0, 0.001, 0.001),
    "ETHUSDT":     (3_400.0, 0.01,  0.01),
    "SOLUSDT":       (145.0, 0.1,   0.1),
    "LINKUSDT":       (14.5, 0.1,   0.1),
    "AAVEUSDT":       (92.0, 0.01,  0.01),
    "XRPUSDT":         (0.52, 1.0,   1.0),
    "DOGEUSDT":        (0.13, 1.0,   1.0),
    "ADAUSDT":         (0.38, 1.0,   1.0),
    "NEARUSDT":        (4.10, 0.1,   0.1),
    "FILUSDT":         (4.85, 0.1,   0.1),
    "ONDOUSDT":        (0.72, 1.0,   1.0),
    "WLDUSDT":         (1.85, 0.1,   0.1),
    "ZECUSDT":        (28.40, 0.01,  0.01),
    "SUIUSDT":         (0.95, 0.1,   0.1),
    "AVAXUSDT":       (24.30, 0.1,   0.1),
}


class MockBybitClient:
    """Implements exactly the surface bot/execution.py and reconciliation use."""

    def __init__(self, instruments: Optional[Dict[str, tuple]] = None,
                 equity: float = 200_000.0,
                 failing_symbols: Optional[set] = None):
        self.instruments = dict(instruments or DEFAULT_INSTRUMENTS)
        self.equity = equity
        self.failing_symbols = failing_symbols or set()
        # symbol -> {"side": "Buy"/"Sell", "size": float}
        self._positions: Dict[str, dict] = {}
        self.order_log: List[dict] = []
        self.sl_log: List[dict] = []
        self.tp_log: List[dict] = []
        self.one_way_checked: List[str] = []

    # ------------------------------------------------------------ helpers
    def seed_position(self, symbol: str, side: str, size: float) -> None:
        self._positions[symbol] = {"side": side, "size": float(size)}

    def _mark(self, symbol: str) -> float:
        if symbol not in self.instruments:
            raise ValueError(f"unknown symbol {symbol}")
        return self.instruments[symbol][0]

    # ------------------------------------------------------- client API
    def get_positions(self, symbol: Optional[str] = None) -> List[dict]:
        out = []
        for s, p in self._positions.items():
            if symbol and s != symbol:
                continue
            if p["size"] == 0:
                continue
            # Bybit returns strings, not floats -- callers must cope
            out.append({
                "symbol": s,
                "side": p["side"],
                "size": str(p["size"]),
                "markPrice": str(self._mark(s)),
                "avgPrice": str(self._mark(s)),
                "unrealisedPnl": "0",
            })
        return out

    def get_ticker(self, symbol: str) -> dict:
        m = self._mark(symbol)
        return {"symbol": symbol, "lastPrice": str(m), "markPrice": str(m),
                "indexPrice": str(m)}

    def get_wallet_balance(self) -> float:
        return self.equity

    def quantize_qty(self, symbol: str, qty: float) -> Optional[str]:
        if symbol not in self.instruments:
            raise ValueError(f"unknown symbol {symbol}")
        _, step, min_qty = self.instruments[symbol]
        steps = int(abs(qty) / step)          # floor, never round up into more risk
        q = steps * step
        if q < min_qty:
            return None
        decimals = max(0, len(str(step).split(".")[1]) if "." in str(step) else 0)
        return f"{q:.{decimals}f}"

    def quantize_price(self, symbol: str, price: float) -> str:
        return f"{price:.8f}"

    def assert_one_way_mode(self, symbol: str) -> bool:
        self.one_way_checked.append(symbol)
        return True

    def place_order_chunked(self, symbol: str, side: str, qty: float,
                            reduce_only: bool = False, **kw) -> List[str]:
        if symbol in self.failing_symbols:
            raise RuntimeError(f"simulated exchange rejection for {symbol}")
        if symbol not in self.instruments:
            raise ValueError(f"unknown symbol {symbol}")
        qty = float(qty)
        if qty <= 0:
            raise ValueError(f"non-positive qty {qty}")

        cur = self._positions.get(symbol, {"side": "Buy", "size": 0.0})
        # Coerce: the real Bybit API returns position size as a STRING, so any
        # code path that seeds or reads positions must tolerate both.
        cur_size = float(cur["size"])
        signed = cur_size * (1 if cur["side"] == "Buy" else -1)
        signed += qty if side == "Buy" else -qty

        if reduce_only and abs(signed) > abs(cur_size) + 1e-12:
            raise RuntimeError(f"reduceOnly order would increase {symbol}")

        if abs(signed) < 1e-12:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = {"side": "Buy" if signed > 0 else "Sell",
                                       "size": abs(signed)}
        oid = f"mock-{len(self.order_log)+1:05d}"
        self.order_log.append({"orderId": oid, "symbol": symbol, "side": side,
                               "qty": qty, "reduceOnly": reduce_only})
        return [oid]

    def modify_sl(self, symbol: str, sl_price: float) -> bool:
        if symbol not in self._positions:
            return False
        self.sl_log.append({"symbol": symbol, "sl": sl_price})
        return True

    def modify_tp(self, symbol: str, tp_price: float) -> bool:
        if symbol not in self._positions:
            return False
        if not hasattr(self, "tp_log"):
            self.tp_log = []
        self.tp_log.append({"symbol": symbol, "tp": tp_price})
        return True

    def close_position(self, symbol: str, position_side: str, qty: float) -> List[str]:
        side = "Sell" if position_side == "Buy" else "Buy"
        return self.place_order_chunked(symbol, side, qty, reduce_only=True)


def _step_for(price: float) -> tuple:
    """Approximate Bybit's qtyStep/minOrderQty tiering from price magnitude.

    Bybit's real filters are per-symbol and only available from
    /v5/market/instruments-info, which is geo-blocked from this sandbox. This
    is a deliberate APPROXIMATION for offline testing only -- it reproduces the
    *shape* of the constraint (coarse steps on cheap coins, fine steps on
    expensive ones) so quantization and min-qty rejection paths get exercised.
    It is NOT the real filter table. Any live run must read the real filters.
    """
    if price >= 10_000:  return (0.001, 0.001)
    if price >= 1_000:   return (0.01, 0.01)
    if price >= 100:     return (0.01, 0.01)
    if price >= 10:      return (0.1, 0.1)
    if price >= 1:       return (0.1, 0.1)
    return (1.0, 1.0)


def instruments_from_panel(hist_dirs: Optional[List[str]] = None) -> Dict[str, tuple]:
    """Build a mock instrument table covering the ACTUAL traded universe, with
    each coin's last close from the panel as its mark.

    The hardcoded DEFAULT_INSTRUMENTS covers only 15 symbols while the strategy
    routinely holds 34, so every cycle generated ~21 'unknown symbol' planning
    errors that had nothing to do with the bot's logic.
    """
    import csv as _csv
    import glob as _glob
    # BOTH panels. The bot trades a UNION of two universes served by two
    # different directories -- Universe A (Main/Flow/Short) from run/, and
    # Universe B (DELTA/RELVOL/BOS) from clean_panel/. Reading only one of
    # them leaves ~10 traded symbols with no mark price.
    if hist_dirs is None:
        hist_dirs = [os.path.join(WORKSPACE, "clean_panel", "hist"),
                     os.path.join(WORKSPACE, "run", "hist")]
    out: Dict[str, tuple] = {}
    paths = [p for d in hist_dirs for p in _glob.glob(os.path.join(d, "*_1h.csv"))]
    for path in paths:
        coin = os.path.basename(path)[:-len("_1h.csv")]
        last_close = None
        try:
            with open(path, newline="") as fh:
                for row in _csv.reader(fh):
                    if len(row) > 5 and row[5].strip() and row[5] != "close":
                        try:
                            last_close = float(row[5])
                        except ValueError:
                            continue
        except OSError:
            continue
        if last_close and last_close > 0:
            step, minq = _step_for(last_close)
            out[f"{coin}USDT"] = (last_close, step, minq)
    return out
