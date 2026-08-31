"""
bybit_client.py — Robust Bybit V5 API wrapper.

Key features:
  - Auto-fetches instrument info (qty step, price step, min/max qty) at startup
  - Decimal-based quantization (no float precision issues)
  - Order chunking when qty exceeds maxOrderQty
  - Mark price queries for accurate equity calculation
"""
import time
import hmac
import hashlib
import json as json_mod
import logging
from decimal import Decimal, ROUND_DOWN, getcontext
import re as re_mod
from typing import Optional, List, Dict
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None

getcontext().prec = 28  # plenty for crypto


def _decimal_str(value, step) -> str:
    """Quantize value DOWN to a multiple of step, return clean string."""
    d = Decimal(str(value))
    s = Decimal(str(step))
    if s == 0:
        return str(d)
    quantized = (d / s).quantize(Decimal('1'), rounding=ROUND_DOWN) * s
    # Format without scientific notation, strip trailing zeros after decimal
    result = format(quantized, 'f')
    if '.' in result:
        result = result.rstrip('0').rstrip('.')
    return result if result else '0'


class BybitClient:
    MAINNET_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-demo.bybit.com"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL
        self.recv_window = 5000
        self._instrument_cache: Dict[str, dict] = {}

    # ─── Internal HTTP ───
    def _sign(self, payload_str: str, timestamp: str) -> str:
        param_str = f"{timestamp}{self.api_key}{self.recv_window}{payload_str}"
        return hmac.HMAC(self.api_secret.encode(), param_str.encode(), hashlib.sha256).hexdigest()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        params = params or {}
        url = f"{self.base_url}{endpoint}"
        ts = str(int(time.time() * 1000))
        qs = urlencode(params) if params else ""
        sign = self._sign(qs, ts)
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": sign,
            "Content-Type": "application/json",
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            data = resp.json()
            if data.get("retCode") != 0:
                logger.error(f"GET {endpoint} retCode={data.get('retCode')} msg={data.get('retMsg')}")
            return data
        except Exception as e:
            logger.error(f"GET {endpoint} exception: {e}")
            return {}

    def _post(self, endpoint: str, params: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        ts = str(int(time.time() * 1000))
        body = json_mod.dumps(params)
        sign = self._sign(body, ts)
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": sign,
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, data=body, headers=headers, timeout=10)
            data = resp.json()
            if data.get("retCode") != 0:
                logger.error(f"POST {endpoint} retCode={data.get('retCode')} msg={data.get('retMsg')} params={params}")
            return data
        except Exception as e:
            logger.error(f"POST {endpoint} exception: {e}")
            return {}

    # ─── Public market data ───
    def get_klines(self, symbol: str, interval: str = "60", limit: int = 200) -> List[dict]:
        resp = self._get("/v5/market/kline", {
            "category": "linear", "symbol": symbol,
            "interval": interval, "limit": limit
        })
        klines = []
        for k in resp.get("result", {}).get("list", []):
            klines.append({
                "timestamp": int(k[0]),
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5]),
            })
        klines.reverse()  # Bybit returns newest-first; reverse to oldest-first
        return klines

    def get_ticker(self, symbol: str) -> dict:
        resp = self._get("/v5/market/tickers", {"category": "linear", "symbol": symbol})
        tickers = resp.get("result", {}).get("list", [])
        if tickers:
            t = tickers[0]
            return {
                "last_price": float(t["lastPrice"]),
                "mark_price": float(t.get("markPrice", t["lastPrice"])),
                "bid": float(t.get("bid1Price", 0)),
                "ask": float(t.get("ask1Price", 0)),
            }
        return {}

    def get_instrument_info(self, symbol: str) -> dict:
        """Cached. Returns: min_qty, max_qty, qty_step, tick_size, min_notional"""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        resp = self._get("/v5/market/instruments-info", {"category": "linear", "symbol": symbol})
        instruments = resp.get("result", {}).get("list", [])
        if instruments:
            info = instruments[0]
            lot = info["lotSizeFilter"]
            price = info["priceFilter"]
            result = {
                "min_qty": Decimal(lot["minOrderQty"]),
                "max_qty": Decimal(lot["maxOrderQty"]),
                "qty_step": Decimal(lot["qtyStep"]),
                "tick_size": Decimal(price["tickSize"]),
                "min_notional": Decimal(lot.get("minNotionalValue", "0")),
            }
            self._instrument_cache[symbol] = result
            return result
        return {}

    # ─── Account ───
    def get_wallet_balance(self) -> float:
        resp = self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        coins = resp.get("result", {}).get("list", [{}])[0].get("coin", [])
        for coin in coins:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0

    def get_positions(self, symbol: Optional[str] = None) -> List[dict]:
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        resp = self._get("/v5/position/list", params)
        positions = []
        for p in resp.get("result", {}).get("list", []):
            if float(p["size"]) > 0:
                positions.append({
                    "symbol": p["symbol"],
                    "side": p["side"],   # "Buy" or "Sell"
                    "size": float(p["size"]),
                    "entry_price": float(p["avgPrice"]),
                    "mark_price": float(p.get("markPrice", p["avgPrice"])),
                    "unrealized_pnl": float(p["unrealisedPnl"]),
                    "leverage": p["leverage"],
                    "stop_loss": float(p["stopLoss"]) if p.get("stopLoss") else 0,
                    "take_profit": float(p["takeProfit"]) if p.get("takeProfit") else 0,
                })
        return positions

    def set_leverage(self, symbol: str, leverage: int):
        try:
            self._post("/v5/position/set-leverage", {
                "category": "linear", "symbol": symbol,
                "buyLeverage": str(leverage), "sellLeverage": str(leverage),
            })
        except Exception as e:
            logger.warning(f"set_leverage {symbol}: {e}")

    # ─── Order placement (with chunking) ───
    def quantize_qty(self, symbol: str, qty: float) -> Optional[str]:
        """Returns clean string qty respecting qty_step, or None if below min."""
        info = self.get_instrument_info(symbol)
        if not info:
            logger.error(f"No instrument info for {symbol}")
            return None

        step = info["qty_step"]
        min_qty = info["min_qty"]

        d = Decimal(str(qty))
        if step > 0:
            d = (d / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step

        if d < min_qty:
            logger.warning(f"{symbol}: qty {d} below min {min_qty}")
            return None

        result = format(d, 'f')
        if '.' in result:
            result = result.rstrip('0').rstrip('.')
        return result if result else None

    def quantize_price(self, symbol: str, price: float) -> str:
        info = self.get_instrument_info(symbol)
        if not info:
            return f"{price:.6f}"
        tick = info["tick_size"]
        d = Decimal(str(price))
        if tick > 0:
            d = (d / tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick
        result = format(d, 'f')
        return result

    def _parse_real_max(self, error_msg: str, symbol: str) -> Optional[float]:
        """Parse real max_qty from Bybit error message."""
        match = re_mod.search(r'max_qty:(\d+)', error_msg)
        if not match:
            return None
        internal_max = int(match.group(1))
        real_max = internal_max / 100_000_000  # Bybit internal = qty * 10^8
        info = self.get_instrument_info(symbol)
        if info:
            step = info["qty_step"]
            d = Decimal(str(real_max))
            if step > 0:
                d = (d / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step
            real_max = float(d)
        logger.info(f"{symbol}: parsed REAL max_qty from error = {real_max}")
        return real_max

    def place_order_chunked(self, symbol: str, side: str, qty: float,
                            sl_price: Optional[float] = None,
                            reduce_only: bool = False) -> List[str]:
        """
        Place order(s) with auto-chunking.
        If order fails with 'exceeds maximum', parses real max from error and retries.
        SL attached to FIRST chunk only.
        """
        info = self.get_instrument_info(symbol)
        if not info:
            logger.error(f"Cannot place order — no instrument info for {symbol}")
            return []

        max_qty = float(info["max_qty"])
        qty_str_total = self.quantize_qty(symbol, qty)
        if not qty_str_total:
            return []

        total_qty = float(qty_str_total)
        order_ids = []

        # Try single order first if under reported max
        if total_qty <= max_qty:
            result = self._place_single(symbol, side, qty_str_total, sl_price, reduce_only)
            if result.get("oid"):
                return [result["oid"]]
            # Check if failed due to max_qty being wrong (testnet vs mainnet)
            if result.get("retCode") == 10001 and "exceeds maximum" in result.get("error", ""):
                real_max = self._parse_real_max(result["error"], symbol)
                if real_max and real_max < total_qty:
                    max_qty = real_max
                    # Update cache so future orders use real limit
                    self._instrument_cache[symbol]["max_qty"] = Decimal(str(real_max))
                    logger.info(f"{symbol}: updated cached max_qty to {real_max}")
                else:
                    logger.error(f"{symbol}: order failed but couldn't parse real max")
                    return []
            else:
                return []

        # Split into chunks using the (possibly updated) max_qty
        remaining = total_qty
        chunk_num = 0
        while remaining > 0:
            chunk_size = min(remaining, max_qty)
            chunk_str = self.quantize_qty(symbol, chunk_size)
            if not chunk_str or float(chunk_str) <= 0:
                break

            # SL only on first chunk
            sl = sl_price if chunk_num == 0 else None
            result = self._place_single(symbol, side, chunk_str, sl, reduce_only)

            if result.get("oid"):
                order_ids.append(result["oid"])
                remaining -= float(chunk_str)
                chunk_num += 1
                logger.info(f"{symbol}: chunk {chunk_num} placed ({chunk_str}), "
                           f"remaining={remaining:.2f}")
            elif result.get("retCode") == 10001 and "exceeds maximum" in result.get("error", ""):
                # Max still too high, parse again
                real_max = self._parse_real_max(result["error"], symbol)
                if real_max and real_max < max_qty:
                    max_qty = real_max
                    self._instrument_cache[symbol]["max_qty"] = Decimal(str(real_max))
                    continue  # Retry this chunk with new max
                else:
                    logger.error(f"{symbol}: chunk failed, can't reduce max further")
                    break
            else:
                logger.error(f"{symbol}: chunk {chunk_num+1} failed: {result.get('error')}")
                break

        if order_ids:
            logger.info(f"{symbol}: total {len(order_ids)} chunks placed for {total_qty}")
        return order_ids

    def _place_single(self, symbol: str, side: str, qty_str: str,
                      sl_price: Optional[float], reduce_only: bool) -> dict:
        """Returns {'oid': str|None, 'error': str|None, 'retCode': int}"""
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": "Market",
            "qty": qty_str,
        }
        if sl_price is not None and sl_price > 0:
            params["stopLoss"] = self.quantize_price(symbol, sl_price)
            params["slTriggerBy"] = "LastPrice"
        if reduce_only:
            params["reduceOnly"] = True

        resp = self._post("/v5/order/create", params)
        ret_code = resp.get("retCode", -1)
        ret_msg = resp.get("retMsg", "")
        oid = resp.get("result", {}).get("orderId")
        if oid:
            logger.info(f"Order placed: {side} {qty_str} {symbol}"
                       + (f" SL={params['stopLoss']}" if 'stopLoss' in params else "")
                       + f" ID={oid}")
        return {"oid": oid, "error": ret_msg, "retCode": ret_code}

    def modify_sl(self, symbol: str, sl_price: float) -> bool:
        sl_str = self.quantize_price(symbol, sl_price)
        resp = self._post("/v5/position/trading-stop", {
            "category": "linear", "symbol": symbol,
            "stopLoss": sl_str, "slTriggerBy": "LastPrice",
        })
        return resp.get("retCode") == 0

    def close_position(self, symbol: str, position_side: str, qty: float) -> List[str]:
        """Close position — uses chunking if needed. position_side is the side of the OPEN position."""
        close_side = "Sell" if position_side.upper() == "BUY" else "Buy"
        return self.place_order_chunked(symbol, close_side, qty, sl_price=None, reduce_only=True)
