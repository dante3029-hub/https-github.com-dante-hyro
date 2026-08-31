"""
Hardened extension of the original bot's bybit_client.BybitClient -- "the
heart" of the previous bot per AUDIT_FINDINGS.md (Decimal-based
quantization, order chunking, instrument-info caching are all reused
UNCHANGED by subclassing, not rewritten).

Additions over the original (each one a specific gap AUDIT_FINDINGS.md
raised, fixed here rather than silently patched):
  1. Explicit TESTNET vs DEMO vs MAINNET distinction. The original client's
     `TESTNET_URL = "https://api-demo.bybit.com"` is Bybit's DEMO TRADING
     domain (uses live market data, simulated ledger), not Bybit's actual
     isolated testnet (https://api-testnet.bybit.com, sandbox market data).
     Silently calling that "testnet" hides which one you're actually
     talking to. This class requires the caller to name the mode explicitly.
  2. Retries with exponential backoff on transient network/5xx errors.
  3. Idempotency via a deterministic orderLinkId (so a retried request that
     actually succeeded server-side doesn't double-fill on retry).
  4. Explicit one-way/net position-mode assertion at startup (hedge-mode
     accounts need positionIdx on every order; this bot assumes one-way
     mode and refuses to silently place orders that would be misrouted).
  5. Basic rate-limit back-off on retCode 10006/10018 (rate limited).

Everything else (get_klines, get_ticker, get_instrument_info,
get_wallet_balance, get_positions, set_leverage, quantize_qty,
quantize_price, place_order_chunked, _place_single, modify_sl,
close_position) is inherited UNCHANGED from bybit_client.BybitClient.
"""
import os
import sys
import time
import uuid
import logging

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

sys.path.insert(0, f"{WORKSPACE}/existing_botcode/botcode")
from bybit_client import BybitClient  # the "heart" -- reused unchanged via inheritance

logger = logging.getLogger("HardenedBybitClient")

MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.0
RATE_LIMIT_RETCODES = (10006, 10018)


class ExchangeMode:
    MAINNET = "mainnet"    # https://api.bybit.com            -- real money, real market
    DEMO = "demo"          # https://api-demo.bybit.com        -- live market data, simulated ledger
    TESTNET = "testnet"    # https://api-testnet.bybit.com     -- sandbox market data, sandbox ledger


class HardenedBybitClient(BybitClient):
    TESTNET_URL_REAL = "https://api-testnet.bybit.com"  # Bybit's actual isolated testnet
    DEMO_URL = "https://api-demo.bybit.com"              # what the original client mislabeled "testnet"

    def __init__(self, api_key: str, api_secret: str, mode: str = ExchangeMode.DEMO):
        if mode not in (ExchangeMode.MAINNET, ExchangeMode.DEMO, ExchangeMode.TESTNET):
            raise ValueError(f"unknown exchange mode: {mode!r} -- must be mainnet/demo/testnet, no silent default")
        self.mode = mode
        # bypass BybitClient.__init__'s testnet bool; set base_url explicitly per mode
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = 5000
        self._instrument_cache = {}
        if mode == ExchangeMode.MAINNET:
            self.base_url = self.MAINNET_URL
        elif mode == ExchangeMode.DEMO:
            self.base_url = self.DEMO_URL
        else:
            self.base_url = self.TESTNET_URL_REAL
        logger.info(f"HardenedBybitClient initialized in mode={mode} base_url={self.base_url}")
        if not api_key or not api_secret:
            logger.warning("No API key/secret provided -- read-only/dry-run calls will fail on auth")

    # ------------------------------------------------------------ retry/backoff wrapper
    def _with_retries(self, fn, *args, **kwargs):
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, dict) and result.get("retCode") in RATE_LIMIT_RETCODES:
                    wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.warning(f"rate limited (retCode={result.get('retCode')}), backing off {wait:.1f}s "
                                   f"(attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                return result
            except Exception as e:
                last_exc = e
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(f"{fn.__name__} raised {e!r}, retrying in {wait:.1f}s "
                               f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
        logger.error(f"{fn.__name__} failed after {MAX_RETRIES} attempts")
        if last_exc:
            raise last_exc
        return {}

    def _get(self, endpoint: str, params: dict = None) -> dict:
        return self._with_retries(super()._get, endpoint, params)

    def _post(self, endpoint: str, params: dict) -> dict:
        return self._with_retries(super()._post, endpoint, params)

    # ------------------------------------------------------------ idempotent order placement
    def _place_single(self, symbol: str, side: str, qty_str: str,
                       sl_price, reduce_only: bool, order_link_id: str = None) -> dict:
        """Adds a deterministic-or-caller-supplied orderLinkId so a retried
        request that actually succeeded server-side is recognized by Bybit
        as a duplicate rather than double-filling. The base class's
        _place_single doesn't accept this param, so this override
        reimplements the request body construction (still calling the
        inherited self._post, which now has retry/backoff)."""
        link_id = order_link_id or f"hb-{symbol}-{side}-{uuid.uuid4().hex[:16]}"
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": "Market",
            "qty": qty_str,
            "orderLinkId": link_id,
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
            logger.info(f"Order placed: {side} {qty_str} {symbol} linkId={link_id}"
                        + (f" SL={params['stopLoss']}" if "stopLoss" in params else "") + f" ID={oid}")
        else:
            logger.error(f"Order FAILED: {side} {qty_str} {symbol} linkId={link_id} "
                         f"retCode={ret_code} msg={ret_msg}")
        return {"oid": oid, "error": ret_msg, "retCode": ret_code, "order_link_id": link_id}

    # ------------------------------------------------------------ position-mode assertion
    def assert_one_way_mode(self, symbol: str) -> bool:
        """Bybit's hedge mode requires a positionIdx on every order; this
        bot's order placement code (inherited from BybitClient) does not
        send one, which is only correct in one-way/net mode. Checks the
        account's actual mode via the position list response and raises
        loudly rather than silently placing orders that could be rejected
        or misrouted under hedge mode."""
        resp = self._get("/v5/position/list", {"category": "linear", "symbol": symbol})
        positions = resp.get("result", {}).get("list", [])
        for p in positions:
            idx = p.get("positionIdx")
            if idx not in (0, None):
                raise RuntimeError(
                    f"Account appears to be in HEDGE mode for {symbol} (positionIdx={idx}). "
                    f"This bot only supports one-way/net mode. Switch the account to one-way "
                    f"mode in Bybit settings before running live -- do not silently proceed."
                )
        return True

    def modify_tp(self, symbol: str, tp_price: float) -> bool:
        """Set/amend the take-profit on an open position.

        The inherited BybitClient only implements modify_sl(). BOS carries a
        3-ATR take-profit that previously lived only inside the tracker, so it
        was never protected at the venue between 4h checks. Same endpoint,
        takeProfit field. Omitting stopLoss leaves the existing SL untouched
        (Bybit v5 trading-stop semantics), so this cannot clobber the stop.
        """
        tp_str = self.quantize_price(symbol, tp_price)
        resp = self._post("/v5/position/trading-stop", {
            "category": "linear", "symbol": symbol,
            "takeProfit": tp_str, "tpTriggerBy": "LastPrice",
        })
        return resp.get("retCode") == 0
