#!/usr/bin/env python3
"""
Live execution engine.

Converts the orchestrator's per-coin DOLLAR targets into actual exchange
orders. This is the layer that was previously a stub -- orchestrator step 8
computed `tracked` (with qty still denominated in DOLLARS, not contracts) and
then did nothing with it.

Design rules, all of which exist because getting them wrong loses real money:

  1. DOLLARS ARE NOT CONTRACTS. Every target must be divided by mark price and
     then quantized to the instrument's qtyStep before it can be sent. The old
     `tracked` dict passed dollars straight through as `qty`.

  2. DIFF, DON'T RESET. Never flatten-then-rebuild. Compute the delta between
     the current exchange position and the target and trade only the
     difference. A flatten/rebuild pays the spread twice on the entire book
     every cycle -- on a $146k gross book at ~8.5bp round trip that is roughly
     $250/cycle of pure, avoidable cost.

  3. NO-TRADE BAND. Do not send an order for a trivial delta. Churning a $12
     adjustment on a $9,000 position pays fees to achieve nothing. Default
     band is the larger of $25 and 1% of the target.

  4. REDUCE BEFORE INCREASE. Execute all size-reducing orders before
     size-increasing ones so that peak margin usage during the rebalance never
     exceeds the starting or ending state. Increasing first can margin-call a
     book that is fine at both endpoints.

  5. FAIL CLOSED, PER SYMBOL. One symbol's failure must not abort the cycle or,
     worse, leave the book half-rebalanced with no record. Every order is
     wrapped; failures are collected and reported, never swallowed silently.

  6. FLATTEN IS UNCONDITIONAL. When the kill switch or a compliance bust fires,
     the target book is empty and every live position must be closed with
     reduceOnly orders, regardless of the no-trade band.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from bot import config

logger = logging.getLogger("Execution")

# minimum dollar delta worth trading
MIN_TRADE_USD = 25.0
# ...or this fraction of the target position, whichever is larger
MIN_TRADE_FRAC = 0.01


@dataclass
class PlannedOrder:
    symbol: str
    side: str                 # "Buy" | "Sell"
    qty_contracts: float
    qty_str: str              # exchange-quantized string, ready to send
    usd_delta: float          # signed dollar change this order effects
    reduce_only: bool
    mark_price: float
    reason: str

    def __repr__(self) -> str:
        return (f"<{self.side} {self.qty_str} {self.symbol} "
                f"(${self.usd_delta:+,.0f}, {'reduceOnly' if self.reduce_only else 'open'}, {self.reason})>")


@dataclass
class ExecutionPlan:
    orders: List[PlannedOrder] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)   # (symbol, why)
    errors: List[Tuple[str, str]] = field(default_factory=list)    # (symbol, error)

    @property
    def gross_traded_usd(self) -> float:
        return sum(abs(o.usd_delta) for o in self.orders)

    def summary(self) -> dict:
        return {
            "n_orders": len(self.orders),
            "n_reduce": sum(1 for o in self.orders if o.reduce_only),
            "n_increase": sum(1 for o in self.orders if not o.reduce_only),
            "gross_traded_usd": round(self.gross_traded_usd, 2),
            "n_skipped": len(self.skipped),
            "n_errors": len(self.errors),
        }


def _fmt_qty(qty: float, client, symbol: str) -> Optional[str]:
    """Format an already-valid contract quantity for the wire without
    re-flooring it. Falls back to the client's quantizer if the value does not
    cleanly match the instrument step (which would mean the exchange reported
    a size we cannot represent)."""
    q = client.quantize_qty(symbol, qty)
    try:
        if q is not None and abs(float(q) - qty) < 1e-9:
            return q
    except (TypeError, ValueError):
        pass
    # quantizer lost precision -- send the exchange's own number, trimmed
    s = f"{qty:.8f}".rstrip("0").rstrip(".")
    return s if s and float(s) > 0 else q


def _signed_usd(pos: dict, mark: float) -> float:
    """Current position expressed as signed USD notional."""
    size = float(pos.get("size", 0) or 0)
    if size == 0:
        return 0.0
    sign = 1.0 if str(pos.get("side", "")).lower() == "buy" else -1.0
    return sign * size * mark


def build_execution_plan(client,
                         combined_dollar_targets: Dict[str, float],
                         force_flat: bool = False,
                         min_trade_usd: float = MIN_TRADE_USD,
                         min_trade_frac: float = MIN_TRADE_FRAC) -> ExecutionPlan:
    """
    Diff current exchange state against the dollar targets and return the set
    of orders that moves the book from where it is to where it should be.

    Pure planning -- sends nothing. Separating plan from send is what makes the
    live path testable and what allows a dry-run to print exactly the orders
    that would have gone out.
    """
    plan = ExecutionPlan()

    try:
        positions = client.get_positions() or []
    except Exception as e:                                    # noqa: BLE001
        plan.errors.append(("<get_positions>", f"{type(e).__name__}: {e}"))
        logger.error(f"cannot read positions, refusing to plan any orders: {e}")
        return plan

    live: Dict[str, dict] = {}
    for p in positions:
        try:
            if float(p.get("size", 0) or 0) != 0:
                live[p["symbol"]] = p
        except (TypeError, ValueError, KeyError):
            continue

    # target book, keyed by exchange symbol
    targets: Dict[str, float] = {}
    if not force_flat:
        for coin, usd in combined_dollar_targets.items():
            if abs(usd) > 1.0:
                targets[config.to_bybit_symbol(coin)] = usd

    for symbol in sorted(set(targets) | set(live)):
        target_usd = targets.get(symbol, 0.0)
        pos = live.get(symbol)

        try:
            mark = _get_mark(client, symbol, pos)
        except Exception as e:                                # noqa: BLE001
            plan.errors.append((symbol, f"mark price unavailable: {type(e).__name__}: {e}"))
            continue
        if not mark or mark <= 0:
            plan.errors.append((symbol, f"invalid mark price {mark!r}"))
            continue

        current_usd = _signed_usd(pos, mark) if pos else 0.0
        delta_usd = target_usd - current_usd

        # closing a position entirely is never suppressed by the band
        closing = (target_usd == 0.0 and current_usd != 0.0)
        band = max(min_trade_usd, abs(target_usd) * min_trade_frac)
        if not closing and abs(delta_usd) < band:
            plan.skipped.append((symbol, f"delta ${delta_usd:+,.2f} inside no-trade band ${band:,.2f}"))
            continue

        # BUG FIX (found by burn_in.py phase 3, 2026-08-09).
        #
        # For a full close, do NOT re-derive the quantity from dollars.
        # size -> USD -> size is a lossy float round-trip, and quantize_qty
        # FLOORS to the instrument step, so the result is short by up to one
        # full step. Observed: a 1422.1-contract ICPUSDT position produced a
        # 1422.0 close order, leaving 0.1 contracts live. This happened on
        # EVERY position in a kill-switch flatten -- the bot reported
        # "forced flat", gross target $0.00, and was still carrying residual
        # exposure on all 34 legs. A kill switch that does not actually go
        # flat is not a kill switch.
        #
        # The exchange-reported position size is already a valid multiple of
        # the step, so for a close it is used verbatim.
        if closing and pos:
            try:
                qty_contracts = abs(float(pos.get("size", 0.0) or 0.0))
            except (TypeError, ValueError):
                qty_contracts = abs(delta_usd) / mark
        else:
            qty_contracts = abs(delta_usd) / mark
        try:
            if closing and pos:
                # exact size, no flooring
                qty_str = _fmt_qty(qty_contracts, client, symbol)
            else:
                qty_str = client.quantize_qty(symbol, qty_contracts)
        except Exception as e:                                # noqa: BLE001
            plan.errors.append((symbol, f"quantize_qty failed: {type(e).__name__}: {e}"))
            continue
        if qty_str is None:
            plan.skipped.append((symbol, f"qty {qty_contracts:.8f} below instrument minimum after quantization"))
            continue
        try:
            if float(qty_str) <= 0:
                plan.skipped.append((symbol, "quantized qty rounded to zero"))
                continue
        except (TypeError, ValueError):
            plan.errors.append((symbol, f"non-numeric quantized qty {qty_str!r}"))
            continue

        side = "Buy" if delta_usd > 0 else "Sell"
        # reduceOnly iff we are shrinking magnitude without crossing zero
        reduce_only = (
            current_usd != 0.0
            and (target_usd == 0.0 or (current_usd > 0) == (target_usd > 0))
            and abs(target_usd) < abs(current_usd)
        )
        reason = ("flatten" if force_flat else
                  "close" if closing else
                  "open" if current_usd == 0.0 else
                  "reduce" if reduce_only else "adjust")

        plan.orders.append(PlannedOrder(
            symbol=symbol, side=side, qty_contracts=float(qty_str), qty_str=qty_str,
            usd_delta=delta_usd, reduce_only=reduce_only, mark_price=mark, reason=reason,
        ))

    # RULE 4: reducers first, so peak margin never exceeds the endpoints
    plan.orders.sort(key=lambda o: (not o.reduce_only, -abs(o.usd_delta)))
    return plan


def _get_mark(client, symbol: str, pos: Optional[dict]) -> float:
    """Mark price, preferring the position's own mark, falling back to ticker."""
    if pos:
        for k in ("markPrice", "avgPrice", "entryPrice"):
            v = pos.get(k)
            if v:
                try:
                    f = float(v)
                    if f > 0:
                        return f
                except (TypeError, ValueError):
                    pass
    t = client.get_ticker(symbol) or {}
    for k in ("markPrice", "mark_price", "lastPrice", "last_price", "indexPrice", "index_price"):
        v = t.get(k)
        if v:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                pass
    raise ValueError(f"no usable mark price for {symbol} (ticker keys: {list(t)[:6]})")


def execute_plan(client, plan: ExecutionPlan, dry_run: bool = True,
                 stop_loss_frac: Optional[float] = None) -> dict:
    """
    Send the planned orders.

    dry_run=True logs the exact orders without sending, which is what the
    burn-in harness uses to prove the plan is well-formed before any capital
    is at risk.

    stop_loss_frac, if set, attaches a protective stop at that fraction
    adverse to the mark on newly-opened/increased positions. This is RULE 4
    of the strategy spec carried into live trading; without it an alt short
    has unbounded loss.
    """
    results = {"sent": [], "failed": [], "dry_run": dry_run,
               "stops_set": [], "stops_failed": []}

    for o in plan.orders:
        if dry_run:
            results["sent"].append({"symbol": o.symbol, "side": o.side, "qty": o.qty_str,
                                    "reduceOnly": o.reduce_only, "usd": round(o.usd_delta, 2),
                                    "reason": o.reason, "SENT": False})
            logger.info(f"[DRY RUN] would send: {o!r}")
            continue
        try:
            ids = client.place_order_chunked(
                symbol=o.symbol, side=o.side, qty=o.qty_contracts,
                reduce_only=o.reduce_only,
            )
            results["sent"].append({"symbol": o.symbol, "side": o.side, "qty": o.qty_str,
                                    "reduceOnly": o.reduce_only, "usd": round(o.usd_delta, 2),
                                    "reason": o.reason, "SENT": True, "order_ids": ids})
            logger.info(f"sent {o!r} -> {ids}")
        except Exception as e:                                # noqa: BLE001
            results["failed"].append({"symbol": o.symbol, "side": o.side,
                                      "qty": o.qty_str, "error": f"{type(e).__name__}: {e}"})
            logger.error(f"ORDER FAILED {o!r}: {type(e).__name__}: {e}")

    if stop_loss_frac:
        _attach_stops(client, plan, stop_loss_frac, dry_run, results)

    return results


def _attach_stops(client, plan: ExecutionPlan, frac: float, dry_run: bool, results: dict) -> None:
    """Attach/refresh a protective stop on every position we opened or increased."""
    for o in plan.orders:
        if o.reduce_only or o.reason in ("close", "flatten"):
            continue
        # long -> stop below, short -> stop above
        sl = o.mark_price * (1 - frac) if o.side == "Buy" else o.mark_price * (1 + frac)
        if dry_run:
            results["stops_set"].append({"symbol": o.symbol, "sl": round(sl, 8), "SENT": False})
            logger.info(f"[DRY RUN] would set SL {o.symbol} @ {sl:.8f} ({frac:.0%} adverse)")
            continue
        try:
            ok = client.modify_sl(o.symbol, sl)
            (results["stops_set"] if ok else results["stops_failed"]).append(
                {"symbol": o.symbol, "sl": round(sl, 8), "SENT": bool(ok)})
        except Exception as e:                                # noqa: BLE001
            results["stops_failed"].append({"symbol": o.symbol, "sl": round(sl, 8),
                                            "error": f"{type(e).__name__}: {e}"})
            logger.error(f"SL FAILED {o.symbol} @ {sl}: {e}")


def sync_protective_stops(client, stops_by_symbol: dict, dry_run: bool,
                          tps_by_symbol: Optional[dict] = None) -> dict:
    """
    Push event-sleeve protective stops (and TPs) to the exchange.

    WHY THIS EXISTS -- bug found 2026-08-09.

    _attach_stops() only fires for symbols that appear in an execution plan,
    i.e. positions just opened or increased, and it sets a single STATIC stop
    at LIVE_STOP_LOSS_FRAC from the mark. That is correct for the four
    cross-sectional sleeves, whose stop genuinely is a fixed 60% from entry.

    It is WRONG for the two event sleeves:

      * SHORT uses a CHANDELIER TRAIL that tightens on every 4h bar as new
        highs print. The tracker computed the tightened level and discarded
        it. Live, the position sat on its loose entry stop while the bot's
        internal state believed it was protected at a much tighter level. The
        backtest exits on the trail; the live bot would not have.

      * BOS carries a 2-ATR stop and a 3-ATR take-profit that existed only
        inside the tracker. Between 4h checks the live position had neither.

    This function is called every cycle, not only on rebalance cycles, because
    a trailing stop that is only refreshed when a rebalance happens is not a
    trailing stop.

    Idempotent: modify_sl with an unchanged level is a no-op at the venue.
    Per-symbol failures are collected, never raised -- one rejected amend must
    not prevent the other five from being tightened.
    """
    out = {"stops_synced": [], "stops_failed": [], "tps_synced": [], "tps_failed": []}
    tps_by_symbol = tps_by_symbol or {}

    for symbol, sl in sorted(stops_by_symbol.items()):
        if sl is None or sl <= 0:
            continue
        if dry_run:
            out["stops_synced"].append({"symbol": symbol, "sl": round(sl, 8), "SENT": False})
            logger.info(f"[DRY RUN] would sync trailing SL {symbol} @ {sl:.8f}")
            continue
        try:
            ok = client.modify_sl(symbol, sl)
            (out["stops_synced"] if ok else out["stops_failed"]).append(
                {"symbol": symbol, "sl": round(sl, 8), "SENT": bool(ok)})
        except Exception as e:                                  # noqa: BLE001
            out["stops_failed"].append({"symbol": symbol, "sl": round(sl, 8),
                                        "error": f"{type(e).__name__}: {e}"})
            logger.error(f"trailing SL sync FAILED {symbol} @ {sl}: {e}")

    for symbol, tp in sorted(tps_by_symbol.items()):
        if tp is None or tp <= 0:
            continue
        if dry_run:
            out["tps_synced"].append({"symbol": symbol, "tp": round(tp, 8), "SENT": False})
            continue
        fn = getattr(client, "modify_tp", None)
        if fn is None:
            out["tps_failed"].append({"symbol": symbol, "tp": round(tp, 8),
                                      "error": "client has no modify_tp"})
            continue
        try:
            ok = fn(symbol, tp)
            (out["tps_synced"] if ok else out["tps_failed"]).append(
                {"symbol": symbol, "tp": round(tp, 8), "SENT": bool(ok)})
        except Exception as e:                                  # noqa: BLE001
            out["tps_failed"].append({"symbol": symbol, "tp": round(tp, 8),
                                      "error": f"{type(e).__name__}: {e}"})
    return out
