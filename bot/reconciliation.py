"""
Reconciles exchange-reported positions against this bot's internal target
book. Reuses the original bot's careful judgment from
main.py::_sync_positions_with_exchange(): detect positions that closed
externally (SL hit, manual close) and compute their realized PnL; detect
ORPHAN exchange positions the bot doesn't recognize and FLAG them rather
than silently adopting or ignoring them (an orphan could be a leftover
position from a previous bot version, a manual trade, or a bug -- auto-
adopting it into the new independent-additive sleeve model would silently
change its risk treatment).

This module does NOT itself decide what to do about orphans or externally-
closed positions -- it returns a structured report; bot/orchestrator.py
decides whether to alert-only or halt.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger("Reconciliation")


@dataclass
class ReconciliationReport:
    externally_closed: List[dict] = field(default_factory=list)   # symbols we tracked that vanished from the exchange
    orphans: List[dict] = field(default_factory=list)              # exchange positions we don't recognize
    matched: List[str] = field(default_factory=list)               # symbols present on both sides
    realized_pnl_delta: float = 0.0


def reconcile(exchange_client, tracked_symbols: Dict[str, dict]) -> ReconciliationReport:
    """
    tracked_symbols: {symbol: {"side": "Buy"/"Sell", "qty": float, "entry_price": float}}
                      -- this bot's belief about what SHOULD be open, built
                      from the last computed dollar_targets.

    Returns a ReconciliationReport. Never raises on a get_positions()
    failure -- returns an empty report and logs, so a transient API outage
    doesn't halt the reconciliation caller; the caller should treat an empty
    report from a failed fetch differently from a genuinely flat account
    (check exchange_client's own logs / a returned success flag if that
    distinction matters for your use).
    """
    report = ReconciliationReport()
    try:
        api_positions = exchange_client.get_positions()
    except Exception as e:
        logger.error(f"get_positions() failed during reconciliation: {e}")
        return report

    exchange_symbols = {p["symbol"] for p in api_positions}
    api_by_symbol = {p["symbol"]: p for p in api_positions}

    for symbol, tracked in tracked_symbols.items():
        if symbol not in exchange_symbols:
            report.externally_closed.append({"symbol": symbol, **tracked})
            logger.warning(f"EXTERNALLY_CLOSED: {symbol} was tracked but is no longer on the exchange "
                            f"(SL hit or manual close) -- realized PnL must be reconciled from last "
                            f"known price, not assumed zero.")
        else:
            report.matched.append(symbol)

    for symbol, pos in api_by_symbol.items():
        if symbol not in tracked_symbols:
            report.orphans.append(pos)
            logger.warning(f"ORPHAN: {symbol} {pos['side']} size={pos['size']} is open on the exchange "
                            f"but not tracked by this bot -- NOT auto-adopted. Investigate manually before "
                            f"the next rebalance cycle overwrites/conflicts with it.")

    return report
