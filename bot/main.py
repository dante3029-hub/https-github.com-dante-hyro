#!/usr/bin/env python3
"""
CLI entrypoint for the HyroTrader 200k eval bot.

    python3 bot/main.py                     # dry-run against replay data (default, safe)
    python3 bot/main.py --dry-run            # explicit, same as above
    python3 bot/main.py --live               # REFUSES to run -- see below
    python3 bot/main.py --cycles 3           # run N cycles then exit (dry-run loop, for testing)

API keys are read from environment variables ONLY:
    BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_USE_DEMO (default true), DISCORD_WEBHOOK_URL
Never hardcode secrets in this file or config.py -- see config.py docstring.

--live is intentionally a hard stop right now. LiveDataFeed.get_snapshot()
raises NotImplementedError the moment it's called (live order book /
cross-venue basis / OI / taker-flow ingestion was never built -- confirmed
in this session's audit, not a guess). This flag exists so the CLI surface
is ready for Phase 5+, not to suggest the bot can trade live today.
"""
import os
import sys
import argparse
import logging
import datetime as dt

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from bot import config
from bot.data_feed import ReplayDataFeed, LiveDataFeed
from bot.orchestrator import HyroTraderBot
from bot.alerts import discord_msg

logger = logging.getLogger("main")


def build_parser():
    p = argparse.ArgumentParser(description="HyroTrader 200k eval bot")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                       help="(default) replay historical data, compute targets, place NO orders")
    mode.add_argument("--live", action="store_true", default=False,
                       help="attempt live trading -- WILL currently raise, live data ingestion not built")
    p.add_argument("--execute", action="store_true", default=False,
                   help="PLACE REAL ORDERS using ReplayDataFeed over live_refresh.py-updated "
                        "CSVs. NOT --live: LiveDataFeed (streaming ingestion) is still a stub. "
                        "This path is only valid if live_refresh.py has run THIS cycle -- "
                        "stale panels mean trading on stale signals.")
    p.add_argument("--cycles", type=int, default=1, help="number of cycles to run before exiting")
    p.add_argument("--as-of", type=str, default=None,
                   help="YYYY-MM-DD to replay as-of (dry-run only); default = latest available data")
    return p


def main():
    args = build_parser().parse_args()
    dry_run = not args.live

    as_of_date = dt.date.fromisoformat(args.as_of) if args.as_of else None

    if args.live:
        logger.warning("`--live` requested. LiveDataFeed will raise NotImplementedError on first "
                        "get_snapshot() call -- this is intentional, see main.py / data_feed.py docstrings.")
        data_feed = LiveDataFeed()
        exchange_client = None
        if config.BYBIT_API_KEY and config.BYBIT_API_SECRET:
            from bot.exchange_client import HardenedBybitClient, ExchangeMode
            mode = ExchangeMode.DEMO if config.BYBIT_USE_DEMO else ExchangeMode.MAINNET
            exchange_client = HardenedBybitClient(config.BYBIT_API_KEY, config.BYBIT_API_SECRET, mode=mode)
    else:
        data_feed = ReplayDataFeed()
        exchange_client = None
        if args.execute:
            # Orders ON, data from refreshed CSVs. Guard: refuse if panel is stale.
            from live_refresh import panel_staleness, MAX_STALE_HOURS
            import os as _os
            for _panel in ("clean_panel",):
                _newest, _age = panel_staleness(_os.path.join(_os.environ.get(
                    "HYRO_WORKSPACE", "."), _panel))
                if _newest is None or _age > MAX_STALE_HOURS:
                    raise SystemExit(f"REFUSING TO EXECUTE: {_panel} newest bar {_newest} "
                                     f"({_age:.0f}h old, max {MAX_STALE_HOURS}h). "
                                     f"Run live_refresh.py first.")
            if not (config.BYBIT_API_KEY and config.BYBIT_API_SECRET):
                raise SystemExit("REFUSING TO EXECUTE: no API credentials in environment")
            from bot.exchange_client import HardenedBybitClient, ExchangeMode
            _mode = ExchangeMode.DEMO if config.BYBIT_USE_DEMO else ExchangeMode.MAINNET
            exchange_client = HardenedBybitClient(config.BYBIT_API_KEY,
                                                  config.BYBIT_API_SECRET, mode=_mode)
            dry_run = False
            logger.warning(f"EXECUTE MODE: orders WILL be placed on {_mode} "
                           f"({exchange_client.base_url})")

    bot = HyroTraderBot(data_feed=data_feed, exchange_client=exchange_client, dry_run=dry_run)

    for i in range(args.cycles):
        try:
            report = bot.run_cycle(as_of_date=as_of_date)
            logger.info(f"Cycle {report['cycle']} complete: gross_notional=${report.get('gross_notional', 0):,.0f} "
                        f"flags={len(report.get('flags', []))} rebalanced={report.get('rebalanced_sleeves', [])}")
            for f in report.get("flags", []):
                logger.warning(f"FLAG: {f}")
        except NotImplementedError as e:
            logger.error(f"HALTED: {e}")
            discord_msg(f"[HyroTraderBot] HALTED on cycle {i}: {e}")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"Unhandled exception on cycle {i}: {e}")
            discord_msg(f"[HyroTraderBot] CRASHED on cycle {i}: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
