"""
Discord webhook alerting -- fire-and-forget pattern reused verbatim from the
original bot's main.py::discord_msg(). If DISCORD_WEBHOOK_URL isn't set,
alerts are logged locally instead of silently discarded.
"""
import logging

try:
    import requests
except ImportError:
    requests = None

from . import config

logger = logging.getLogger("Alerts")


def discord_msg(content: str, webhook: str = None):
    """Fire-and-forget Discord notification. Never raises -- an alerting
    failure must not crash the trading loop."""
    url = webhook or config.DISCORD_WEBHOOK_URL
    if not url:
        logger.info(f"[no webhook configured] {content}")
        return
    if requests is None:
        logger.warning(f"[requests unavailable] {content}")
        return
    try:
        requests.post(url, json={"content": content}, timeout=5)
    except Exception as e:
        logger.warning(f"discord_msg failed: {e}")
