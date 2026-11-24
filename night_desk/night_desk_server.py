#!/usr/bin/env python3
"""
night_desk_server.py -- serve the Night Desk dashboard and relay its
read-only Hyperliquid queries. Python 3.8+, standard library only.

WHY THIS EXISTS
The dashboard can call Hyperliquid's public API straight from a browser, but
some browsers, extensions and networks block cross-origin calls, and a page
opened from disk behaves differently from one served over http. Running this
on the DigitalOcean box sidesteps all of it: the page and its data come from
the same origin, and only this process talks to Hyperliquid.

WHAT IT WILL AND WILL NOT DO
  * Serves exactly one file: night_desk.html (from the same directory).
  * Forwards POST /info to https://api.hyperliquid.xyz/info, but ONLY for a
    whitelist of read-only query types. It has no route to /exchange, holds
    no keys, and cannot place, modify or cancel an order.
  * Binds to 127.0.0.1 by default. Reach it with an SSH tunnel:
        ssh -L 8787:127.0.0.1:8787 root@<your-server>
        then open http://localhost:8787 on your own machine.
    Binding to 0.0.0.0 exposes it to the internet; the data is public, but
    anyone could use your server's IP to hit Hyperliquid, so don't.

ROBUSTNESS
  * Request bodies capped at 8 KB and must be JSON objects.
  * Upstream calls time out after 10 s; failures return 502 with a reason.
  * Short TTL cache so several open tabs do not multiply upstream calls.
  * Global token bucket (default 240 requests/min) keeps well inside
    Hyperliquid's per-IP limits, which also cover any bot on the same box.
  * Clean shutdown on SIGINT / SIGTERM.

USAGE
    python3 night_desk_server.py                 # 127.0.0.1:8787
    python3 night_desk_server.py --port 9000
    python3 night_desk_server.py --selftest      # offline checks, no network
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = logging.getLogger("night_desk")
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "night_desk.html")
UPSTREAM = "https://api.hyperliquid.xyz/info"
MAX_BODY = 8 * 1024
UPSTREAM_TIMEOUT = 10.0

# Read-only info queries the dashboard uses (plus two harmless extras).
# Anything else -- including every /exchange action -- is refused.
ALLOWED_TYPES = {
    "clearinghouseState", "userFills", "portfolio", "metaAndAssetCtxs",
    "candleSnapshot", "openOrders", "frontendOpenOrders",
}
# Seconds a response may be reused. Market-wide data changes slowly enough to
# share across tabs; account data is kept almost live.
TTL = {"metaAndAssetCtxs": 15.0, "candleSnapshot": 20.0, "portfolio": 20.0}
DEFAULT_TTL = 3.0
ADDRESS_FIELDS = {"user"}


class TokenBucket:
    """Thread-safe token bucket: `rate` tokens per minute, burst `rate`/4."""

    def __init__(self, per_minute: int):
        self.capacity = max(1.0, per_minute / 4)
        self.tokens = self.capacity
        self.fill = per_minute / 60.0
        self.at = time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.at) * self.fill)
            self.at = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


class Cache:
    """Tiny TTL cache keyed by the canonical request body."""

    def __init__(self, max_items: int = 256):
        self.data: dict[str, tuple[float, int, bytes]] = {}
        self.max_items = max_items
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            hit = self.data.get(key)
            if hit and hit[0] > time.monotonic():
                return hit[1], hit[2]
            if hit:
                del self.data[key]
            return None

    def put(self, key: str, ttl: float, status: int, body: bytes) -> None:
        with self.lock:
            if len(self.data) >= self.max_items:
                # drop the entry closest to expiry; bounded memory for months of uptime
                oldest = min(self.data, key=lambda k: self.data[k][0])
                del self.data[oldest]
            self.data[key] = (time.monotonic() + ttl, status, body)


def validate(payload) -> str | None:
    """Return an error message, or None if the request may be forwarded."""
    if not isinstance(payload, dict):
        return "body must be a JSON object"
    qtype = payload.get("type")
    if qtype not in ALLOWED_TYPES:
        return f"query type {qtype!r} is not allowed"
    for field in ADDRESS_FIELDS & payload.keys():
        v = payload[field]
        if not (isinstance(v, str) and len(v) == 42 and v.startswith("0x")
                and all(c in "0123456789abcdefABCDEF" for c in v[2:])):
            return f"{field} must be a 0x address"
    if qtype == "candleSnapshot":
        req = payload.get("req")
        if not isinstance(req, dict) or not isinstance(req.get("coin"), str):
            return "candleSnapshot needs req.coin"
        if req.get("interval") not in {"1m", "5m", "15m", "1h", "4h", "1d"}:
            return "unsupported candle interval"
        for k in ("startTime", "endTime"):
            if not isinstance(req.get(k), (int, float)):
                return f"candleSnapshot needs numeric req.{k}"
    return None


def forward(body: bytes, upstream: str) -> tuple[int, bytes]:
    """POST to Hyperliquid. Returns (status, body); never raises."""
    req = urllib.request.Request(
        upstream, data=body, method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "night-desk/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as r:
            return r.status, r.read(5 * 1024 * 1024)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read(2048)
        except Exception:  # noqa: BLE001 -- a broken error body must not mask the status
            detail = b""
        LOG.warning("upstream HTTP %s: %s", e.code, detail[:200])
        return e.code, json.dumps({"error": f"upstream HTTP {e.code}"}).encode()
    except urllib.error.URLError as e:
        LOG.warning("upstream unreachable: %s", e.reason)
        return 502, json.dumps({"error": f"upstream unreachable: {e.reason}"}).encode()
    except TimeoutError:
        LOG.warning("upstream timed out")
        return 504, json.dumps({"error": "upstream timed out"}).encode()
    except OSError as e:
        LOG.warning("upstream socket error: %s", e)
        return 502, json.dumps({"error": f"upstream error: {e}"}).encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "NightDesk/1.0"
    bucket: TokenBucket
    cache: Cache
    upstream: str

    def log_message(self, fmt, *args):  # route through logging, not stderr spam
        LOG.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status: int, msg: str) -> None:
        self._send(status, json.dumps({"error": msg}).encode())

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html", "/night_desk.html"):
            try:
                with open(PAGE, "rb") as f:
                    page = f.read()
            except OSError as e:
                LOG.error("cannot read %s: %s", PAGE, e)
                return self._err(500, "dashboard file missing next to the server")
            return self._send(200, page, "text/html; charset=utf-8")
        if path == "/fund.json":
            # The fund registry: bots, accounts, strategies and their status.
            # Edit this file on the server; the page re-reads it on load.
            try:
                with open(os.path.join(HERE, "fund.json"), "rb") as f:
                    raw = f.read()
                json.loads(raw)            # refuse to serve a broken registry
            except FileNotFoundError:
                return self._err(404, "no fund.json next to the server")
            except (OSError, ValueError) as e:
                LOG.error("fund.json unreadable: %s", e)
                return self._err(500, f"fund.json is not valid JSON: {e}")
            return self._send(200, raw)
        if path == "/health":
            return self._send(200, json.dumps({"relay": "night-desk", "ok": True}).encode())
        return self._err(404, "not found")

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/info":
            return self._err(404, "only /info is relayed")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._err(400, "bad Content-Length")
        if length <= 0 or length > MAX_BODY:
            return self._err(413, f"body must be 1..{MAX_BODY} bytes")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return self._err(400, "body is not valid JSON")
        problem = validate(payload)
        if problem:
            return self._err(400, problem)

        key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        hit = self.cache.get(key)
        if hit:
            return self._send(hit[0], hit[1])
        if not self.bucket.take():
            return self._err(429, "relay rate limit reached, retry shortly")
        status, body = forward(key.encode(), self.upstream)
        if status == 200:
            self.cache.put(key, TTL.get(payload["type"], DEFAULT_TTL), status, body)
        self._send(status, body)

    def do_PUT(self):
        self._err(405, "method not allowed")

    do_DELETE = do_PATCH = do_PUT


def selftest() -> int:
    ok = lambda b: validate(b) is None  # noqa: E731
    addr = "0x" + "ab" * 20
    assert ok({"type": "clearinghouseState", "user": addr})
    assert ok({"type": "metaAndAssetCtxs"})
    assert ok({"type": "candleSnapshot", "req": {"coin": "BTC", "interval": "15m", "startTime": 1, "endTime": 2}})
    assert not ok({"type": "order", "action": {}}), "an exchange action slipped through"
    assert not ok({"type": "clearinghouseState", "user": "0xnothex"}), "bad address accepted"
    assert not ok(["not", "an", "object"])
    assert not ok({"type": "candleSnapshot", "req": {"coin": "BTC", "interval": "3m", "startTime": 1, "endTime": 2}})
    b = TokenBucket(60)
    taken = sum(b.take() for _ in range(100))
    assert taken == 15, f"burst should be 15, got {taken}"
    c = Cache(max_items=2)
    c.put("a", 10, 200, b"1"); c.put("b", 10, 200, b"2"); c.put("c", 10, 200, b"3")
    assert len(c.data) == 2, "cache is not bounded"
    c.put("x", -1, 200, b"old")
    assert c.get("x") is None, "expired entry served"
    assert os.path.exists(PAGE), f"{PAGE} missing"
    print("selftest: whitelist, address check, rate limit, bounded cache, page present -- OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--rate", type=int, default=240, help="max upstream requests per minute")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if a.selftest:
        return selftest()
    if not os.path.exists(PAGE):
        LOG.error("night_desk.html must sit next to this script (%s)", HERE)
        return 1
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        LOG.warning("binding to %s exposes the relay beyond this machine", a.host)

    Handler.bucket = TokenBucket(a.rate)
    Handler.cache = Cache()
    Handler.upstream = a.upstream
    try:
        httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError as e:
        LOG.error("cannot listen on %s:%s -- %s", a.host, a.port, e)
        return 1
    httpd.daemon_threads = True

    def stop(signum, _frame):
        LOG.info("signal %s, shutting down", signum)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    LOG.info("Night Desk on http://%s:%s  (relaying read-only queries to %s)", a.host, a.port, a.upstream)
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
        LOG.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
