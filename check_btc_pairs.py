#!/usr/bin/env python3
"""Which BTC pairs actually exist on Binance spot? One call, no downloading.

Binance has delisted many BTC pairs over the years, so this checks the full
exchange listing before committing to a fetch.
"""
import json, os, glob, urllib.request

SRC = os.path.expanduser("~/bot_hyrotrader_v1/taker_data")
UNPREFIX = {"1000PEPE": "PEPE", "1000SHIB": "SHIB", "1000BONK": "BONK",
            "1000FLOKI": "FLOKI", "1000RATS": "RATS"}

req = urllib.request.Request("https://api.binance.com/api/v3/exchangeInfo",
                             headers={"User-Agent": "check/1.0"})
with urllib.request.urlopen(req, timeout=30) as r:
    info = json.loads(r.read().decode())

live = {s["symbol"] for s in info["symbols"]
        if s["status"] == "TRADING" and s["quoteAsset"] == "BTC"}
print(f"Binance spot has {len(live)} live BTC pairs in total\n")

alts = sorted({os.path.basename(f).replace("_1h.csv", "")
               for f in glob.glob(f"{SRC}/*_1h.csv")} - {"BTC"})
have, miss = [], []
for c in alts:
    sym = UNPREFIX.get(c, c) + "BTC"
    (have if sym in live else miss).append(c)

print(f"OF YOUR {len(alts)} ALTS:")
print(f"  {len(have)} have a live BTC pair:")
print("   ", " ".join(have))
print(f"\n  {len(miss)} do not:")
print("   ", " ".join(miss))
