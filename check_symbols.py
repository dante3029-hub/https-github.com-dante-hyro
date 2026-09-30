#!/usr/bin/env python3
"""
check_symbols.py — does every coin in the book map to a real Bybit contract?

`to_bybit_symbol` is `f"{coin}USDT"`, and its own docstring warns that a coin
which does not map that simply will produce a symbol Bybit rejects. The demo
burn-in proved it:

    GET /v5/market/tickers retCode=10001 msg=params error: symbol invalid
    execution planning errors: 1000SHIBUSDT mark price unavailable

Bybit lists that contract as SHIB1000USDT -- the multiplier is on the RIGHT.
This checks all 24 at once rather than fixing them one failure at a time.

    python3 check_symbols.py
"""
import json
import sys
import urllib.request

CORE24 = ["1000PEPE", "1000RATS", "1000SHIB", "AAVE", "ADA", "AVAX", "BCH",
          "BNB", "DOGE", "DOT", "ETH", "FIL", "LDO", "LINK", "LTC", "NEAR",
          "SOL", "SUI", "TRX", "UNI", "WLD", "XLM", "XRP", "ZEC"]

req = urllib.request.Request(
    "https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000",
    headers={"User-Agent": "symcheck/1.0"})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.loads(r.read().decode())

live = {i["symbol"] for i in data.get("result", {}).get("list", [])
        if i.get("status") == "Trading"}
print(f"Bybit has {len(live)} linear perps trading\n")

ok, bad = [], []
for c in CORE24:
    direct = f"{c}USDT"
    if direct in live:
        ok.append((c, direct))
        continue
    # try the alternatives: multiplier on the right, or bare
    cands = []
    if c.startswith("1000"):
        base = c[4:]
        cands += [f"{base}1000USDT", f"{base}USDT"]
    cands += [f"1000{c}USDT"]
    found = next((x for x in cands if x in live), None)
    bad.append((c, direct, found))

print(f"{len(ok)}/{len(CORE24)} map directly with +USDT")
if bad:
    print(f"\n{len(bad)} DO NOT:")
    for c, tried, found in bad:
        print(f"  {c:<12} {tried:<18} -> {found or 'NO MATCH FOUND'}")
    print("\n  add to SYMBOL_ALIAS in bot/config.py:")
    print("  SYMBOL_ALIAS = {")
    for c, _, found in bad:
        if found:
            print(f'      "{c}": "{found}",')
    print("  }")
else:
    print("every coin maps cleanly")
sys.exit(1 if bad else 0)
