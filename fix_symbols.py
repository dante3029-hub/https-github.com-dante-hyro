#!/usr/bin/env python3
"""
fix_symbols.py — add Bybit's symbol aliases to to_bybit_symbol().

THE BUG, found by the demo burn-in:

    GET /v5/market/tickers retCode=10001 msg=params error: symbol invalid
    execution planning errors: 1000SHIBUSDT mark price unavailable

`to_bybit_symbol` was `f"{coin}USDT"`, and its own docstring warned that a coin
which does not map that simply would produce a symbol Bybit rejects. It was
right: Bybit lists the 1000x SHIB contract as **SHIB1000USDT** -- multiplier on
the RIGHT, not the left. Binance uses 1000SHIBUSDT, which is where the coin
names came from.

check_symbols.py verified all 24 against Bybit's live instrument list:
23 map directly, only 1000SHIB does not.

WHY THIS MATTERED
The mark price lookup failed, so 1000SHIB could never be priced, so it was
dropped from every execution plan -- one of 24 coins silently absent from the
book every cycle. The orchestrator DID log it as an explicit error rather than
skipping quietly, which is the only reason it was visible.

    python3 fix_symbols.py --check
    python3 fix_symbols.py
"""
import shutil
import sys

PATH = "bot/config.py"
BAK = PATH + ".bak_symbols"

OLD = '''def to_bybit_symbol(coin: str) -> str:'''

NEW = '''# Coins whose Bybit contract name is NOT simply <coin>USDT.
#
# Verified against Bybit's live instrument list (890 linear perps) by
# check_symbols.py: 23 of 24 map directly, 1000SHIB does not. Bybit puts the
# multiplier on the RIGHT (SHIB1000USDT); Binance puts it on the left
# (1000SHIBUSDT), and the coin names in this project came from Binance.
#
# Re-run check_symbols.py whenever a coin is added -- a wrong symbol fails at
# the mark-price lookup and silently drops that coin from every execution plan.
SYMBOL_ALIAS = {
    "1000SHIB": "SHIB1000USDT",
}


def to_bybit_symbol(coin: str) -> str:'''

OLD_BODY = '''    return f"{coin}USDT"'''

NEW_BODY = '''    if coin in SYMBOL_ALIAS:
        return SYMBOL_ALIAS[coin]
    return f"{coin}USDT"'''

if __name__ == "__main__":
    try:
        s = open(PATH).read()
    except FileNotFoundError:
        print(f"{PATH} not found -- run this from the repo root")
        sys.exit(1)

    if "SYMBOL_ALIAS" in s:
        print("already applied")
        sys.exit(0)
    if s.count(OLD) != 1 or s.count(OLD_BODY) != 1:
        print(f"anchors not unique (def={s.count(OLD)}, body={s.count(OLD_BODY)})"
              " -- not patching")
        sys.exit(1)
    if "--check" in sys.argv:
        print("ready to patch to_bybit_symbol")
        sys.exit(0)

    shutil.copy2(PATH, BAK)
    s = s.replace(OLD, NEW, 1).replace(OLD_BODY, NEW_BODY, 1)
    open(PATH, "w").write(s)
    print(f"patched {PATH}  (backup: {BAK})")
    print("\nverify:")
    print("  python3 -c \"from bot.config import to_bybit_symbol as f; "
          "print(f('1000SHIB'), f('SOL'), f('1000PEPE'))\"")
    print("  -> SHIB1000USDT SOLUSDT 1000PEPEUSDT")
