#!/usr/bin/env python3
"""
venues.py -- the crypto sportsbook / casino universe, as DATA not a list.

WHY THIS IS NOT JUST A LIST OF NAMES. An arbitrage scanner that treats
every brand as an independent price source will report arbs that do not
exist, because dozens of these brands are the same book wearing a
different logo. There are two distinct kinds of collusion to collapse:

  OPERATOR   same company, same risk desk, same limiting decision, and
             they net your positions internally. Betting both sides
             across two brands of one operator is betting against
             yourself and paying the vig twice.

  ENGINE     different operators, same ODDS PROVIDER. Betby prices
             BC.Game, Rollbit, Roobet, FortuneJack and others. Their
             prices move together because one feed drives them, and a
             documented BC.Game dispute shows that provider applying
             limits ACROSS the operators it serves. So an "arb" between
             two Betby books is usually a stale-feed artifact, and the
             provider can see both sides of it.

Both are collapsed by independent_sources() below. The 1xBet platform
alone is cloned across a dozen brands -- a scanner that misses that will
light up with phantom opportunities all day.

FIELD MEANINGS
  operator   corporate grouping. None = independent or unknown.
  engine     odds/pricing provider where known. Books sharing an engine
             are correlated price sources even under different operators.
  api        'full'    documented odds AND bet-placement API
             'odds'    odds/feed API only
             'none'    no public API (manual or browser only)
             'exchange' order book with a public trading API
  feeds      which commercial aggregators carry it, so you can get the
             odds programmatically even with api='none'.
  payout     'good' | 'mixed' | 'poor' | 'unknown' -- from documented
             complaint patterns, not vibes. 'poor' means a repeated,
             documented pattern of withheld or voided winnings.
  arb_clause True where the terms explicitly void bets for arbitrage,
             matched betting, or hedging across accounts/platforms.
  bot_clause True where automated/scripted betting is prohibited.

SOURCING. Compiled October 2026 from operator terms, ACMA publications
and complaint-board records. Entries marked payout='unknown' were not
verified to a citable standard -- absence of a bad record is not a good
record. Terms change; re-check quarterly. The ACMA blocked-list is
published each quarter and `acma_blocked` will go stale fastest.

USAGE
    python3 -u venues.py                     # the table
    python3 -u venues.py --api               # only venues you can automate
    python3 -u venues.py --groups            # collapse into price sources
    python3 -u venues.py --selftest
"""
from __future__ import annotations

import argparse

# name, operator, engine, product, api, feeds, payout, arb_clause, bot_clause, acma_blocked
V = [
    # ---- the big crypto books -------------------------------------------
    ("Stake.com",      "medium_rare", None,    "both",  "none", "oddspapi",
     "good",    False, True,  False),
    ("Primedice",      "medium_rare", None,    "casino", "none", "",
     "good",    False, True,  False),
    ("Cloudbet",       None,          None,    "both",  "full", "betstamp",
     "good",    False, False, False),   # bot clause exempts the API
    ("Roobet",         None,          "betby", "both",  "none", "",
     "mixed",   True,  True,  False),
    ("Shuffle",        "natural_nine", None,   "both",  "none", "",
     "unknown", True,  True,  False),
    ("BC.Game",        None,          "betby", "both",  "none", "",
     "mixed",   True,  True,  False),
    ("Rollbit",        "bull_gaming", "betby", "both",  "none", "",
     "poor",    True,  True,  False),
    ("Duelbits",       "liquid_ent",  None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("Thunderpick",    None,          None,    "both",  "none", "",
     "unknown", True,  True,  False),
    ("Betplay.io",     "lama_tech",   None,    "both",  "none", "",
     "unknown", False, False, False),
    ("FortuneJack",    None,          "betby", "both",  "none", "",
     "unknown", False, True,  False),
    ("Gamdom",         None,          None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("BetFury",        None,          None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("Trustdice",      None,          None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("Wild.io",        None,          None,    "casino", "none", "",
     "unknown", False, True,  False),
    ("Bitsler",        None,          None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("Nitrogen Sports", None,         None,    "sports", "none", "",
     "unknown", False, True,  False),
    ("Chips.gg",       None,          None,    "casino", "none", "",
     "unknown", False, True,  False),
    ("Vave",           None,          None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("Mega Dice",      None,          None,    "both",  "none", "oddspapi",
     "unknown", False, True,  False),
    ("Flush.com",      None,          None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("Coinplay",       None,          None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("Betpanda",       None,          None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("Crashino",       None,          None,    "both",  "none", "",
     "unknown", False, True,  False),
    ("500.casino",     None,          "betby", "both",  "none", "",
     "mixed",   True,  True,  False),

    # ---- Yolo Group (one operator, several brands) ----------------------
    ("Sportsbet.io",   "yolo",        None,    "both",  "none", "",
     "mixed",   False, True,  False),
    ("Bitcasino.io",   "yolo",        None,    "casino", "none", "",
     "mixed",   False, True,  False),
    ("Livecasino.io",  "yolo",        None,    "casino", "none", "",
     "unknown", False, True,  False),

    # ---- SoftSwiss / Direx platform family ------------------------------
    ("BitStarz",       "direx",       None,    "casino", "none", "",
     "good",    False, True,  False),
    ("mBit Casino",    "direx",       None,    "casino", "none", "",
     "unknown", False, True,  False),
    ("7Bit Casino",    "direx",       None,    "casino", "none", "",
     "unknown", False, True,  False),

    # ---- the 1xBet platform: ONE price source, many logos ---------------
    ("1xBet",          "1xbet",       "1xbet", "both",  "none", "odds_api",
     "poor",    True,  True,  False),
    ("Melbet",         "1xbet",       "1xbet", "both",  "none", "oddspapi",
     "poor",    True,  True,  False),
    ("22Bet",          "1xbet",       "1xbet", "both",  "none", "oddspapi",
     "poor",    True,  True,  False),
    ("Betwinner",      "1xbet",       "1xbet", "both",  "none", "",
     "poor",    True,  True,  False),
    ("Megapari",       "1xbet",       "1xbet", "both",  "none", "",
     "poor",    True,  True,  False),
    ("Paripesa",       "1xbet",       "1xbet", "both",  "none", "",
     "poor",    True,  True,  False),
    ("1xBit",          "1xbet",       "1xbet", "both",  "none", "",
     "poor",    True,  True,  False),
    ("888Starz",       "1xbet",       "1xbet", "both",  "none", "",
     "poor",    True,  True,  False),
    ("Helabet",        "1xbet",       "1xbet", "both",  "none", "",
     "poor",    True,  True,  False),
    ("Linebet",        "1xbet",       "1xbet", "both",  "none", "",
     "poor",    True,  True,  False),
    ("Spinbetter",     "1xbet",       "1xbet", "both",  "none", "",
     "poor",    True,  True,  False),

    # ---- Bodog group (one operator) -------------------------------------
    ("Bovada",         "bodog",       None,    "sports", "none", "odds_api",
     "mixed",   False, True,  False),
    ("Bodog",          "bodog",       None,    "sports", "none", "",
     "mixed",   False, True,  False),
    ("Ignition",       "bodog",       None,    "casino", "none", "",
     "mixed",   False, True,  False),

    # ---- US-facing offshore, carried by The Odds API --------------------
    ("BetOnline",      None,          None,    "sports", "none", "odds_api",
     "mixed",   False, True,  False),
    ("MyBookie",       None,          None,    "sports", "none", "odds_api",
     "mixed",   False, True,  False),
    ("BetUS",          None,          None,    "sports", "none", "odds_api",
     "unknown", False, True,  False),
    ("Everygame",      None,          None,    "sports", "none", "odds_api",
     "unknown", False, True,  False),
    ("LowVig",         None,          None,    "sports", "none", "odds_api",
     "unknown", False, True,  False),

    # ---- ACMA-blocked (Australia), still real venues elsewhere ----------
    ("JackBit",        None,          None,    "both",  "none", "odds_api_io",
     "unknown", False, True,  True),
    ("MyStake",        None,          None,    "both",  "none", "oddspapi",
     "unknown", False, True,  True),
    ("Betwhale",       None,          None,    "both",  "none", "",
     "unknown", False, True,  True),
    ("Jacks Club",     None,          None,    "both",  "none", "",
     "unknown", False, True,  True),
    ("Leon",           None,          None,    "both",  "none", "",
     "unknown", False, True,  True),
    ("N1Bet",          None,          None,    "both",  "none", "",
     "unknown", False, True,  True),
    ("Metaspins",      None,          None,    "both",  "none", "",
     "unknown", False, True,  True),
    ("Rain.gg",        None,          None,    "casino", "none", "",
     "unknown", False, True,  True),
    ("CoinPoker",      None,          None,    "poker", "none", "",
     "unknown", False, True,  True),
    ("Spinit",         None,          None,    "casino", "none", "",
     "unknown", False, True,  True),

    # ---- EXCHANGES / prediction markets: order books, real APIs ---------
    ("SX Bet",         None,          None,    "exchange", "exchange",
     "opticodds", "good",    False, False, False),
    ("Polymarket",     None,          None,    "exchange", "exchange",
     "odds_api",  "good",    False, False, True),
    ("Kalshi",         None,          None,    "exchange", "exchange",
     "odds_api",  "good",    False, False, False),
    ("Betfair",        "flutter_ex",  None,    "exchange", "exchange",
     "odds_api",  "good",    False, False, False),
    ("Smarkets",       None,          None,    "exchange", "exchange",
     "odds_api",  "good",    False, False, False),
    ("Matchbook",      None,          None,    "exchange", "exchange",
     "odds_api",  "good",    False, False, False),
    ("Betdaq",         None,          None,    "exchange", "exchange",
     "",          "good",    False, False, False),
    ("Novig",          None,          None,    "exchange", "exchange",
     "odds_api",  "good",    False, False, False),
    ("ProphetX",       None,          None,    "exchange", "exchange",
     "odds_api",  "good",    False, False, False),
    ("Overtime",       None,          None,    "amm",      "odds",
     "",          "unknown", False, False, False),
    ("Azuro",          None,          None,    "amm",      "odds",
     "",          "unknown", False, False, False),
]

COLS = ("name", "operator", "engine", "product", "api", "feeds",
        "payout", "arb_clause", "bot_clause", "acma_blocked")


def rows():
    return [dict(zip(COLS, r)) for r in V]


def independent_sources(names):
    """Collapse a set of venues into INDEPENDENT price sources.

    Two books count as one source if they share an operator OR an odds
    engine. This is the function that stops the scanner reporting an arb
    between Melbet and 1xBet, or between BC.Game and Rollbit.

    Returns {source_key: [venue names]}.
    """
    # UNION-FIND, not "pick operator, else engine".
    #
    # The first version chose the operator key when one existed and fell
    # back to the engine, which split the Betby group: Rollbit has an
    # operator (bull_gaming) so it keyed on that, while BC.Game and Roobet
    # keyed on the engine. Three Betby books, two "independent" sources,
    # and the scanner would happily arb Rollbit against BC.Game on prices
    # driven by the same feed.
    #
    # Sharing an operator and sharing an engine are two different
    # equivalence relations and correlation flows through both. If A and B
    # share an operator and B and C share an engine, all three move
    # together and all three are one source. That is connected components,
    # and nothing less than that is correct.
    by = {r["name"].lower(): r for r in rows()}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    resolved, unknown = [], []
    for n in names:
        r = by.get(str(n).lower())
        (resolved if r else unknown).append(r if r else n)
    for r in resolved:
        find(r["name"])
        if r["operator"]:
            union(r["name"], f"op:{r['operator']}")
        if r["engine"]:
            union(r["name"], f"eng:{r['engine']}")

    out = {}
    for r in resolved:
        out.setdefault(find(r["name"]), []).append(r["name"])
    # name each component after its largest shared attribute, for display
    named = {}
    for root, members in out.items():
        ops = {by[m.lower()]["operator"] for m in members} - {None}
        eng = {by[m.lower()]["engine"] for m in members} - {None}
        key = (f"eng:{sorted(eng)[0]}" if len(members) > 1 and eng
               else f"op:{sorted(ops)[0]}" if ops
               else f"solo:{members[0]}")
        while key in named:
            key += "+"
        named[key] = sorted(members)
    for n in unknown:
        named[f"unknown:{n}"] = [n]
    return named


def selftest() -> int:
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"    [{'ok' if cond else 'FAIL'}] {name}")

    print("  collapsing brands into independent price sources:")
    g = independent_sources(["1xBet", "Melbet", "22Bet"])
    chk("the 1xBet clones collapse to ONE source", len(g) == 1)
    g = independent_sources(["BC.Game", "Rollbit", "Roobet"])
    chk("Betby-powered books collapse to ONE source", len(g) == 1)
    g = independent_sources(["Stake.com", "Primedice"])
    chk("Stake and Primedice collapse (same operator)", len(g) == 1)
    g = independent_sources(["Sportsbet.io", "Bitcasino.io"])
    chk("Yolo Group brands collapse", len(g) == 1)
    g = independent_sources(["Cloudbet", "Betfair", "SX Bet"])
    chk("genuinely independent venues stay separate", len(g) == 3)
    g = independent_sources(["1xBet", "Melbet", "Cloudbet"])
    chk("mixed set -> 2 sources, not 3", len(g) == 2)

    n_api = sum(1 for r in rows() if r["api"] != "none")
    n_tot = len(rows())
    print()
    print(f"  {n_tot} venues, {n_api} with any API, "
          f"{n_tot - n_api} manual-only")
    print(f"  {sum(1 for r in rows() if r['payout']=='poor')} flagged poor, "
          f"{sum(1 for r in rows() if r['payout']=='unknown')} unverified")
    print()
    print(f"  {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true",
                    help="only venues with some API")
    ap.add_argument("--groups", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.groups:
        g = independent_sources([r["name"] for r in rows()])
        print(f"  {len(rows())} brands -> {len(g)} independent price"
              f" sources\n")
        for k, v in sorted(g.items(), key=lambda x: -len(x[1])):
            if len(v) > 1:
                print(f"    {k:<22}{len(v):>3}  {', '.join(v)}")
        return 0
    rs = [r for r in rows() if (not a.api or r["api"] != "none")]
    print(f"  {len(rs)} venues\n")
    print(f"  {'venue':<17}{'product':<10}{'api':<10}{'payout':<9}"
          f"{'arb?':<6}{'feeds':<12}{'group'}")
    for r in rs:
        grp = r["operator"] or r["engine"] or ""
        print(f"  {r['name']:<17}{r['product']:<10}{r['api']:<10}"
              f"{r['payout']:<9}{('VOID' if r['arb_clause'] else '-'):<6}"
              f"{(r['feeds'] or '-'):<12}{grp}"
              + ("  [ACMA]" if r["acma_blocked"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
