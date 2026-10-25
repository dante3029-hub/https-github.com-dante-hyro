#!/usr/bin/env python3
"""
arb.py -- the whole pipeline: odds in, filtered opportunities out.

Ties together the pieces that already exist:

    venues.independent_sources   collapse brands sharing an operator or
                                 an odds engine into ONE price source
    arb_scan.find_arb            margin, commission, staleness, rounding
    consensus.outlier_report     is the edge distributed, or is it one
                                 book being wrong?
    consensus.void_economics     if the outlier leg is voided, what
                                 honour rate makes the bet break even

ORDER MATTERS. Sources are collapsed BEFORE the margin is computed, not
after. Collapsing afterwards means the scanner has already found an
"arb" between Melbet and 1xBet and you are deciding whether to believe
a number that was never real.

THE CANONICAL SCHEMA is The Odds API's, because it is the simplest of
the three and every other provider maps onto it cleanly:

    {id, sport_key, commence_time, home_team, away_team,
     bookmakers: [{key, title, last_update,
                   markets: [{key, last_update,
                              outcomes: [{name, price, point}]}]}]}

Adapters convert into that. Nothing downstream knows which provider the
data came from.

THE ODDSPAPI ADAPTER IS WRITTEN AGAINST THEIR DOCUMENTED SHAPE AND IS
NOT VERIFIED. I cannot reach their API from here, so it accepts several
plausible field spellings, validates what it got, and on a mismatch
prints the keys it actually received instead of silently producing an
empty list. Run --sample first: it dumps one raw event so the mapping
can be corrected against reality rather than guessed at twice.

USAGE
    python3 -u arb.py --selftest                 # whole chain, offline
    python3 -u arb.py --sample --source oddspapi # dump one raw event
    python3 -u arb.py --source oddspapi --sport soccer
    python3 -u arb.py --source file --path saved.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import arb_scan as AS
import consensus as CS
import venues as VN

MIN_EDGE = 0.3          # %, below this the admin is not worth it
MAX_OUTLIER_PP = 8.0    # legs further than this from consensus are flagged


def log(m=""):
    print(m, flush=True)


# ------------------------------------------------------------- adapters
def from_oddspapi(raw):
    """Map an OddsPapi payload into the canonical schema.

    UNVERIFIED -- see the module docstring. Field names are accepted in
    several spellings because providers differ on camelCase vs snake,
    'price' vs 'odds', and 'bookmaker' vs 'book'.
    """
    def g(d, *names, default=None):
        for n in names:
            if isinstance(d, dict) and n in d and d[n] is not None:
                return d[n]
        return default

    events = raw if isinstance(raw, list) else g(raw, "data", "events",
                                                 "results", default=[])
    out = []
    for e in events or []:
        bks = []
        for b in g(e, "bookmakers", "books", "sportsbooks", default=[]) or []:
            mks = []
            for m in g(b, "markets", "odds", default=[]) or []:
                ocs = []
                for o in g(m, "outcomes", "selections", default=[]) or []:
                    price = g(o, "price", "odds", "decimal_odds", "decimal")
                    if price is None:
                        continue
                    ocs.append({"name": g(o, "name", "label", "selection",
                                          "outcome"),
                                "price": float(price),
                                "point": g(o, "point", "line", "handicap")})
                if ocs:
                    mks.append({"key": g(m, "key", "market", "market_key",
                                         default="h2h"),
                                "last_update": g(m, "last_update",
                                                 "updated_at", "timestamp"),
                                "outcomes": ocs})
            if mks:
                bks.append({"key": str(g(b, "key", "bookmaker", "book",
                                         "name", default="")).lower(),
                            "title": g(b, "title", "name", default=""),
                            "last_update": g(b, "last_update", "updated_at",
                                             "timestamp"),
                            "markets": mks})
        if bks:
            out.append({"id": g(e, "id", "event_id", "uuid"),
                        "sport_key": g(e, "sport_key", "sport", "league"),
                        "commence_time": g(e, "commence_time", "start_time",
                                           "starts", "commence"),
                        "home_team": g(e, "home_team", "home", "team_home"),
                        "away_team": g(e, "away_team", "away", "team_away"),
                        "bookmakers": bks})
    if events and not out:
        keys = sorted(events[0].keys()) if isinstance(events[0], dict) else []
        raise ValueError(
            "OddsPapi payload did not map to any event. Top-level keys of "
            f"the first item: {keys}. Run --sample and send me that output; "
            "the adapter is written against their docs, not their wire "
            "format.")
    return out


def from_theoddsapi(raw):
    """Already canonical."""
    return raw if isinstance(raw, list) else []


ADAPTERS = {"oddspapi": from_oddspapi, "theoddsapi": from_theoddsapi,
            "file": from_theoddsapi}


# -------------------------------------------------------------- pipeline
def collapse_sources(event):
    """Keep only the BEST price per outcome per independent source.

    Done before any margin is computed. Two brands sharing an operator or
    an odds engine become one book, holding whichever of them quoted the
    better price -- which is also what you would actually bet.
    """
    bks = event.get("bookmakers", []) or []
    groups = VN.independent_sources([b.get("key", "") for b in bks])
    member_to_group = {}
    for gkey, members in groups.items():
        for m in members:
            member_to_group[str(m).lower()] = gkey

    best = {}
    for b in bks:
        g = member_to_group.get(str(b.get("key", "")).lower(),
                                f"solo:{b.get('key','')}")
        cur = best.get(g)
        if cur is None:
            best[g] = dict(b, key=g, title=b.get("title") or b.get("key"))
            continue
        # merge: per market/outcome keep the higher price
        for m in b.get("markets", []):
            tgt = next((x for x in cur["markets"]
                        if x.get("key") == m.get("key")), None)
            if tgt is None:
                cur["markets"].append(m)
                continue
            for o in m.get("outcomes", []):
                t = next((x for x in tgt["outcomes"]
                          if x.get("name") == o.get("name")), None)
                if t is None:
                    tgt["outcomes"].append(o)
                elif (o.get("price") or 0) > (t.get("price") or 0):
                    t["price"] = o["price"]
    out = dict(event)
    out["bookmakers"] = list(best.values())
    return out, len(bks), len(best)


def ladder_for(event, market_key, outcome_name):
    """Every book's price for one outcome -- the consensus input."""
    lad, labs = [], []
    for b in event.get("bookmakers", []) or []:
        for m in b.get("markets", []) or []:
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []) or []:
                if o.get("name") == outcome_name and o.get("price"):
                    lad.append(float(o["price"]))
                    labs.append(b.get("key", "?"))
    return lad, labs


def assess(raw_event, market_key="h2h", total=1000.0, asof=None):
    """One event -> an opportunity with its consensus verdict, or None."""
    asof = asof or datetime.now(timezone.utc)
    collapsed, n_books, n_sources = collapse_sources(raw_event)
    if n_sources < 2:
        return None
    opp = AS.find_arb(collapsed, market_key, 600.0, total, asof)
    if opp is None or opp["edge_pct"] < MIN_EDGE:
        return None

    worst_dev, worst_leg = 0.0, None
    for leg in opp["legs"]:
        lad, _ = ladder_for(raw_event, market_key, leg["outcome"])
        if len(lad) < 3:
            continue
        cons, _, _ = CS.outlier_report(lad)
        dev = (CS.implied(leg["raw_odds"]) - cons) * 100.0
        leg["consensus_odds"] = 1.0 / cons if cons else None
        leg["dev_pp"] = dev
        if abs(dev) > abs(worst_dev):
            worst_dev, worst_leg = dev, leg

    opp["n_books"] = n_books
    opp["n_sources"] = n_sources
    opp["worst_dev_pp"] = worst_dev
    opp["rests_on_outlier"] = abs(worst_dev) >= MAX_OUTLIER_PP
    if worst_leg is not None and len(opp["legs"]) == 2:
        other = [l for l in opp["legs"] if l is not worst_leg][0]
        cons_p = CS.implied(worst_leg["raw_odds"]) - worst_dev / 100.0
        opp["void"] = CS.void_economics(
            worst_leg["raw_odds"], worst_leg["stake"],
            other["raw_odds"], other["stake"], max(min(cons_p, 0.999), 0.001))
    return opp


def report(opps):
    if not opps:
        log("  no opportunities cleared the filters.")
        log("  That is the normal result. It is not a bug.")
        return
    clean = [o for o in opps if not o["rests_on_outlier"]]
    sus = [o for o in opps if o["rests_on_outlier"]]
    log(f"  {len(opps)} opportunities: {len(clean)} distributed,"
        f" {len(sus)} resting on an outlier\n")
    for tag, group in (("REAL -- edge is spread across books", clean),
                       ("SUSPECT -- one book is the whole edge", sus)):
        if not group:
            continue
        log(f"  {tag}")
        for o in sorted(group, key=lambda x: -x["edge_pct"]):
            log(f"    {o['edge_pct']:>6.2f}%  {o['home']} v {o['away']}"
                f"   {o['n_sources']} sources (of {o['n_books']} brands)"
                f"   worst leg {o['worst_dev_pp']:+.1f}pp")
            for l in o["legs"]:
                c = l.get("consensus_odds")
                log(f"        {l['outcome']:<18}{l['book']:<16}"
                    f"@{l['raw_odds']:>7.3f}"
                    + (f"   consensus {c:>6.3f}" if c else "")
                    + (f"   {l['dev_pp']:+.1f}pp" if "dev_pp" in l else ""))
            if o.get("void") and o["rests_on_outlier"]:
                b = o["void"]["breakeven_honour"]
                if b == b:
                    log(f"        -> +EV only if that book honours it"
                        f" >{b*100:.0f}% of the time")
        log("")


# -------------------------------------------------------------- selftest
def _bk(key, outs, upd="2026-10-25T00:00:00Z"):
    return {"key": key, "title": key, "last_update": upd,
            "markets": [{"key": "h2h", "last_update": upd,
                         "outcomes": outs}]}


def _ev(bks, eid="e1"):
    return {"id": eid, "sport_key": "test",
            "commence_time": "2026-10-25T12:00:00Z",
            "home_team": "Alpha", "away_team": "Beta", "bookmakers": bks}


def selftest() -> int:
    ok = True
    asof = datetime(2026, 10, 25, 0, 0, 0, tzinfo=timezone.utc)

    def chk(name, cond):
        nonlocal ok
        ok = ok and cond
        log(f"    [{'ok' if cond else 'FAIL'}] {name}")

    log("  1. a GENUINE arb, edge spread across independent books:")
    e = _ev([_bk("cloudbet", [{"name": "Alpha", "price": 2.10},
                              {"name": "Beta", "price": 1.95}]),
             _bk("betfair",  [{"name": "Alpha", "price": 1.98},
                              {"name": "Beta", "price": 2.08}]),
             _bk("bovada",   [{"name": "Alpha", "price": 2.02},
                              {"name": "Beta", "price": 1.99}]),
             _bk("mybookie", [{"name": "Alpha", "price": 2.05},
                              {"name": "Beta", "price": 1.97}])])
    o = assess(e, asof=asof)
    chk("found", o is not None)
    if o:
        chk(f"not flagged as outlier-driven (worst leg"
            f" {o['worst_dev_pp']:+.1f}pp)", not o["rests_on_outlier"])

    log("")
    log("  2. a FAKE arb: one book 19 points off the market (the PB case):")
    outs_norm = [{"name": "Alpha", "price": 1.08}, {"name": "Beta", "price": 9.0}]
    # pb must NOT also hold the best price on the other outcome, or
    # find_arb correctly refuses it as a single-book "arb" and this test
    # proves nothing. My first fixture gave pb 9.0 on Beta as well.
    e = _ev([_bk("pb",       [{"name": "Alpha", "price": 1.357},
                              {"name": "Beta", "price": 8.0}]),
             _bk("bovada",   [{"name": "Alpha", "price": 1.06},
                              {"name": "Beta", "price": 8.5}]),
             _bk("betfair",  outs_norm),
             _bk("mybookie", [{"name": "Alpha", "price": 1.07},
                              {"name": "Beta", "price": 8.8}]),
             _bk("betonline", outs_norm)])
    o = assess(e, asof=asof)
    chk("found as an arb", o is not None)
    if o:
        chk(f"FLAGGED as outlier-driven ({o['worst_dev_pp']:+.1f}pp)",
            o["rests_on_outlier"])
        b = o.get("void", {}).get("breakeven_honour")
        chk("void economics computed", b is not None and b == b)

    log("")
    log("  3. a PHANTOM arb between two brands of one price engine:")
    e = _ev([_bk("melbet", [{"name": "Alpha", "price": 2.20},
                            {"name": "Beta", "price": 1.60}]),
             _bk("1xbet",  [{"name": "Alpha", "price": 1.60},
                            {"name": "Beta", "price": 2.20}])])
    collapsed, nb, ns = collapse_sources(e)
    chk(f"2 brands collapse to {ns} source", ns == 1)
    chk("so no arb is reported at all", assess(e, asof=asof) is None)

    log("")
    log("  4. Betby books (different operators, same odds feed):")
    e = _ev([_bk("bc.game", [{"name": "Alpha", "price": 2.20},
                             {"name": "Beta", "price": 1.60}]),
             _bk("rollbit", [{"name": "Alpha", "price": 1.60},
                             {"name": "Beta", "price": 2.20}])])
    _, _, ns = collapse_sources(e)
    chk(f"BC.Game + Rollbit collapse to {ns} source", ns == 1)

    log("")
    log("  5. adapter fails LOUDLY on an unexpected payload:")
    try:
        from_oddspapi([{"totally": "wrong", "shape": 1}])
        chk("raised", False)
    except ValueError as ex:
        chk(f"raised with the keys it saw", "totally" in str(ex))

    log("")
    log(f"  {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sample", action="store_true",
                    help="dump one raw event, to fix the adapter mapping")
    ap.add_argument("--source", default="file",
                    choices=sorted(ADAPTERS))
    ap.add_argument("--path", default="")
    ap.add_argument("--sport", default="")
    ap.add_argument("--market", default="h2h")
    ap.add_argument("--total", type=float, default=1000.0)
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.source == "file":
        if not a.path or not os.path.exists(a.path):
            log("  --source file needs --path to a saved JSON payload")
            return 2
        raw = json.load(open(a.path))
    else:
        log(f"  live fetch for '{a.source}' is not wired yet: I cannot")
        log(f"  reach it from here to verify the request shape.")
        log(f"  Save one response to a file and use --source {a.source}")
        log(f"  --path that_file.json, or send me the --sample output.")
        return 2

    events = ADAPTERS[a.source](raw)
    log(f"  {len(events)} events from {a.source}")
    if a.sample and events:
        log(json.dumps(events[0], indent=2)[:2000])
        return 0
    opps = [x for x in (assess(e, a.market, a.total) for e in events) if x]
    report(opps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
