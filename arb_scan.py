#!/usr/bin/env python3
"""
arb_scan.py -- cross-book arbitrage scanner, built to measure first.

WHAT IT DOES. Polls odds for a set of events across bookmakers, finds
outcome sets where the implied probabilities sum to under 1, and LOGS
every opportunity with timestamps so the three numbers that decide
executability can be measured before a dollar moves:

    PERSISTENCE    how long does an arb survive? If it is 4 seconds, a
                   solo bot loses to people who solved that years ago.
                   If obscure markets hold for minutes, it is tradeable.
    ACCEPTED SIZE  displayed odds are not liquidity. An arb paying $40 on
                   a $2,000 stake with a $50 limit is not an arb. The
                   scanner cannot measure this (it needs a real account),
                   so it records what it THINKS the stake should be and
                   leaves a column for what was actually accepted.
    FALSE POSITIVE rate: stale feeds, mismatched markets, same-operator
                   brands, and exchange odds quoted before commission.
                   Every one of those manufactures an arb that is not
                   there, and they are handled explicitly below.

THE SIX WAYS A NAIVE SCANNER LIES, all of which this one guards against:

  1. MISMATCHED MARKETS. You can only arb outcomes that partition the
     same event. 1X2 against draw-no-bet is not an arb, it is an
     uncovered draw. Market keys must match exactly and the outcome set
     must be complete -- a 3-way market with 2 legs is rejected.

  2. STALE QUOTES. Each book's price carries a last_update. An arb built
     from a quote 10 minutes old is an arb against the past. Legs older
     than --max-stale seconds are rejected and the staleness of the
     surviving legs is recorded, because an arb whose legs are 20 and 90
     seconds old is worse than one whose legs are both 5.

  3. SAME-OPERATOR BRANDS. Ladbrokes and Neds are both Entain. Arbing
     two brands of one operator is one account, they net you off
     internally, and they limit both at once. OPERATOR_GROUP collapses
     them so the scanner cannot pair a book with itself.

  4. EXCHANGE COMMISSION. Betfair is an exchange: winnings are charged
     commission, so face odds of 2.10 at 5% are really 2.045. Quoting an
     exchange at face value manufactures arbs out of the commission.
     Every venue carries its own commission and it is applied BEFORE the
     margin test, not after.

  5. STAKE ROUNDING. Real stakes are rounded to the dollar, which breaks
     the equal-payout property the arb relies on. The profit reported is
     the WORST-CASE outcome after rounding, never the idealised one.

  6. THE MARGIN SIGN. sum(1/odds) < 1 is an arb; > 1 is the bookmaker's
     margin. Getting that backwards finds arbs everywhere, which is how
     you discover a 1,200-arb-per-hour "edge" that is just the vig.

WHAT IT DELIBERATELY DOES NOT DO. It does not place bets. Execution is a
separate problem and should not be written until persistence and accepted
size are known numbers rather than assumptions.

REQUEST BUDGET, and this shapes the whole design. The Odds API free tier
is a few hundred requests a MONTH. Polling everything every 30 seconds
burns it before lunch. To measure PERSISTENCE you need high frequency on
FEW markets, not low frequency on many -- a 5-second cadence on three
events for an hour tells you how long arbs live; a daily sweep of every
league tells you nothing about it. So --study runs short high-cadence
bursts and the budget is tracked and enforced rather than hoped about.

USAGE
    python3 -u arb_scan.py --selftest            # no key, no network
    export ODDS_API_KEY=...                      # the-odds-api.com
    python3 -u arb_scan.py --sports              # what is available
    python3 -u arb_scan.py --once --sport aussierules_afl
    python3 -u arb_scan.py --study --sport aussierules_afl --minutes 60
    python3 -u arb_scan.py --report              # persistence from the log
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = "https://api.the-odds-api.com/v4"

# Commission on NET WINNINGS per venue. Bookmakers charge none; exchanges
# do, and quoting an exchange at face value invents arbs out of thin air.
# Betfair AU base rate is ~5-6% depending on market and discount; 6% is
# the conservative choice because understating commission creates phantom
# arbs and overstating it only hides marginal real ones.
COMMISSION = {
    "betfair_ex_au": 0.06,
    "betfair": 0.06,
    "betfair_ex_uk": 0.05,
    "matchbook": 0.015,
    "smarkets": 0.02,
}

# Brands that share a book. Arbing two of these is arbing yourself: one
# operator, one risk desk, one limiting decision, and they net the
# positions internally.
OPERATOR_GROUP = {
    "ladbrokes_au": "entain",
    "neds": "entain",
    "sportsbet": "flutter",
    "betfair_ex_au": "flutter_exchange",   # separate book from sportsbet
    "tab": "tabcorp",
    "tabtouch": "tabcorp",
    "unibet": "kindred",
    "betstar": "entain",
    "bookmaker": "entain",
}

DB_DEFAULT = os.path.expanduser("~/arb_log.sqlite")


def log(m=""):
    print(m, flush=True)


def now_utc():
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------- maths
def effective_odds(odds: float, commission: float) -> float:
    """Decimal odds after commission on net winnings.

    A $1 stake at decimal odds d returns d, of which (d-1) is winnings.
    An exchange takes c of the winnings, so the realised return is
    1 + (d-1)*(1-c). At d=2.10 and c=5% that is 2.045, and treating it as
    2.10 is how an exchange's commission becomes a fake 2.7% arb.
    """
    if odds <= 1.0 or commission <= 0:
        return odds
    return 1.0 + (odds - 1.0) * (1.0 - commission)


def implied_margin(odds_list) -> float:
    """sum(1/odds). BELOW 1 is an arbitrage; ABOVE 1 is the vig."""
    if not odds_list or any(o <= 1.0 for o in odds_list):
        return math.inf
    return sum(1.0 / o for o in odds_list)


def arb_stakes(odds_list, total: float):
    """Stakes that equalise payout across every outcome.

    stake_i = total * (1/odds_i) / margin, so stake_i * odds_i is the same
    for every i. Returns (stakes, payout, profit_fraction) on the ideal
    unrounded split.
    """
    m = implied_margin(odds_list)
    if not math.isfinite(m) or m <= 0:
        return None, 0.0, 0.0
    stakes = [total * (1.0 / o) / m for o in odds_list]
    payout = total / m
    return stakes, payout, (1.0 / m) - 1.0


def rounded_outcome(odds_list, total: float, step: float = 1.0):
    """Worst-case profit after rounding stakes to a real betting increment.

    The ideal split equalises payout. Rounding does not, so the realised
    profit is the WORST outcome, not the average and certainly not the
    best. On thin arbs this is the difference between a profit and a loss:
    a 0.8% arb on $200 with $1 rounding can round to negative.
    """
    stakes, _, ideal = arb_stakes(odds_list, total)
    if stakes is None:
        return None
    rs = [max(step, round(s / step) * step) for s in stakes]
    spent = sum(rs)
    payouts = [r * o for r, o in zip(rs, odds_list)]
    worst = min(payouts) - spent
    return {
        "stakes": rs,
        "spent": spent,
        "worst_payout": min(payouts),
        "best_payout": max(payouts),
        "worst_profit": worst,
        "worst_profit_pct": (worst / spent * 100.0) if spent > 0 else 0.0,
        "ideal_profit_pct": ideal * 100.0,
    }


# ------------------------------------------------------------ arb finding
def best_per_outcome(bookmakers, market_key, max_stale_s, asof,
                     expected_outcomes=None):
    """Best COMMISSION-ADJUSTED price for each outcome, one per operator.

    Returns {outcome_name: (eff_odds, raw_odds, book_key, stale_seconds)}
    keeping, for each outcome, the best price available from any single
    operator group. Collapsing by operator group here is what stops the
    scanner pairing Ladbrokes against Neds and calling it an arb.
    """
    best = {}
    for bk in bookmakers:
        key = bk.get("key", "")
        grp = OPERATOR_GROUP.get(key, key)
        comm = COMMISSION.get(key, 0.0)
        for mk in bk.get("markets", []):
            if mk.get("key") != market_key:
                continue
            # A BOOK QUOTING A DIFFERENT NUMBER OF OUTCOMES IS QUOTING A
            # DIFFERENT CONTRACT. In a 3-way market (1X2), a book showing
            # two outcomes is offering draw-no-bet or double-chance: its
            # "Home" is not the same bet as the 3-way "Home", because the
            # draw is refunded or included. Cherry-picking that price as
            # the Home leg of a 3-way arb leaves the draw uncovered while
            # the margin maths says you are safe.
            #
            # My own selftest for this case did not catch it, because the
            # fixture I wrote still covered all three outcomes between the
            # two books, so the completeness check passed and the scanner
            # happily reported a 112% edge. The test name was right and the
            # test was wrong -- a reminder that a passing suite only proves
            # the cases you actually built.
            outs = mk.get("outcomes", []) or []
            if expected_outcomes and len(outs) != expected_outcomes:
                continue
            upd = mk.get("last_update") or bk.get("last_update")
            stale = _stale_seconds(upd, asof)
            if stale is None or stale > max_stale_s:
                continue
            for oc in mk.get("outcomes", []):
                name = oc.get("name")
                raw = oc.get("price")
                if not name or not isinstance(raw, (int, float)) or raw <= 1:
                    continue
                eff = effective_odds(float(raw), comm)
                cur = best.get(name)
                if cur is None or eff > cur[0]:
                    best[name] = (eff, float(raw), key, stale, grp)
    return best


def _stale_seconds(iso, asof):
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except Exception:
        return None
    return max(0.0, (asof - t).total_seconds())


def find_arb(event, market_key="h2h", max_stale_s=120.0, total=1000.0,
             asof=None):
    """One event, one market. Returns an opportunity dict or None.

    Rejects incomplete outcome sets: a 3-way market quoted on 2 legs is
    not an arb with a smaller margin, it is an uncovered third outcome.
    The expected outcome count is taken from the widest book rather than
    assumed, because 2-way and 3-way markets both exist.
    """
    asof = asof or now_utc()
    bks = event.get("bookmakers", []) or []
    expected = 0
    for bk in bks:
        for mk in bk.get("markets", []):
            if mk.get("key") == market_key:
                expected = max(expected, len(mk.get("outcomes", []) or []))
    if expected < 2:
        return None

    best = best_per_outcome(bks, market_key, max_stale_s, asof,
                            expected_outcomes=expected)
    if len(best) != expected:
        return None            # incomplete partition -- NOT an arb

    # all legs from one operator group is not an arb either
    groups = {v[4] for v in best.values()}
    if len(groups) < 2:
        return None

    names = list(best)
    eff = [best[n][0] for n in names]
    margin = implied_margin(eff)
    if not math.isfinite(margin) or margin >= 1.0:
        return None

    r = rounded_outcome(eff, total)
    if r is None or r["worst_profit"] <= 0:
        return None            # rounding ate it

    return {
        "ts": asof.isoformat(),
        "event_id": event.get("id"),
        "sport": event.get("sport_key"),
        "commence": event.get("commence_time"),
        "home": event.get("home_team"),
        "away": event.get("away_team"),
        "market": market_key,
        "margin": margin,
        "edge_pct": (1.0 / margin - 1.0) * 100.0,
        "legs": [
            {"outcome": n, "book": best[n][2], "raw_odds": best[n][1],
             "eff_odds": best[n][0], "stale_s": best[n][3],
             "stake": r["stakes"][i]}
            for i, n in enumerate(names)
        ],
        "max_stale_s": max(best[n][3] for n in names),
        "worst_profit": r["worst_profit"],
        "worst_profit_pct": r["worst_profit_pct"],
        "ideal_profit_pct": r["ideal_profit_pct"],
        "spent": r["spent"],
        "key": _opp_key(event, market_key, best, names),
    }


def _opp_key(event, market_key, best, names):
    """Stable identity for ONE opportunity across scans, so persistence is
    measured on the same thing. Keyed on the event, market, and which book
    holds which outcome -- if the book set changes it is a different
    opportunity, not the same one continuing."""
    pairs = ",".join(f"{n}@{best[n][2]}" for n in sorted(names))
    return f"{event.get('id')}|{market_key}|{pairs}"


# ------------------------------------------------------------------- http
class Budget:
    """Hard request budget. The free tier is a few hundred calls a MONTH;
    a 30-second poll loop spends it before lunch. Tracked in the DB so it
    survives restarts, and enforced rather than trusted."""

    def __init__(self, con, limit):
        self.con, self.limit = con, limit

    def used(self):
        row = self.con.execute(
            "SELECT COUNT(*) FROM requests WHERE ts > datetime('now',"
            " '-30 days')").fetchone()
        return int(row[0]) if row else 0

    def spend(self, n=1, note=""):
        if self.limit and self.used() + n > self.limit:
            raise RuntimeError(
                f"request budget exhausted: {self.used()}/{self.limit} in the"
                f" last 30 days. Raise --budget deliberately or wait.")
        for _ in range(n):
            self.con.execute("INSERT INTO requests(ts,note) VALUES"
                             " (datetime('now'),?)", (note,))
        self.con.commit()


def http_get(path, params, budget=None, note="", retries=3):
    """GET with backoff. Returns (payload, headers). Never swallows the
    error silently -- a scanner that treats a failed fetch as 'no arbs'
    reports a quiet market instead of a broken one."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set. Get a free key at "
                           "the-odds-api.com and export it.")
    p = dict(params)
    p["apiKey"] = key
    url = f"{API}{path}?{urllib.parse.urlencode(p)}"
    last = None
    for attempt in range(retries):
        try:
            if budget:
                budget.spend(1, note)
            req = urllib.request.Request(url, headers={
                "User-Agent": "arb-scan/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                hdr = {k.lower(): v for k, v in r.headers.items()}
                return json.loads(r.read().decode()), hdr
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            if e.code in (401, 422):
                raise RuntimeError(f"HTTP {e.code} -- {body}") from None
            last = f"HTTP {e.code}: {body}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after {retries} tries: {last}")


# --------------------------------------------------------------- storage
SCHEMA = """
CREATE TABLE IF NOT EXISTS requests(ts TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS sightings(
  ts TEXT, key TEXT, event_id TEXT, sport TEXT, market TEXT,
  home TEXT, away TEXT, commence TEXT,
  edge_pct REAL, worst_profit_pct REAL, max_stale_s REAL,
  spent REAL, legs TEXT);
CREATE INDEX IF NOT EXISTS i_key ON sightings(key);
CREATE INDEX IF NOT EXISTS i_ts ON sightings(ts);
"""


def open_db(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.commit()
    return con


def record(con, opp):
    con.execute(
        "INSERT INTO sightings(ts,key,event_id,sport,market,home,away,"
        "commence,edge_pct,worst_profit_pct,max_stale_s,spent,legs)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (opp["ts"], opp["key"], opp["event_id"], opp["sport"], opp["market"],
         opp["home"], opp["away"], opp["commence"], opp["edge_pct"],
         opp["worst_profit_pct"], opp["max_stale_s"], opp["spent"],
         json.dumps(opp["legs"])))
    con.commit()


def persistence_report(con):
    rows = con.execute(
        "SELECT key, COUNT(*) n, MIN(ts) first, MAX(ts) last,"
        " AVG(edge_pct) e, AVG(worst_profit_pct) w, AVG(max_stale_s) s,"
        " MIN(home), MIN(away), MIN(sport)"
        " FROM sightings GROUP BY key ORDER BY first").fetchall()
    if not rows:
        log("  no sightings logged yet. Run --study first.")
        return
    lives = []
    for r in rows:
        try:
            a = datetime.fromisoformat(r[2])
            b = datetime.fromisoformat(r[3])
            lives.append((b - a).total_seconds())
        except Exception:
            lives.append(0.0)
    lives.sort()
    n = len(lives)
    log("")
    log("  " + "=" * 68)
    log("  PERSISTENCE -- the number that decides whether this is tradeable")
    log("  " + "=" * 68)
    log(f"    distinct opportunities      {n}")
    log(f"    seen once only              "
        f"{sum(1 for r in rows if r[1] == 1)}"
        f"   <- lifetime shorter than one poll interval")
    for q, lbl in ((0.50, "median"), (0.75, "75th pct"), (0.90, "90th pct"),
                   (1.00, "longest")):
        i = min(n - 1, int(q * (n - 1)))
        log(f"    {lbl:<12} lifetime      {lives[i]:>8.0f}s")
    log("")
    log(f"    mean edge                   "
        f"{sum(r[4] for r in rows)/n:>7.2f}%")
    log(f"    mean worst-case after round "
        f"{sum(r[5] for r in rows)/n:>7.2f}%")
    log(f"    mean worst leg staleness    "
        f"{sum(r[6] for r in rows)/n:>7.1f}s")
    log("")
    log("    READ IT THIS WAY. 'Seen once only' is the important row: those")
    log("    opportunities did not survive a single poll interval, so their")
    log("    true lifetime is unknown and shorter than your cadence. If")
    log("    most are in that bucket, poll faster before concluding")
    log("    anything -- the measurement is cadence-limited, not the market.")
    log("    And every lifetime here is an UPPER bound on the time you")
    log("    actually have: the arb dies when either leg moves, which can")
    log("    happen between your scan and your second bet landing.")


# -------------------------------------------------------------- selftest
def selftest() -> int:
    ok = True

    def chk(name, got, want, tol=1e-9):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        log(f"    [{'ok' if good else 'FAIL'}] {name}: "
            f"got {got:.6f} want {want:.6f}")

    log("  arb maths, hand-computed expectations:")
    # two-way 2.10 / 2.10 -> margin 0.952381, edge 5%
    chk("margin(2.10,2.10)", implied_margin([2.10, 2.10]), 0.9523809524, 1e-9)
    s, p, e = arb_stakes([2.10, 2.10], 1000.0)
    chk("stake split equal", s[0], 500.0, 1e-6)
    chk("payout", p, 1050.0, 1e-6)
    chk("edge", e * 100, 5.0, 1e-6)
    # three-way 3.0/3.5/4.0
    chk("margin(3,3.5,4)", implied_margin([3.0, 3.5, 4.0]), 0.8690476190, 1e-9)
    # vig market must NOT be an arb
    m = implied_margin([1.90, 1.90])
    log(f"    [{'ok' if m > 1 else 'FAIL'}] vig market margin {m:.4f} > 1"
        f" (not an arb)")
    ok = ok and m > 1
    # commission: 2.10 at 5% -> 2.045
    chk("effective_odds(2.10, 5%)", effective_odds(2.10, 0.05), 2.045, 1e-9)
    chk("effective_odds(2.10, 0%)", effective_odds(2.10, 0.0), 2.10, 1e-12)

    log("")
    log("  commission turns a fake arb back into the vig:")
    fake = implied_margin([2.02, 2.02])
    real = implied_margin([effective_odds(2.02, 0.06), 2.02])
    log(f"    face 2.02/2.02  margin {fake:.4f}  -> 'arb' "
        f"{'YES' if fake < 1 else 'no'}")
    log(f"    6% on one leg   margin {real:.4f}  -> arb "
        f"{'YES' if real < 1 else 'no'}")

    log("")
    log("  rounding can eat a thin arb:")
    r = rounded_outcome([1.30, 5.00], 50.0)
    log(f"    1.30 / 5.00 on $50: stakes {r['stakes']}")
    log(f"    ideal {r['ideal_profit_pct']:+.3f}%  ->  worst case after $1"
        f" rounding {r['worst_profit_pct']:+.3f}%")

    log("")
    log("  find_arb rejections (each must return None):")
    asof = datetime(2026, 10, 24, 0, 0, 0, tzinfo=timezone.utc)
    fresh = "2026-10-24T00:00:00Z"
    old = "2026-10-23T00:00:00Z"

    def ev(bks, **kw):
        d = {"id": "e1", "sport_key": "s", "home_team": "H",
             "away_team": "A", "commence_time": fresh, "bookmakers": bks}
        d.update(kw)
        return d

    def bk(key, outs, upd=fresh):
        return {"key": key, "last_update": upd, "markets": [
            {"key": "h2h", "last_update": upd, "outcomes": outs}]}

    cases = [
        ("same operator group (ladbrokes vs neds)",
         ev([bk("ladbrokes_au", [{"name": "H", "price": 2.10},
                                 {"name": "A", "price": 1.50}]),
             bk("neds", [{"name": "H", "price": 1.50},
                         {"name": "A", "price": 2.10}])])),
        ("stale leg beyond max_stale",
         ev([bk("sportsbet", [{"name": "H", "price": 2.10},
                              {"name": "A", "price": 1.50}]),
             bk("tab", [{"name": "H", "price": 1.50},
                        {"name": "A", "price": 2.10}], upd=old)])),
        ("incomplete 3-way partition",
         ev([bk("sportsbet", [{"name": "H", "price": 3.0},
                              {"name": "A", "price": 3.5},
                              {"name": "Draw", "price": 4.0}]),
             bk("tab", [{"name": "H", "price": 9.0},
                        {"name": "A", "price": 9.0}])])),
        ("plain vig market, no arb",
         ev([bk("sportsbet", [{"name": "H", "price": 1.90},
                              {"name": "A", "price": 1.90}]),
             bk("tab", [{"name": "H", "price": 1.88},
                        {"name": "A", "price": 1.88}])])),
    ]
    for name, e in cases:
        got = find_arb(e, asof=asof)
        good = got is None
        ok = ok and good
        log(f"    [{'ok' if good else 'FAIL'}] rejected: {name}")

    log("")
    log("  find_arb acceptance (a real cross-operator arb):")
    e = ev([bk("sportsbet", [{"name": "H", "price": 2.20},
                             {"name": "A", "price": 1.60}]),
            bk("tab", [{"name": "H", "price": 1.60},
                       {"name": "A", "price": 2.20}])])
    got = find_arb(e, asof=asof)
    good = got is not None and got["edge_pct"] > 0
    ok = ok and good
    if got:
        log(f"    [{'ok' if good else 'FAIL'}] edge {got['edge_pct']:.3f}%"
            f"  worst-case {got['worst_profit_pct']:.3f}%"
            f"  books {[l['book'] for l in got['legs']]}")
    else:
        log("    [FAIL] a genuine arb was rejected")

    log("")
    log(f"  {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sports", action="store_true", help="list sport keys")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--study", action="store_true",
                    help="high-cadence burst, to measure persistence")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--sport", default="aussierules_afl")
    ap.add_argument("--regions", default="au")
    ap.add_argument("--market", default="h2h")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="seconds between polls in --study")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--total", type=float, default=1000.0)
    ap.add_argument("--max-stale", type=float, default=120.0)
    ap.add_argument("--budget", type=int, default=450,
                    help="max API requests per 30 days (free tier ~500)")
    ap.add_argument("--db", default=DB_DEFAULT)
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    con = open_db(a.db)
    if a.report:
        persistence_report(con)
        return 0

    budget = Budget(con, a.budget)

    if a.sports:
        data, hdr = http_get("/sports", {}, budget, "sports")
        log(f"  {len(data)} sports; remaining per API header: "
            f"{hdr.get('x-requests-remaining', '?')}")
        for s in data:
            if s.get("active"):
                log(f"    {s['key']:<34}{s.get('title','')}")
        return 0

    def scan():
        data, hdr = http_get(
            f"/sports/{a.sport}/odds",
            {"regions": a.regions, "markets": a.market,
             "oddsFormat": "decimal"}, budget, f"odds:{a.sport}")
        asof = now_utc()
        found = []
        for ev_ in data:
            o = find_arb(ev_, a.market, a.max_stale, a.total, asof)
            if o:
                found.append(o)
                record(con, o)
        return data, found, hdr

    if a.once:
        data, found, hdr = scan()
        log(f"  {len(data)} events, {len(found)} arbs, "
            f"{budget.used()}/{a.budget} requests used, "
            f"API remaining {hdr.get('x-requests-remaining','?')}")
        for o in found:
            log(f"    {o['home']} v {o['away']}  edge {o['edge_pct']:+.2f}%"
                f"  worst-case {o['worst_profit_pct']:+.2f}%"
                f"  stale<={o['max_stale_s']:.0f}s")
            for l in o["legs"]:
                log(f"      {l['outcome']:<22}{l['book']:<16}"
                    f"@{l['raw_odds']:>6.2f} (eff {l['eff_odds']:.3f})"
                    f"  stake {l['stake']:>8.2f}")
        if not found:
            log("    none. That is the expected result most of the time and")
            log("    is not a bug -- log enough scans before concluding.")
        return 0

    if a.study:
        polls = int(a.minutes * 60 / a.interval)
        cost = polls
        log(f"  persistence study: {polls} polls at {a.interval:.0f}s"
            f" over {a.minutes:.0f} min")
        log(f"  cost {cost} requests; {budget.used()}/{a.budget} already"
            f" used in the last 30 days")
        if budget.limit and budget.used() + cost > budget.limit:
            log(f"  THAT EXCEEDS THE BUDGET. Reduce --minutes, raise")
            log(f"  --interval, or raise --budget deliberately.")
            return 2
        seen_keys = set()
        for i in range(polls):
            try:
                _, found, _ = scan()
            except RuntimeError as e:
                log(f"  poll {i+1}: {e}")
                break
            for o in found:
                seen_keys.add(o["key"])
            log(f"    poll {i+1}/{polls}  live arbs {len(found)}"
                f"  distinct so far {len(seen_keys)}")
            if i < polls - 1:
                time.sleep(a.interval)
        persistence_report(con)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
