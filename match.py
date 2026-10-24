#!/usr/bin/env python3
"""
match.py -- deciding that two venues are quoting the SAME event.

THIS IS THE MOAT, not the odds ingestion. Pulling prices from twelve
feeds is a weekend. Knowing that

    Cloudbet : "Montreal Canadiens vs Winnipeg Jets"
    SX Bet   : "MTL Canadiens @ WPG Jets"
    1xBet    : "Canadiens Montreal - Jets Winnipeg"

are one event, while

    "Manchester United vs Manchester City"
    "Manchester City vs Manchester United"

are the same event with the sides flipped, and

    "Manchester United vs Liverpool"
    "Manchester City vs Liverpool"

are NOT the same event despite sharing most of their tokens, is where
these projects die. Get it wrong in the permissive direction and the
scanner reports arbitrage between two different games.

A WRONG MATCH IS WORSE THAN A MISSED ONE. A missed match costs an
opportunity. A wrong match produces a confident, well-formatted,
completely fictional arb that passes every downstream check -- the
margin maths is correct, the commission is right, the stakes are right,
and the two bets are on different football matches. So every rule here
is biased toward refusing to match.

FOUR GUARDS, in order of how much they matter:

  1. TIME WINDOW. Two quotes are the same event only if the start times
     agree within a few minutes. This single rule removes almost all
     cross-fixture confusion, because the same two teams rarely play
     twice within minutes of each other. Books do disagree by a minute
     or two, hence a window rather than equality.

  2. MARGIN OVER RUNNER-UP. The best candidate must beat the second
     best by a clear gap. In a league full of shared tokens --
     Manchester United, Manchester City; Inter Milan, AC Milan --
     a high absolute score means little if another fixture scores
     almost as well. Ambiguity is a refusal, not a coin flip.

  3. DISCRIMINATING TOKENS. Generic club words (FC, SC, United, City,
     Real, Athletic) are stripped before comparison, so matching cannot
     rest on them. "Manchester United" and "Manchester City" reduce to
     {manchester} and {manchester} and are correctly judged ambiguous
     rather than identical -- the nickname or the abbreviation has to
     carry the match.

  4. ORIENTATION. Home and away are swapped between books constantly
     ("A vs B" and "B @ A"). Both orientations are tried and the better
     one is returned WITH a flag, because an outcome labelled "Home"
     means different things on the two sides of a flipped fixture and
     silently merging them inverts the bet.

USAGE
    python3 -u match.py --selftest
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from datetime import datetime, timedelta, timezone

# STRUCTURAL abbreviations only -- the letters books add or drop freely
# around a name ("FC Zurich" / "Zurich"). NOTHING that distinguishes one
# club from another belongs here.
#
# My first version also listed united, city, town, county, athletic,
# real, sporting, sociedad and deportivo as noise, on the reasoning that
# half a league shares them. The selftest scored "Manchester United"
# against "Manchester City" at 1.00: both reduced to {manchester} and
# became indistinguishable. I stripped the exact tokens that carry the
# distinction, in a file whose docstring warns about that case. Jaccard
# handles common-but-identifying words correctly on its own -- it is only
# the structural noise that needs removing.
NOISE = {
    "fc", "cf", "sc", "ac", "afc", "cfc", "sv", "bk", "if", "ff", "fk",
    "cd", "ud", "sd", "ca", "club", "calcio", "the",
    "de", "do", "da", "el", "la", "los", "las",
}

# Abbreviations books use instead of names. Expanded before comparison so
# "MTL" and "Montreal" reach the same token.
ABBREV = {
    "mtl": "montreal", "wpg": "winnipeg", "tor": "toronto",
    "nyr": "rangers", "nyi": "islanders", "njd": "devils",
    "tb": "tampa", "tbl": "tampa", "lak": "kings", "sjs": "sharks",
    "vgk": "vegas", "cgy": "calgary", "edm": "edmonton",
    "van": "vancouver", "col": "colorado", "min": "minnesota",
    "chi": "chicago", "det": "detroit", "bos": "boston",
    "phi": "philadelphia", "pit": "pittsburgh", "wsh": "washington",
    "car": "carolina", "fla": "florida", "nsh": "nashville",
    "stl": "louis", "dal": "dallas", "ari": "arizona", "uta": "utah",
    "ott": "ottawa", "buf": "buffalo", "cbj": "columbus",
    "man": "manchester", "utd": "united",          # "Man Utd"
    "psg": "paris", "bvb": "dortmund",
    "lafc": "losangeles", "lag": "galaxy",
}

TIME_WINDOW = timedelta(minutes=10)
MIN_SCORE = 0.60          # absolute similarity floor
MIN_MARGIN = 0.15         # how far the winner must beat the runner-up


def log(m=""):
    print(m, flush=True)


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def tokens(name):
    """Name -> the set of tokens that actually identify the team."""
    s = strip_accents(str(name)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    raw = [t for t in s.split() if t]
    out = set()
    for t in raw:
        t = ABBREV.get(t, t)
        if t in NOISE or len(t) < 2:
            continue
        out.add(t)
    # if stripping removed everything, fall back to the raw tokens rather
    # than returning an empty set that matches everything
    return out or {t for t in raw if len(t) >= 2}


def similarity(a, b):
    """Jaccard over identifying tokens, with a containment bonus.

    Containment matters because books truncate: "Montreal Canadie..." in
    a UI, "Montreal" in a feed. Pure Jaccard punishes that; containment
    of the smaller set inside the larger does not.
    """
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0          # identical identifying tokens IS the same name;
                            # the single-token guard below must not cap it
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    jac = inter / len(ta | tb)
    # Containment was weighted 0.5 and applied unconditionally, which let
    # "AC Milan" sit inside "Inter Milan" at containment 1.0 and score
    # 0.75. A one-token name is contained in half the league, so the
    # bonus only applies when the smaller name has at least two tokens --
    # where it is doing its real job, rescuing truncated feed strings
    # like "Montreal Canadie..." -- and it is weighted lightly.
    smaller = min(len(ta), len(tb))
    if smaller < 2:
        return 0.8 * jac
    return 0.8 * jac + 0.2 * (inter / smaller)


def pair_score(home_a, away_a, home_b, away_b):
    """Best of the two orientations. Returns (score, flipped)."""
    straight = (similarity(home_a, home_b) + similarity(away_a, away_b)) / 2
    flipped = (similarity(home_a, away_b) + similarity(away_a, home_b)) / 2
    return (straight, False) if straight >= flipped else (flipped, True)


def _parse_ts(x):
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def match_event(target, candidates, window=TIME_WINDOW,
                min_score=MIN_SCORE, min_margin=MIN_MARGIN):
    """Find the candidate that is the SAME event as `target`, or None.

    target/candidates: dicts with home, away, commence (ISO or datetime).
    Returns (candidate, score, flipped) or (None, best_score, reason).
    """
    t0 = _parse_ts(target.get("commence"))
    scored = []
    for c in candidates:
        t1 = _parse_ts(c.get("commence"))
        if t0 and t1 and abs(t1 - t0) > window:
            continue                       # guard 1: different kickoff
        s, flip = pair_score(target.get("home", ""), target.get("away", ""),
                             c.get("home", ""), c.get("away", ""))
        scored.append((s, flip, c))
    if not scored:
        return None, 0.0, "no candidate inside the time window"
    scored.sort(key=lambda x: -x[0])
    best_s, best_f, best_c = scored[0]
    if best_s < min_score:
        return None, best_s, f"best score {best_s:.2f} below floor"
    if len(scored) > 1:
        margin = best_s - scored[1][0]
        if margin < min_margin:            # guard 2: ambiguous
            return (None, best_s,
                    f"ambiguous: runner-up within {margin:.2f}"
                    f" ({scored[1][2].get('home')} v"
                    f" {scored[1][2].get('away')})")
    return best_c, best_s, best_f


# ------------------------------------------------------------ market keys
MARKET_ALIASES = {
    "h2h": "h2h", "moneyline": "h2h", "ml": "h2h", "match winner": "h2h",
    "1x2": "h2h_3way", "match odds": "h2h_3way", "winner 3way": "h2h_3way",
    "totals": "totals", "total": "totals", "over/under": "totals",
    "total goals": "totals", "total points": "totals",
    "total goals reg time": "totals_reg", "total rounds": "totals",
    "spreads": "spreads", "handicap": "spreads", "line": "spreads",
    "asian handicap": "spreads_asian",
}


def market_key(name):
    """Normalise a market name. Returns None when unrecognised, because
    guessing a market is the same failure as guessing an event."""
    s = re.sub(r"[^a-z0-9/\s]", " ", strip_accents(str(name)).lower())
    s = " ".join(s.split())
    return MARKET_ALIASES.get(s)


LINE_RE = re.compile(r"(over|under|o|u)\s*([0-9]+(?:\.[0-9]+)?)", re.I)


def outcome_key(name):
    """'Over 2.5' / 'O 2.5' / 'over 2.5 goals' -> ('over', 2.5).

    Returns (label, line) or (normalised_name, None). The LINE is part
    of the identity: Over 2.5 and Over 3.5 are different bets, and a
    scanner that drops the number will pair them and call it an arb.
    """
    s = strip_accents(str(name)).lower().strip()
    m = LINE_RE.match(s)
    if m:
        side = "over" if m.group(1).startswith("o") else "under"
        return side, float(m.group(2))
    return " ".join(s.split()), None


# -------------------------------------------------------------- selftest
def selftest() -> int:
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok = ok and cond
        log(f"    [{'ok' if cond else 'FAIL'}] {name}")

    T = "2026-10-25T10:00:00Z"
    log("  SAME event across venues:")
    cands = [{"home": "MTL Canadiens", "away": "WPG Jets", "commence": T}]
    c, s, f = match_event(
        {"home": "Montreal Canadiens", "away": "Winnipeg Jets",
         "commence": T}, cands)
    chk(f"Montreal/Winnipeg matches MTL/WPG (score {s:.2f})", c is not None)

    c, s, f = match_event(
        {"home": "Montreal Canadie...", "away": "Winnipeg Jets",
         "commence": T}, cands)
    chk(f"truncated UI name still matches (score {s:.2f})", c is not None)

    log("")
    log("  ORIENTATION flipped:")
    c, s, f = match_event(
        {"home": "Winnipeg Jets", "away": "Montreal Canadiens",
         "commence": T}, cands)
    chk("flipped fixture matches AND is flagged flipped",
        c is not None and f is True)

    log("")
    log("  REFUSALS (each must return None):")
    c, s, why = match_event(
        {"home": "Montreal Canadiens", "away": "Winnipeg Jets",
         "commence": T},
        [{"home": "MTL Canadiens", "away": "WPG Jets",
          "commence": "2026-10-25T13:00:00Z"}])
    chk(f"different kickoff rejected ({why})", c is None)

    amb = [{"home": "Manchester United", "away": "Liverpool",
            "commence": T},
           {"home": "Manchester City", "away": "Liverpool",
            "commence": T}]
    c, s, why = match_event(
        {"home": "Manchester", "away": "Liverpool", "commence": T}, amb)
    chk(f"ambiguous Manchester refused ({why if c is None else 'MATCHED'})",
        c is None)

    c, s, why = match_event(
        {"home": "Arsenal", "away": "Chelsea", "commence": T},
        [{"home": "Everton", "away": "Fulham", "commence": T}])
    chk("unrelated fixture rejected", c is None)

    log("")
    log("  the derby that must NOT collapse:")
    s1 = similarity("Manchester United", "Manchester City")
    chk(f"Man Utd vs Man City similarity {s1:.2f} < floor {MIN_SCORE}",
        s1 < MIN_SCORE)
    s2 = similarity("Inter Milan", "AC Milan")
    chk(f"Inter vs AC Milan similarity {s2:.2f} < floor", s2 < MIN_SCORE)

    log("")
    log("  diacritics and club noise:")
    chk("FC Zurich == FC Zürich", similarity("FC Zürich", "FC Zurich") > 0.9)
    chk("Real Madrid != Real Sociedad",
        similarity("Real Madrid", "Real Sociedad") < MIN_SCORE)

    log("")
    log("  market and outcome keys:")
    chk("'Moneyline' -> h2h", market_key("Moneyline") == "h2h")
    chk("'Match Odds' -> h2h_3way", market_key("Match Odds") == "h2h_3way")
    chk("unknown market -> None (never guessed)",
        market_key("Player Shots On Target Alt") is None)
    chk("'Over 2.5' -> ('over', 2.5)", outcome_key("Over 2.5") == ("over", 2.5))
    chk("'U 4.5' -> ('under', 4.5)", outcome_key("U 4.5") == ("under", 4.5))
    chk("line is part of identity: 2.5 != 3.5",
        outcome_key("Over 2.5") != outcome_key("Over 3.5"))

    log("")
    log(f"  {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
