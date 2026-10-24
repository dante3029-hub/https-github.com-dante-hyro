#!/usr/bin/env python3
"""
consensus.py -- is this arb real, or is it one book being wrong?

THE PROBLEM IT SOLVES. An arb scanner shows you a 17% "guaranteed
profit". The maths is correct and the bet is not, because the entire
edge sits on one book quoting a price nineteen points away from the
fourteen other books on the same market. That is not a soft line, it is
an error, and errors get voided under the palpable-error clause every
book has.

Voiding does NOT confiscate the stake -- that money comes back. The loss
comes from the OTHER leg, which is not voided, is still live, and was
only ever sensible as a hedge. Kill one side of an arb and the survivor
is a naked bet on whatever it was covering.

So this module answers two questions about any opportunity:

  1. HOW MUCH OF THE EDGE IS ONE BOOK? Compute the consensus fair price
     from every book quoting the market, and report each leg's deviation
     in implied-probability POINTS. A leg inside the pack is a real
     price. A leg 19 points outside it is a typo with odds attached.

  2. WHAT IS THE BET, REALLY? If the outlier leg gets voided, you hold
     the other one naked. Price that: EV if honoured, EV if voided, and
     the break-even probability that the book stands behind its own
     mistake. That converts "17% risk-free" into "profitable only if
     they honour it more than 77% of the time", which is a claim you
     can actually assess.

DE-VIGGING. A book's two prices include its margin, so raw implied
probabilities sum above 1. Where both sides are available the margin is
removed proportionally before taking the consensus. Where only one side
is visible -- which is what you get from a screenshot -- raw implied
probabilities are compared instead and the output says so, because a
19-point outlier is obvious either way and a 1-point one is not
measurable without the other side.

WHY THE MEDIAN, NOT THE MEAN. The outlier is exactly what is being
detected. A mean is dragged toward it; the median is not. With one book
at 1.357 and fourteen between 1.05 and 1.10, the mean moves the
reference and shrinks the apparent deviation of the thing you are
hunting.

USAGE
    python3 -u consensus.py --selftest
    python3 -u consensus.py --check \
        --ladder 1.357,1.07,1.09,1.06,1.09,1.10,1.09,1.08,1.08,1.05 \
        --leg-odds 1.357 --leg-stake 1000 \
        --other-odds 8.5 --other-stake 159.66
"""
from __future__ import annotations

import argparse
import statistics

OUTLIER_PP = 5.0          # implied-probability points from consensus


def log(m=""):
    print(m, flush=True)


def implied(odds):
    return 1.0 / odds if odds and odds > 1.0 else float("nan")


def devig_pair(o_a, o_b):
    """Fair probabilities for a two-way market, margin removed
    proportionally. Returns (p_a, p_b) summing to 1."""
    pa, pb = implied(o_a), implied(o_b)
    s = pa + pb
    if not s or s <= 0:
        return float("nan"), float("nan")
    return pa / s, pb / s


def consensus_probability(ladder, paired=None):
    """Consensus fair probability for an outcome.

    ladder: decimal odds for THIS outcome across books.
    paired: optional matching odds for the OPPOSITE outcome at the same
            books, same order, for proper de-vigging.

    Median, not mean -- see the module docstring.
    """
    if paired and len(paired) == len(ladder):
        ps = []
        for a, b in zip(ladder, paired):
            if a and b and a > 1 and b > 1:
                ps.append(devig_pair(a, b)[0])
        if ps:
            return statistics.median(ps), True
    ps = [implied(o) for o in ladder if o and o > 1]
    return (statistics.median(ps), False) if ps else (float("nan"), False)


def outlier_report(ladder, paired=None, labels=None):
    """Every book's distance from consensus, in probability points."""
    cons, devigged = consensus_probability(ladder, paired)
    rows = []
    for i, o in enumerate(ladder):
        if paired and len(paired) == len(ladder) and o and paired[i]:
            p = devig_pair(o, paired[i])[0]
        else:
            p = implied(o)
        rows.append({
            "label": (labels[i] if labels and i < len(labels) else f"#{i+1}"),
            "odds": o,
            "prob": p,
            "dev_pp": (p - cons) * 100.0,
        })
    rows.sort(key=lambda r: abs(r["dev_pp"]), reverse=True)
    return cons, devigged, rows


def void_economics(leg_odds, leg_stake, other_odds, other_stake,
                   p_leg_wins):
    """Price the bet you are ACTUALLY making when one leg may be voided.

    The voided leg returns its stake -- no loss there. The surviving leg
    is unhedged, so it wins or loses outright at its true probability.

      honoured : both settle, the arb pays its guaranteed profit
      voided   : outlier stake returned; the other leg stands alone.
                 It wins when the outlier leg would have LOST
                 (they cover opposite outcomes), i.e. with probability
                 1 - p_leg_wins.
    """
    total = leg_stake + other_stake
    payout_a = leg_stake * leg_odds
    payout_b = other_stake * other_odds
    guaranteed = min(payout_a, payout_b) - total

    p_other_wins = 1.0 - p_leg_wins
    ev_void = (p_other_wins * (other_stake * other_odds - other_stake)
               + p_leg_wins * (-other_stake))

    # break-even probability that the book honours its price
    denom = guaranteed - ev_void
    be = (-ev_void / denom) if denom > 0 else float("nan")
    return {
        "total_stake": total,
        "guaranteed": guaranteed,
        "guaranteed_pct_stake": guaranteed / total * 100.0 if total else 0.0,
        "ev_if_voided": ev_void,
        "breakeven_honour": be,
    }


def check(ladder, leg_odds, leg_stake, other_odds, other_stake,
          paired=None, labels=None, leg_label="your leg"):
    cons, devigged, rows = outlier_report(ladder, paired, labels)
    log("")
    log("  " + "=" * 68)
    log("  CONSENSUS")
    log("  " + "=" * 68)
    log(f"    books quoting this outcome   {len(ladder)}")
    log(f"    consensus fair probability   {cons*100:>6.2f}%"
        f"   (fair odds {1/cons:.3f})")
    log(f"    method                       "
        f"{'de-vigged, both sides' if devigged else 'raw implied, ONE side only'}")
    if not devigged:
        log("    one-sided input, so small deviations are not measurable.")
        log("    A large one still is, and that is what matters here.")
    log("")
    log(f"    {'book':<12}{'odds':>9}{'implied':>10}{'vs consensus':>15}")
    for r in rows:
        flag = "  <-- OUTLIER" if abs(r["dev_pp"]) >= OUTLIER_PP else ""
        log(f"    {r['label']:<12}{r['odds']:>9.3f}{r['prob']*100:>9.1f}%"
            f"{r['dev_pp']:>+14.1f}pp{flag}")

    p_leg = implied(leg_odds)
    dev = (p_leg - cons) * 100.0
    log("")
    log("  " + "=" * 68)
    log("  WHAT IS THE BET, REALLY?")
    log("  " + "=" * 68)
    log(f"    {leg_label} is quoted {leg_odds:.3f}, consensus says"
        f" {1/cons:.3f}")
    log(f"    deviation                    {dev:>+6.1f} probability points")

    e = void_economics(leg_odds, leg_stake, other_odds, other_stake, cons)
    log("")
    log(f"    total staked                 ${e['total_stake']:>10,.2f}")
    log(f"    profit if BOTH stand         ${e['guaranteed']:>10,.2f}"
        f"   ({e['guaranteed_pct_stake']:+.2f}% of stake)")
    log(f"    EV if the outlier is VOIDED  ${e['ev_if_voided']:>10,.2f}")
    log("      (stake returned; the other leg is now unhedged and wins")
    log(f"       only {(1-cons)*100:.1f}% of the time)")
    log("")
    if e["breakeven_honour"] == e["breakeven_honour"]:   # not nan
        log(f"    BREAK-EVEN: this is profitable only if the book honours")
        log(f"    its price more than "
            f"{e['breakeven_honour']*100:.1f}% of the time.")
    log("")
    if abs(dev) >= OUTLIER_PP:
        log(f"    VERDICT: the edge rests on ONE book being {abs(dev):.0f}"
            f" points away from")
        log(f"    every other book on the same market. That is the shape of")
        log(f"    an error, not a soft line, and errors are what the")
        log(f"    palpable-error clause exists for. You are not making a")
        log(f"    risk-free {e['guaranteed_pct_stake']:.1f}% -- you are betting"
            f" they stand behind it.")
        be = e["breakeven_honour"]
        if be == be and be < 0.5:
            log("")
            log(f"    BUT NOTE WHICH WAY THE BAR POINTS. You only need them to")
            log(f"    honour it {be*100:.0f}% of the time, because when they void")
            log(f"    they hand the stake back and the unhedged leg still wins")
            log(f"    sometimes. A low break-even is the opposite of a warning:")
            log(f"    it means the bet survives being voided most of the time.")
            log(f"    The real constraint is more likely the STAKE they accept")
            log(f"    than whether they honour the price.")
    else:
        log(f"    VERDICT: both legs sit inside the pack. The edge does not")
        log(f"    depend on any single book being wrong, which is what a")
        log(f"    real arb looks like. Size it on accepted liquidity.")
    return e


# -------------------------------------------------------------- selftest
def selftest() -> int:
    ok = True

    def chk(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        log(f"    [{'ok' if good else 'FAIL'}] {name}: got {got:.4f}"
            f" want {want:.4f}")

    log("  de-vig, hand-computed:")
    pa, pb = devig_pair(1.90, 1.90)
    chk("devig(1.90,1.90) -> 0.5", pa, 0.5, 1e-12)
    pa, _ = devig_pair(1.50, 3.00)
    chk("devig(1.50,3.00) -> 0.6667", pa, 2.0 / 3.0, 1e-9)

    log("")
    log("  median ignores the outlier the way the mean does not:")
    lad = [1.357, 1.07, 1.09, 1.06, 1.09, 1.10, 1.09, 1.08, 1.08, 1.05,
           1.08, 1.06, 1.06, 1.09, 1.05]
    med, _ = consensus_probability(lad)
    mean_p = sum(implied(o) for o in lad) / len(lad)
    log(f"    median implied {med*100:.2f}%  (fair odds {1/med:.3f})")
    log(f"    mean   implied {mean_p*100:.2f}%  <- dragged by the outlier")
    good = med > mean_p
    ok = ok and good
    log(f"    [{'ok' if good else 'FAIL'}] median is the robust reference")

    log("")
    log("  THE REAL CASE -- Evloev v Volkanovski, Total Rounds Over 1.5")
    log("  ladder read off the OddsJam screenshot, PB first:")
    e = check(lad, leg_odds=1.357, leg_stake=1000.0,
              other_odds=8.5, other_stake=159.66,
              labels=["PB", "NB", "exch", "DK", "1X", "pink", "b365",
                      "B", "B2", "BetRivers", "betwhale", "bodog",
                      "Bovada", "x", "HardRock"],
              leg_label="PB's Over 1.5")

    log("  " + "=" * 68)
    chk("guaranteed profit", e["guaranteed"], 197.48, 0.60)
    # I first wrote 0.769 here, which was my own error carried
    # straight into the test: 76.9% is the break-even VOID rate,
    # and its complement, 23.1%, is the break-even HONOUR rate.
    # Inverting those flips the conclusion from "almost certainly
    # a losing bet" to "probably a winning one", so it is not a
    # cosmetic slip. The test disagreed with the code and the
    # code was right.
    chk("break-even honour rate", e["breakeven_honour"], 0.231, 0.01)
    log("")
    log(f"  {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ladder", default="",
                    help="comma-separated decimal odds for YOUR outcome")
    ap.add_argument("--paired", default="",
                    help="optional: same books' odds for the OTHER outcome")
    ap.add_argument("--labels", default="")
    ap.add_argument("--leg-odds", type=float)
    ap.add_argument("--leg-stake", type=float)
    ap.add_argument("--other-odds", type=float)
    ap.add_argument("--other-stake", type=float)
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.check:
        ap.print_help()
        return 0
    lad = [float(x) for x in a.ladder.split(",") if x.strip()]
    pair = [float(x) for x in a.paired.split(",") if x.strip()] or None
    labs = [x.strip() for x in a.labels.split(",") if x.strip()] or None
    if len(lad) < 3:
        log("  need at least 3 books in --ladder to have a consensus")
        return 2
    check(lad, a.leg_odds, a.leg_stake, a.other_odds, a.other_stake,
          pair, labs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
