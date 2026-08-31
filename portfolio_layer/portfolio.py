"""
Phase 3 orchestrator. Combines:
  - signal_engine.engine.SignalSnapshot (per-sleeve target weight vectors, Phase 2)
  - position_sizer.compute_sleeve_multipliers() (within/between-strategy risk weights)
  - risk_overlay.RiskState (L=1.70 leverage, ORIG_THROTTLE drawdown cut, -$3,000 kill switch)

into ONE final dict of per-coin dollar target positions, ready for Phase 4
(execution) to diff against current live positions and generate orders.

This module does NOT place orders and does NOT itself fetch market data --
it is a pure function of (today's SignalSnapshot, each sleeve's trailing P&L
history, current RiskState). Wiring it to a live data/execution loop is
Phase 4/5 scope.
"""
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .position_sizer import compute_sleeve_multipliers
from .risk_overlay import RiskState, L_DEFAULT

REFERENCE_NOTIONAL = 200_000.0  # $200k HyroTrader account size -- every sleeve's raw weight vector is against this
SLEEVE_NAMES = ("main", "short", "flow", "delta", "relvol", "bos")

# ---------------------------------------------------------------------------
# HyroTrader compliance caps (2026-08-10). Both rules were provided verbatim
# by the account owner and cross-checked against this codebase -- neither had
# ANY enforcement anywhere before this. See STRATEGY_SPECIFICATION.md cross-
# check writeup for the audit trail.
#
# 2026-08-10 CORRECTION: DEFAULT_STOP_FRAC is NO LONGER the same value as
# bot/config.py's LIVE_STOP_LOSS_FRAC. They were decoupled:
#     LIVE_STOP_LOSS_FRAC = 0.40  -> the REAL protective stop sent to the exchange
#     DEFAULT_STOP_FRAC   = 0.60  -> denominator for the per-trade dollar cap ONLY
# Do NOT re-couple them. The 0.60 here is a deliberately CONSERVATIVE denominator
# (it sizes notional as if the stop were 3x wider than it is), which is the
# LOOSEST value in reference_impl.py's
# documented 40-80% stop plateau. That choice is being preserved unchanged:
# the max-loss-per-trade cap below is enforced by capping NOTIONAL (position
# size), never by tightening this stop fraction. Tightening the stop instead
# would trade a different, backtested-to-look-better-as-it-tightens strategy
# (monotonic Sharpe improvement as stop tightens is the classic overfit
# signature on a weekly-held cross-sectional book) -- flagged explicitly and
# rejected as the implementation mechanism for that reason.
DEFAULT_STOP_FRAC = 0.60

# 2026-08-10 FIX. The AGGREGATE cap answers a different question from the
# per-trade cap, and must therefore use a different stop fraction.
#
#   per-trade cap : "how big can ONE leg be before a stop-out breaches $6,000?"
#                   -> use the CONSERVATIVE 0.60 denominator (safety margin for a
#                      stop that fills worse than modelled).
#
#   aggregate cap : "if EVERY open leg stopped out at once, what is the total?"
#                   -> they would stop at the REAL stop distance (0.24), not 0.60.
#                      Using 0.60 here overstates the crash loss by 2.5x and
#                      throttles live size 2.5x harder than the risk warrants.
#
# Measured impact of the old behaviour: with 34 legs the cap scaled the book to
# 14.7% of target, pinning LIVE gross at $50,000 (0.25x) no matter the leg count
# -- while the backtest that produced the headline Sharpe/CAGR ran at L=1.70
# ($340,000 gross). The live bot could not reach the backtested configuration.
AGGREGATE_STOP_FRAC_DEFAULT = 0.24   # keep in sync with bot/config.LIVE_STOP_LOSS_FRAC

# HyroTrader rule: "the realized loss on any individual trade must not exceed
# 3% of the initial account balance." 3% of $200,000 = $6,000. HyroTrader
# states this is NOT monitored by their automated system (manual review only)
# -- this cap is this codebase's own enforcement, not a mirror of anything
# HyroTrader already checks.
MAX_LOSS_PER_TRADE = 6_000.0

# 2026-08-10: aggregate correlated-crash risk cap, added on account-owner
# instruction after confirming the per-trade cap above is NOT pooled across
# concurrently open legs. With ~34 typical concurrent positions (see
# bot/burn_in.py), an unbounded per-leg cap allows aggregate loss-if-all-
# stopped up to 34 x $6,000 = $204,000 (102% of the $200k account) in a
# correlated crash -- far beyond the spirit of a 3% rule, even though each
# INDIVIDUAL trade is still compliant on its own.
#
# Rejected: pooling MAX_LOSS_PER_TRADE by dividing by leg count (e.g. $6,000
# / 34 = $176/leg pool). That collapses the per-leg notional cap to ~$294
# (0.15% of account) and would crush normal-regime position sizing to
# near-nothing on every cycle, not just tail scenarios.
#
# Chosen instead: a soft aggregate ceiling that only binds when the SUM of
# (notional x stop_frac) across ALL currently open legs would exceed it --
# proportionally scaling every open leg down together in that case. Under
# typical operation this does not bind (legs stay at their normal computed
# sizes); it only engages as a correlated-crash circuit breaker. Set to 15%
# of the $200k account ($30,000) -- below the current unbounded worst case
# (102%) and below the realistic all-legs-at-real-stop exposure (40.8%,
# $81,600 at the live 24% stop), while remaining well above normal single-
# cycle risk so it should rarely bind outside a genuine tail event.
AGGREGATE_MAX_LOSS = 30_000.0

# 2026-08-10 NEW: GAP CAP. The per-trade cap protects against a stop-out; it does
# NOT protect against price GAPPING THROUGH the stop, where the fill is far worse
# than the stop level and no stop can help. Worst single-day adverse move observed
# in the traded universe: TRX +95.8% (also FIL +78.4%, XRP +73.2%, ADA +72.0%).
#
#   per-leg notional cap = $6,000 / 0.60 = $10,000
#   a 95.8% gap on $10,000 = $9,580 realised loss  -> BREACHES the $6,000 rule
#
# ASSUMED_ADVERSE_GAP is the gap the book is sized to survive inside $6,000.
# 0.60 covers every day in the sample except the single worst. Set it to 0.96 for
# full historical protection -- that drops the notional cap to $6,250 and costs
# size. This is an explicit, documented risk choice, not an oversight.
ASSUMED_ADVERSE_GAP = 0.60
MAX_NOTIONAL_PER_LEG = MAX_LOSS_PER_TRADE / ASSUMED_ADVERSE_GAP   # $10,000 @0.60

# HyroTrader rule: "you must not allocate more than 5% of your initial account
# balance (including leverage) across all low-cap assets at any time." 5% of
# $200,000 = $10,000, summed across every sleeve and every coin classified as
# low-cap (see bot/config.py LOW_CAP_COINS for the classification and its
# caveats -- it is a manual snapshot, not a live market-cap/listing feed).
LOW_CAP_EXPOSURE_LIMIT = 10_000.0


@dataclass
class SizedPortfolio:
    """Final output: per-sleeve dollar target position per coin, plus the
    full audit trail of multipliers used to produce it (so any single
    number can be traced back to which stage of the pipeline set it)."""
    sleeve_multipliers: Dict[str, float]   # m_main..m_bos, wA, wB (from position_sizer)
    account_multiplier: float              # L * throttle * kill_switch_gate (from risk_overlay), 0.0 if flat
    dollar_targets: Dict[str, Dict[str, float]]  # {sleeve_name: {coin: dollars}}
    gross_notional: float                  # sum of abs(dollar_targets) across all sleeves/coins
    flags: list = field(default_factory=list)  # any warnings surfaced during sizing


def size_portfolio(sleeve_raw_weights: Dict[str, Dict[str, float]],
                    sleeve_return_history: Dict[str, np.ndarray],
                    risk_state: RiskState,
                    window: int = 30,
                    L: float = L_DEFAULT,
                    max_sleeve_multiplier: float = 1.0,
                    stop_fracs: Dict[str, Dict[str, float]] = None,
                    max_loss_per_trade: float = MAX_LOSS_PER_TRADE,
                    low_cap_coins=None,
                    low_cap_exposure_limit: float = LOW_CAP_EXPOSURE_LIMIT,
                    aggregate_max_loss: float = AGGREGATE_MAX_LOSS) -> SizedPortfolio:
    """
    sleeve_raw_weights: {"main": {coin: weight, ...}, "short": {...}, "flow": {...},
                          "delta": {...}, "relvol": {...}, "bos": {...}}
        Each sleeve's dict of per-coin target weight FRACTIONS from
        signal_engine (Phase 2), gross-summing to ~1.0 for a fully-deployed
        sleeve. Missing sleeves are treated as flat (all zeros) with a flag.

    sleeve_return_history: {"main": np.ndarray, ..., "bos": np.ndarray}
        Each sleeve's OWN trailing daily-return history (fraction of
        REFERENCE_NOTIONAL) assuming its raw weight vector was deployed at
        full (~1.0 gross) notional every day. Must be aligned (same end
        date = "yesterday", i.e. NOT including today's not-yet-realized
        return) and >= `window` days long for delta/relvol/bos to size
        non-zero -- shorter histories flag a warning and size those three
        sleeves at 0 (safer than guessing a vol estimate from insufficient data).

    risk_state: current RiskState (tracks peak equity, kill-switch/daily-
        loss-limit trip status for the current session).

    stop_fracs: {"main": {coin: frac}, ...} optional per-sleeve, per-coin
        leg-level stop fraction override, used ONLY to size the max-loss-per-
        trade notional cap below (does not change the actual protective stop
        placed at execution). Missing sleeve/coin entries default to
        DEFAULT_STOP_FRAC (0.60), matching main/flow/delta/relvol's fixed
        leg-level stop. short/bos use ATR-derived stops that vary per slot --
        the orchestrator should supply their real per-coin fraction here
        (see bot/event_sleeves.stop_fracs_from_state); defaulting them to 0.60
        when unknown is the CONSERVATIVE direction (short/bos's real ATR stop
        is typically tighter than 60%, so a real stop_frac would imply a
        LOOSER/larger cap than assuming 0.60 -- defaulting to 0.60 therefore
        never UNDER-clips).

    max_loss_per_trade: HyroTrader's $6,000 (3% of $200k) max realized loss on
        any single trade. Enforced by capping each leg's NOTIONAL at
        max_loss_per_trade / stop_frac -- the stop fraction itself is never
        tightened by this cap (see DEFAULT_STOP_FRAC docstring above for why).
        Set to 0/None to disable (e.g. for an A/B backtest comparison).

    low_cap_coins: set of coin tickers classified as "low-cap" under
        HyroTrader's definition (market cap < $100M, OR 24h volume
        $500K-$5M, OR Innovation Zone listing). See bot/config.py
        LOW_CAP_COINS for the current manual snapshot and its caveats.

    low_cap_exposure_limit: HyroTrader's $10,000 (5% of $200k) cap on
        aggregate notional (summed absolute value, across ALL sleeves) in
        coins classified as low-cap. If the raw aggregate exceeds this, every
        low-cap leg is scaled down proportionally (not just clipped
        individually) so relative sleeve weighting among low-cap legs is
        preserved. Set to 0/None to disable.

    Returns a SizedPortfolio with the full audit trail.
    """
    flags = []
    for name in SLEEVE_NAMES:
        if name not in sleeve_raw_weights or not sleeve_raw_weights[name]:
            flags.append(f"sleeve '{name}' has no raw weights this cycle -- treated as flat")
        if name not in sleeve_return_history or len(sleeve_return_history[name]) == 0:
            flags.append(f"sleeve '{name}' has no return history -- treated as flat, cannot vol-scale")

    def hist(name):
        h = sleeve_return_history.get(name, np.array([]))
        return h if len(h) > 0 else np.zeros(window)

    mult = compute_sleeve_multipliers(
        hist("main"), hist("short"), hist("flow"),
        hist("delta"), hist("relvol"), hist("bos"),
        window=window, max_multiplier=max_sleeve_multiplier,
    )

    for name in ("delta", "relvol", "bos"):
        if len(sleeve_return_history.get(name, [])) < window:
            flags.append(f"sleeve '{name}' has < {window} days of history -- multiplier forced toward 0, "
                         f"not a reliable trailing-vol estimate yet")

    account_mult = risk_state.target_exposure_multiplier(L=L)
    if account_mult == 0.0:
        flags.append("account-level multiplier is 0.0 -- kill switch or daily loss limit tripped this session, "
                     "book should be flat until start_new_session()")

    dollar_targets = {}
    for name in SLEEVE_NAMES:
        raw_weights = sleeve_raw_weights.get(name, {})
        sleeve_mult = mult.get(name, 0.0) * account_mult
        sleeve_dollars = {}
        for coin, w in raw_weights.items():
            d = w * sleeve_mult * REFERENCE_NOTIONAL
            sleeve_dollars[coin] = d
        dollar_targets[name] = sleeve_dollars

    _apply_max_loss_cap(dollar_targets, stop_fracs, max_loss_per_trade, flags)
    _apply_aggregate_loss_cap(dollar_targets, stop_fracs, aggregate_max_loss, flags)
    _apply_low_cap_exposure_cap(dollar_targets, low_cap_coins, low_cap_exposure_limit, flags)

    gross = sum(abs(d) for sleeve in dollar_targets.values() for d in sleeve.values())

    return SizedPortfolio(
        sleeve_multipliers=mult,
        account_multiplier=account_mult,
        dollar_targets=dollar_targets,
        gross_notional=gross,
        flags=flags,
    )


def _apply_max_loss_cap(dollar_targets: Dict[str, Dict[str, float]],
                         stop_fracs: Dict[str, Dict[str, float]],
                         max_loss_per_trade: float,
                         flags: list) -> None:
    """Clip each leg's notional so abs(dollar) * stop_frac <= max_loss_per_trade.
    Mutates dollar_targets in place. Never touches stop_frac itself -- see
    DEFAULT_STOP_FRAC docstring for why that distinction matters."""
    if not max_loss_per_trade or max_loss_per_trade <= 0:
        return
    for sleeve, coins in dollar_targets.items():
        sf_map = (stop_fracs or {}).get(sleeve, {})
        for coin, d in list(coins.items()):
            if d == 0.0:
                continue
            sf = sf_map.get(coin, DEFAULT_STOP_FRAC)
            if not sf or sf <= 0:
                continue
            # binding cap = the TIGHTER of (stop-out cap, gap cap). The gap cap
            # exists because a stop cannot help when price gaps through it.
            cap = min(max_loss_per_trade / sf, MAX_NOTIONAL_PER_LEG)
            if abs(d) > cap + 1e-9:
                clipped = cap if d > 0 else -cap
                flags.append(
                    f"max-loss-per-trade cap: {sleeve}/{coin} notional clipped from "
                    f"${d:,.2f} to ${clipped:,.2f} (stop_frac={sf:.4f}, "
                    f"cap=${cap:,.2f}=${max_loss_per_trade:,.0f}/{sf:.4f})"
                )
                coins[coin] = clipped


def _apply_aggregate_loss_cap(dollar_targets: Dict[str, Dict[str, float]],
                               stop_fracs: Dict[str, Dict[str, float]],
                               aggregate_max_loss: float,
                               flags: list) -> None:
    """Correlated-crash circuit breaker. Computes total loss-if-every-open-leg-
    stops-out (sum of abs(dollar) * stop_frac across ALL open legs, post per-
    leg cap). If that sum exceeds aggregate_max_loss, scales EVERY open leg's
    notional down by the same factor so the sum lands exactly at the cap.
    Preserves relative sizing between legs -- this is a book-wide throttle,
    not a per-leg clip. Set aggregate_max_loss to 0/None to disable.

    Under typical operation (moderate concurrent leg count / sizes), the
    pre-scale sum stays under the cap and this is a no-op -- it only engages
    as a tail-risk brake, unlike dividing MAX_LOSS_PER_TRADE by leg count
    (which would shrink every leg's cap permanently, every cycle).
    """
    if not aggregate_max_loss or aggregate_max_loss <= 0:
        return
    total_loss_if_stopped = 0.0
    for sleeve, coins in dollar_targets.items():
        sf_map = (stop_fracs or {}).get(sleeve, {})
        for coin, d in coins.items():
            if d == 0.0:
                continue
            # REAL stop distance, not the conservative per-trade denominator --
            # see AGGREGATE_STOP_FRAC_DEFAULT. event sleeves supply their own.
            sf = sf_map.get(coin, AGGREGATE_STOP_FRAC_DEFAULT)
            if not sf or sf <= 0:
                continue
            total_loss_if_stopped += abs(d) * sf
    if total_loss_if_stopped <= aggregate_max_loss + 1e-9:
        return
    scale = aggregate_max_loss / total_loss_if_stopped
    n_legs = sum(1 for coins in dollar_targets.values() for d in coins.values() if d != 0.0)
    flags.append(
        f"aggregate-loss-if-all-stopped cap: total \u2248${total_loss_if_stopped:,.2f} across "
        f"{n_legs} open legs exceeded ${aggregate_max_loss:,.2f} cap -- scaling all legs by "
        f"{scale:.4f}"
    )
    for sleeve, coins in dollar_targets.items():
        for coin, d in list(coins.items()):
            if d != 0.0:
                coins[coin] = d * scale


def _apply_low_cap_exposure_cap(dollar_targets: Dict[str, Dict[str, float]],
                                 low_cap_coins,
                                 low_cap_exposure_limit: float,
                                 flags: list) -> None:
    """Scale down (proportionally, across every sleeve) all legs in
    low_cap_coins so their combined absolute notional never exceeds
    low_cap_exposure_limit. Mutates dollar_targets in place."""
    if not low_cap_coins or not low_cap_exposure_limit or low_cap_exposure_limit <= 0:
        return
    total = 0.0
    legs = []
    for sleeve, coins in dollar_targets.items():
        for coin, d in coins.items():
            if coin in low_cap_coins and d != 0.0:
                total += abs(d)
                legs.append((sleeve, coin))
    if total > low_cap_exposure_limit + 1e-9:
        scale = low_cap_exposure_limit / total
        touched_coins = sorted(set(c for _, c in legs))
        for sleeve, coin in legs:
            dollar_targets[sleeve][coin] *= scale
        flags.append(
            f"low-cap exposure cap: aggregate low-cap notional ${total:,.2f} across "
            f"{len(legs)} leg(s) in {touched_coins} exceeded ${low_cap_exposure_limit:,.0f} "
            f"limit -- scaled all low-cap legs by {scale:.4f}"
        )
