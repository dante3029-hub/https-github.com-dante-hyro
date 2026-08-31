"""
Live signal engine for the Bybit 200k prop-eval bot.

Six sleeves, two universes:

  Strategy A (universe: run/hist, 25 coins)
    Main   - sleeve_A   : momentum(14d) + funding-carry(7d), 5 long / 5 short, weekly
    Flow   - sleeve_F   : taker-delta divergence ensemble (lookbacks 2/3/5d), 5L/5S, weekly
    Short  - sleeve_S   : bear-regime market-structure short, 4h bars, event-driven,
                           RECONSTRUCTED FROM PROSE SPEC (not verified against a real
                           reference backtest -- see signal_engine/sleeve_short.py docstring)

  Strategy B (universe: clean_panel/hist, 27 coins)
    DELTA  : cross-sectional taker-delta ranking, 6L/6S, weekly
    RELVOL : cross-sectional relative-volume ranking, 6L/6S, weekly
    BOS    : per-coin market-structure short, 4h bars, event-driven

Every sleeve module in this package calls the *exact same functions* proven out
in the backtest reference files (reference_impl.py, option1_reference.py,
sleeve_S_reconstructed.py) rather than reimplementing the math -- this is
intentional so that "signal parity" is true by construction, not by luck.
See parity_test.py for the numerical diff that verifies this.
"""
