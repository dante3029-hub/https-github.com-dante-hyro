"""
Phase 3 - Portfolio / risk layer.

Turns the 6 independent sleeve-level signal streams from `signal_engine/`
into ONE sized, risk-controlled book. Mirrors -- as exactly as a live system
can -- the position-sizing pipeline actually used to produce every backtested
number in STRATEGY.md:

  1. sleeve_combiner.py
     Strategy A: eq_thirds = (main + short + flow) / 3   [equal, un-fit weights]
     Strategy B: core = (nz(delta) + nz(relvol)) / 2, blend = 2/3*nz(core) + 1/3*nz(bos)
                 nz() in reference_impl.py is WHOLE-SAMPLE (look-ahead) vol
                 normalization -- impossible to reproduce live. This module's
                 causal trailing-window analog is a NEW, EXPLICITLY FLAGGED
                 deviation, not a re-validated replica of the printed Sharpe
                 2.38 number. See sleeve_combiner.py docstring.

  2. ab_blend.py
     30-day trailing inverse-vol weight between combo_A and combo_B. This
     step in the ORIGINAL backtest (dynamic_weight_test.py) IS already
     causal (uses only a[t-W:t], b[t-W:t]) -- the function is reused
     verbatim, so this step carries no new deviation.

  3. risk_overlay.py
     L=1.70 leverage overlay, ORIG_THROTTLE drawdown-based exposure cut
     (100% / 50% / 30% at 4% / 8% trailing drawdown from peak equity), and
     the -$3,000 intraday kill switch (flatten, block re-entry until next
     session) -- reusing the exact threshold logic from mc_hourly_ks.py,
     restructured as a live state-carrying tracker instead of a vectorized
     Monte Carlo batch simulator.

  4. portfolio.py
     Orchestrator: combines signal_engine's per-sleeve target weight vectors
     with the scalar risk weights above into one final per-coin, per-sleeve
     dollar target position. Does NOT place orders (Phase 4).
"""
