# v15 spec sections 3-6, redone on matched-venue data

Everything below was originally computed with `clean_panel`'s taker column,
where taker buy volume exceeded TOTAL volume on 60% of bars. This is the
re-run on Binance futures data where volume and taker_buy come from the same
response.

## §4 Signal half-life — CONCLUSION HOLDS, and is stronger

| horizon | clean IC | t-stat | spec (corrupt) |
|---|---|---|---|
| 1d | 0.0050 | **0.75** | 0.0276 (t 3.24) |
| 3d | 0.0275 | 4.11 | 0.0527 |
| 7d | 0.0359 | 5.42 | 0.0463 |
| 14d | 0.0650 | 9.98 | 0.0640 |
| 21d | **0.0793** | **12.05** | 0.0678 |
| 30d | 0.0835 | 12.83 | — |

The 1-day signal was an artifact of the bad column. The edge strengthens
monotonically to 30 days -- and hold=5 (found by grid search) was using the
weakest part of it.

Realised Sharpe follows the IC: **hold 14 gives 1.28 at 26x turnover** vs
hold 5's 1.18 at 73x. Blend 3.24 -> **3.44**.

At 26x turnover the unmeasured maker fill rate stops being load-bearing.

## §3 Ladder — resmom and carry rebuilt

| sleeve | spec | rebuilt | verdict |
|---|---|---|---|
| resmom | 1.42 | 1.09 solo | **REJECTED** -- 0.52 correlated with delta, costs 0.19 in blend |
| carry | 0.92 | 0.12 | **REJECTED** -- 9 configs, none with both halves + |

resmom's best config is 21d lookback (spec used 7d, which is the WORST of six
at 0.77). Same long-lookback preference as delta. But it is measuring nearly
the same thing as delta -- both rank on recent relative performance.

Carry was initially mismeasured: I counted price return only, ignoring the
funding payment that IS the strategy. Booking it properly is worth +0.25
(-0.20 -> 0.05) but the best config is still 0.12.

**27 funding configs tested across z-fade and carry. None viable.**

## §5 The principle — HOLDS, but weaker than claimed

| variant | clean | spec |
|---|---|---|
| baseline | 1.34 | -- |
| inv_vol | **-0.29** | -0.49 |
| cluster_cap | -0.12 | -0.23 |
| tanh | -0.03 | -0.45 |
| scale_out | -0.00 | -0.41 |

Same rank order -- inverse-vol worst, cluster caps second -- across a complete
change of underlying data. But tanh and scale-out are now NEUTRAL rather than
harmful. "Don't touch the weights" is right about direction; the evidence for
it being a strong effect was partly the corrupt column amplifying the damage.

### Dispersion is the fuel — confirmed, and sharper than vol

| cross-sectional dispersion | days | mean bp | Sharpe |
|---|---|---|---|
| low | 417 | +3.3 | 0.77 |
| mid | 420 | +2.9 | 0.57 |
| **high** | 418 | **+15.9** | **2.35** |

Market vol shows the same hump (0.19 / 2.61 / 1.38) but dispersion is the
cleaner cut -- it is the thing itself, not a proxy.

**Gating out high vol is strictly harmful, monotonically:**

| gate | Sharpe | vs base |
|---|---|---|
| none | 1.34 | -- |
| skip top third | 1.08 | -0.26 |
| skip top half | 0.64 | -0.70 |
| skip top 2/3 | 0.11 | **-1.23** |

No risk-off overlay. 19th filter to fail on this book.

## §6 Survivorship — liquidity-screen look-ahead is NOT material

| universe rule | Sharpe |
|---|---|
| current (survivor-selected) | 1.34 |
| point-in-time, 60d | 1.38 |
| point-in-time, 90d | 1.39 |
| point-in-time, 180d | **1.40** |

Rebuilding eligibility every rebalance slightly HELPS. The spec showed a much
bigger jump (+0.44 at 180d); that was probably the corrupt column interacting
with newer coins.

**STILL OPEN and unquantified:** the panel holds 24 coins that exist TODAY.
Anything liquid in 2024 that has since delisted was never downloaded, so no
test on this data can see it. Fixing it needs archived Bybit instrument lists.

## Still not done

- maker vs taker fee sensitivity (1.5bp vs 8.5bp)
- vol targeting (30% band)
- CPCV across partitions
- portfolio simulator (netting + caps) -- 3.44 is a sum of independent streams
