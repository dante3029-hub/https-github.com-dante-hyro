"""Sizing against a STATIC floor, with daily kill-switch variants.

Account: $183,800 current, $180,000 static floor -> buffer $3,800.
The floor is STATIC, so the buffer GROWS as you profit -- an early loss is far
more dangerous than a late one. That is why path matters and a single
"max drawdown" number is misleading.

Book Sharpe 3.68 backtested. Modelled at 1.8 (the manuals' halving) and at
2.5 as a middle case, because sizing on the backtest figure is how accounts die.
"""
import numpy as np

EQUITY = 200_000.0
FLOOR = 180_000.0
TARGET = 220_000.0        # +$20k to pass
DAYS = 120
SIMS = 20_000


def run(daily_vol, sharpe, kill=None, degear=None, seed=0):
    """kill: (loss, days_off) -- stop for N days after losing `loss` in a day.
    degear: list of (drawdown_from_start, multiplier)."""
    rng = np.random.default_rng(seed)
    mu = sharpe * daily_vol / np.sqrt(365)
    fails = 0; passes = 0; finals = []
    for _ in range(SIMS):
        eq = EQUITY; off = 0; done = False
        for d in range(DAYS):
            if off > 0:
                off -= 1
                continue
            mult = 1.0
            if degear:
                dd = EQUITY - eq
                for thr, m in degear:
                    if dd >= thr:
                        mult = m
            r = rng.normal(mu, daily_vol) * mult
            eq += r
            if kill and r <= -kill[0]:
                off = kill[1]
            if eq <= FLOOR:
                fails += 1; done = True; break
            if eq >= TARGET:
                passes += 1; done = True; break
        if not done:
            finals.append(eq)
    return fails/SIMS*100, passes/SIMS*100, np.mean(finals) if finals else 0


if __name__ == "__main__":
    print(f"equity ${EQUITY:,.0f}  floor ${FLOOR:,.0f}  "
          f"buffer ${EQUITY-FLOOR:,.0f}  target ${TARGET:,.0f}\n")
    for sh in (1.8, 2.5, 3.68):
        print(f"  ===== Sharpe {sh} =====")
        print(f"  {'daily vol':<14}{'fail%':>8}{'pass%':>8}{'median end':>13}")
        for v in (500, 750, 1000, 1500, 2000, 3000):
            f, p, m = run(v, sh)
            print(f"  ${v:<13,}{f:>8.1f}{p:>8.1f}{m:>13,.0f}")
        print()
