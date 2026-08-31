import sys; sys.path.insert(0,"/home/user/workspace")
import numpy as np
from bot.sleeve_history import compute_sleeve_histories
from portfolio_layer.position_sizer import compute_sleeve_multipliers
from portfolio_layer.ab_blend import DEFAULT_WINDOW as W
from bot.config import L_DEFAULT

h=compute_sleeve_histories(); K=("main","short","flow","delta","relvol","bos")
n=min(len(h[k]) for k in K); H={k:np.asarray(h[k],float)[-n:] for k in K}
ANN=np.sqrt(365)

def book(cap):
    rets=[]
    for t in range(2*W+5, n):
        sub={k:H[k][:t] for k in K}
        m=compute_sleeve_multipliers(sub["main"],sub["short"],sub["flow"],
             sub["delta"],sub["relvol"],sub["bos"],window=W,max_multiplier=cap)
        rets.append(sum(m[s]*H[s][t] for s in K))   # t = next unseen day
    return np.array(rets)

print(f"{'cap':>8s} {'ann vol':>9s} {'ann vol xL':>11s} {'Sharpe':>8s} {'CAGR':>8s} {'maxDD':>8s}")
target=0.1462
best=None
for cap in (1.0, 1.5, 2.0, 3.0, 5.0, 1e9):
    r=book(cap)
    v=r.std()*ANN; sh=r.mean()/r.std()*ANN if r.std()>0 else 0
    eq=np.cumprod(1+r*L_DEFAULT); dd=float((1-eq/np.maximum.accumulate(eq)).max())
    cagr=eq[-1]**(365/len(r))-1
    lab="uncapped" if cap>1e8 else f"{cap:.1f}"
    print(f"{lab:>8s} {v:9.4f} {v*L_DEFAULT:11.4f} {sh:8.4f} {cagr:8.2%} {dd:8.2%}")
    if best is None or abs(v-target)<best[0]: best=(abs(v-target),lab,v)
print(f"\nbacktest unlevered ann vol was 0.1462 (Sharpe 2.475, CAGR 42.08%, maxDD 6.28%)")
print(f"closest cap to backtest vol: {best[1]} at ann vol {best[2]:.4f}")
