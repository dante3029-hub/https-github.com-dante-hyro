"""Long-only breakout sleeve -- the thing the book lacks.

WHY: the book is dollar-neutral, so it structurally cannot capture a bull run.
Bull half 1.12 vs bear half 2.25. A long-only momentum/breakout sleeve is the
natural complement -- it should earn most of its return in exactly the regime
where the neutral book struggles.

Triggers tested:
    nday_high   close breaks the N-day high
    pct_move    price up X% over Y days
    vol_break   N-day high AND volume > 2x its 20d average

Long only. No short leg to bleed in a rally.
"""
import os, glob
import numpy as np, pandas as pd
TAKER='/tmp/hyro/taker_data'
FEE, CAP = 0.00085, 0.15
CORE24=['1000PEPE','1000RATS','1000SHIB','AAVE','ADA','AVAX','BCH','BNB','DOGE','DOT',
        'ETH','FIL','LDO','LINK','LTC','NEAR','SOL','SUI','TRX','UNI','WLD','XLM','XRP','ZEC']
ALL=sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{TAKER}/*_1h.csv')})

def load(sym, tf=24):
    df=pd.read_csv(f'{TAKER}/{sym}_1h.csv')
    df['ts']=pd.to_datetime(df['open_time'],unit='ms'); df=df.set_index('ts').sort_index()
    r='1D' if tf==24 else f'{tf}h'
    return pd.DataFrame({'open':df['open'].resample(r).first(),'high':df['high'].resample(r).max(),
        'low':df['low'].resample(r).min(),'close':df['close'].resample(r).last(),
        'volume':df['volume'].resample(r).sum()}).dropna()

def atr(d,n=14):
    h,l,c=d['high'],d['low'],d['close']
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def run(mode, syms=None, tf=24, lookback=20, pct=0.30, pct_days=3,
        stop=3.0, hold=20, trail=False, shuffle=False, seed=0):
    syms = syms or CORE24
    rng=np.random.default_rng(seed); by={}; ne=0
    for s in syms:
        p=f'{TAKER}/{s}_1h.csv'
        if not os.path.exists(p): continue
        try: d=load(s,tf)
        except Exception: continue
        if len(d)<200: continue
        o,h,l,c,v = (d['open'].values,d['high'].values,d['low'].values,
                     d['close'].values,d['volume'].values)
        A=atr(d).values; n=len(d)
        vma=pd.Series(v).rolling(20).mean().values
        i=lookback+2
        while i < n-2:
            trig=False
            if mode=='nday_high':
                trig = c[i] > np.nanmax(h[i-lookback:i])
            elif mode=='pct_move':
                if i>pct_days: trig = (c[i]/c[i-pct_days]-1) >= pct
            elif mode=='vol_break':
                trig = (c[i] > np.nanmax(h[i-lookback:i])
                        and np.isfinite(vma[i]) and v[i] > 2*vma[i])
            if not trig or not np.isfinite(A[i]) or A[i]<=0:
                i+=1; continue
            eb=i+1
            if eb>=n-1: break
            side = -1 if shuffle and rng.random()<0.5 else 1
            ent=o[eb]; stp=ent-side*stop*A[i]; r=None; peak=ent
            for j in range(eb+1, min(eb+1+hold, n)):
                if trail:
                    peak=max(peak,h[j]); stp=max(stp, peak-stop*A[i])
                if side>0 and l[j]<=stp: r=(stp-ent)/ent-2*FEE; break
                if side<0 and h[j]>=stp: r=(ent-stp)/ent-2*FEE; break
            if r is None:
                j=min(eb+hold,n-1); r=side*(c[j]-ent)/ent-2*FEE
            if not np.isfinite(r): continue
            ne+=1; by.setdefault(d.index[eb].date(),[]).append(r)
            i=eb+hold
    if not by: return None,0
    cal=pd.date_range(min(by),max(by),freq='D')
    return pd.Series([sum(x*min(1/len(vv),CAP) for x in vv) if (vv:=by.get(dd.date(),[])) else 0.0
                      for dd in cal], index=cal), ne

def stats(x):
    x=x.dropna()
    if len(x)<50 or x.std()==0: return 0.0,0.0
    return float(x.mean()/x.std()*np.sqrt(365)), float(x.mean()*365*100)
