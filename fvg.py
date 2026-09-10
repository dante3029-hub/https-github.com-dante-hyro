"""Fair Value Gap [LuxAlgo] -- backtest.

SOURCE LOGIC (docs/sr_breaks.txt lines 218-448):
    threshold = auto ? cum((high-low)/low)/bar_index : thresholdPer/100   (default 0)
    bull_fvg  = low > high[2] and close[1] > high[2] and (low-high[2])/high[2] > threshold
    bear_fvg  = high < low[2] and close[1] < low[2] and (low[2]-high)/high  > threshold
    mitigation: price trades back through the gap.

Four tradeable readings, all tested:
    detect     enter in the gap's direction on formation (continuation)
    fade       enter against it (the gap fills)
    mitigate   enter in the gap's direction when price returns to it (retest)
    auto_thr   detect, but with the source's adaptive threshold
"""
import os, glob, sys
import numpy as np, pandas as pd
FEE, CAP, MAX_HOLD, ATR_STOP = 0.00085, 0.15, 15, 2.0
PRICE='/tmp/hyro/price_data'; OIDIR='/tmp/hyro/oi_data'
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{PRICE}/*_1h.csv')})

def load(sym, tf):
    df=pd.read_csv(f'{PRICE}/{sym}_1h.csv')
    ts='open_time' if 'open_time' in df.columns else 'timestamp'
    df['ts']=pd.to_datetime(df[ts],unit='ms',utc=True).dt.tz_localize(None)
    df=df.set_index('ts').sort_index()
    r = f'{tf}h' if tf < 24 else '1D'
    return pd.DataFrame({'open':df['open'].resample(r).first(),'high':df['high'].resample(r).max(),
        'low':df['low'].resample(r).min(),'close':df['close'].resample(r).last(),
        'volume':df['volume'].resample(r).sum()}).dropna()

def load_oi(sym):
    p=f'{OIDIR}/{sym}_oi_1h.csv'
    if not os.path.exists(p): return None
    df=pd.read_csv(p)
    df['ts']=pd.to_datetime(df['timestamp'],unit='ms',utc=True).dt.tz_localize(None)
    return df.set_index('ts').sort_index()['open_interest']

def atr(d,n=14):
    h,l,c=d['high'],d['low'],d['close']
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def stats(x):
    x=x.dropna()
    if len(x)<50 or x.std()==0: return 0.0,0.0,0.0
    eq=x.cumsum()
    return float(x.mean()/x.std()*np.sqrt(365)), float(x.mean()*365*100), float((eq.cummax()-eq).max()*100)

def fvgs(d, thr_pct=0.0, auto=False):
    """Returns list of (bar, is_bull, gap_max, gap_min)."""
    h,l,c = d['high'].values, d['low'].values, d['close'].values
    n=len(d); out=[]
    if auto:
        rng=(d['high']-d['low'])/d['low']
        thr=rng.expanding().mean().values
    else:
        thr=np.full(n, thr_pct/100.0)
    for i in range(2,n):
        if l[i] > h[i-2] and c[i-1] > h[i-2] and (l[i]-h[i-2])/h[i-2] > thr[i]:
            out.append((i, True, l[i], h[i-2]))
        elif h[i] < l[i-2] and c[i-1] < l[i-2] and (l[i-2]-h[i])/h[i] > thr[i]:
            out.append((i, False, l[i-2], h[i]))
    return out

def run(tf, mode, thr=0.0, auto=False, oi_filter=False, shuffle=False, seed=0,
        depth=0.0, window=20):
    rng_=np.random.default_rng(seed); by={}; ne=0
    for sym in SYMS:
        try: d=load(sym,tf)
        except Exception: continue
        if len(d)<400: continue
        A=atr(d).values; o=d['open'].values; hi=d['high'].values
        lo=d['low'].values; c=d['close'].values; n=len(d)
        oi_up=None
        if oi_filter:
            oi=load_oi(sym)
            if oi is None: continue
            od=oi.resample(f'{tf}h').last(); oi_up=(od-od.shift(1))>0
        for (i,is_bull,gmax,gmin) in fvgs(d,thr,auto):
            if mode in ('detect','fade','auto_thr'):
                eb=i+1; side = 1 if is_bull else -1
                if mode=='fade': side=-side
            else:   # mitigate -- wait for price to return INTO the gap.
                    # depth 0.0 = touches the near edge, 1.0 = fills it fully.
                    # Entry is in the GAP'S direction: green -> long, red -> short.
                eb=None
                span = abs(gmax-gmin)
                lvl = (gmax - depth*span) if is_bull else (gmin + depth*span)
                for j in range(i+1, min(i+1+window, n)):
                    if (is_bull and lo[j] <= lvl) or ((not is_bull) and hi[j] >= lvl):
                        eb=j+1; break
                if eb is None: continue
                side = 1 if is_bull else -1
            if eb is None or eb>=n-1 or not np.isfinite(A[eb]) or A[eb]<=0: continue
            if oi_filter:
                v=oi_up.reindex([d.index[eb]],method='ffill')
                if not (len(v) and bool(v.iloc[0])): continue
            if shuffle: side=int(rng_.choice([-1,1]))
            ent=o[eb]; stop=ent-side*ATR_STOP*A[eb]; r=None
            for j in range(eb+1,min(eb+1+MAX_HOLD,n)):
                if side>0 and lo[j]<=stop: r=(stop-ent)/ent-2*FEE; break
                if side<0 and hi[j]>=stop: r=(ent-stop)/ent-2*FEE; break
            if r is None:
                j=min(eb+MAX_HOLD,n-1); r=side*(c[j]-ent)/ent-2*FEE
            if not np.isfinite(r): continue
            ne+=1; by.setdefault(d.index[eb].date(),[]).append(r)
    if not by: return None,0
    cal=pd.date_range(min(by),max(by),freq='D')
    ser=[sum(x*min(1/len(v),CAP) for x in v) if (v:=by.get(dd.date(),[])) else 0.0 for dd in cal]
    return pd.Series(ser,index=cal), ne

if __name__=='__main__':
    print('FAIR VALUE GAP [LuxAlgo]\n')
    print(f"{'  config':<32}{'Sharpe':>8}{'ctrl':>7}{'GAP':>7}{'ann%':>8}{'1st':>7}{'2nd':>7}{'both':>6}{'n':>7}")
    for tf in (4,6):
        for lbl,kw in ((f'{tf}h touch edge',{'mode':'mitigate','depth':0.0}),
                       (f'{tf}h 50% into gap',{'mode':'mitigate','depth':0.5}),
                       (f'{tf}h full fill',{'mode':'mitigate','depth':1.0}),
                       (f'{tf}h full fill +OI',{'mode':'mitigate','depth':1.0,'oi_filter':True}),
                       (f'{tf}h 50% +OI',{'mode':'mitigate','depth':0.5,'oi_filter':True}),
                       (f'{tf}h full fill, 50bar win',{'mode':'mitigate','depth':1.0,'window':50}),
                       (f'{tf}h 50%, thr 0.5%',{'mode':'mitigate','depth':0.5,'thr':0.5})):
            r,n=run(tf,**kw)
            if r is None or n<40: print(f'  {lbl:<30}  too few ({n})'); continue
            c,_=run(tf,**{**kw,'shuffle':True,'seed':7})
            sh,ann,dd=stats(r); csh=stats(c)[0] if c is not None else 0.0
            hh=len(r)//2; s1,s2=stats(r.iloc[:hh])[0],stats(r.iloc[hh:])[0]
            print(f'  {lbl:<30}{sh:>8.2f}{csh:>7.2f}{sh-csh:>7.2f}{ann:>7.1f}%{s1:>7.2f}{s2:>7.2f}{("yes" if s1>0 and s2>0 else "NO"):>6}{n:>7}')
        print()
