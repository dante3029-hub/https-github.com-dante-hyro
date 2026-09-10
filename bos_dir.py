"""BOS long vs short on MATCHED-VENUE delta.

The earlier -1.21 (long) vs +1.19 (short) was measured on clean_panel's taker
column, which was another venue's volume. Long needed dn > 0 -- always true, so
longs ran with NO confirmation filter. Short needed dn < 0 -- never true. That
comparison cannot have meant what it appeared to.
"""
import os, glob, csv
import numpy as np, pandas as pd
PRICE='/tmp/hyro/price_data'; TAKER='/tmp/hyro/taker_data'
FEE, CAP = 0.00085, 0.15
PIVOT_K, ATR_STOP, TP_ATR, EXIT_BAR, MAXPOS = 5, 2.0, 3.0, 30, 6
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{TAKER}/*_1h.csv')})

def load4h(sym, tf=4):
    df = pd.read_csv(f'{TAKER}/{sym}_1h.csv')
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('ts').sort_index()
    r=f'{tf}h'
    d = pd.DataFrame({'open':df['open'].resample(r).first(),
                      'high':df['high'].resample(r).max(),
                      'low':df['low'].resample(r).min(),
                      'close':df['close'].resample(r).last(),
                      'delta':df['delta'].resample(r).sum()}).dropna()
    return d if len(d) > 400 else None

def atr(h,l,c,n=14):
    N=len(c); tr=np.zeros(N); out=np.full(N,np.nan)
    for i in range(1,N):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    a=np.nan
    for i in range(1,N):
        a = tr[i] if not np.isfinite(a) else (a*13+tr[i])/14
        if i>=14: out[i]=a
    return out

def pivots(h,l,k):
    N=len(h); ph=np.full(N,np.nan); pl=np.full(N,np.nan)
    for i in range(k,N-k):
        if h[i]==h[i-k:i+k+1].max(): ph[i+k]=h[i]
        if l[i]==l[i-k:i+k+1].min(): pl[i+k]=l[i]
    return ph,pl

def run(side, use_delta=True, shuffle=False, seed=0, tf=4):
    """side: -1 short (the live config), +1 long."""
    rng=np.random.default_rng(seed); trades=[]
    for sym in SYMS:
        d = load4h(sym, tf)
        if d is None: continue
        o,h,l,c = (d['open'].values,d['high'].values,d['low'].values,d['close'].values)
        dn = d['delta'].values; ts = d.index
        A = atr(h,l,c); ph,pl = pivots(h,l,PIVOT_K)
        lastH=lastL=prevH=prevL=np.nan; state=0; i=PIVOT_K+1
        while i < len(c)-1:
            if np.isfinite(ph[i]): prevH,lastH = lastH,ph[i]
            if np.isfinite(pl[i]): prevL,lastL = lastL,pl[i]
            if np.isfinite(lastH) and np.isfinite(prevH) and np.isfinite(lastL) and np.isfinite(prevL):
                if lastH>prevH and lastL>prevL: state=1
                elif lastH<prevH and lastL<prevL: state=-1
            trig=False
            if side<0 and state==-1 and np.isfinite(lastL):
                trig = c[i]<lastL and c[i-1]>=lastL
                if use_delta: trig = trig and dn[i]<0
            elif side>0 and state==1 and np.isfinite(lastH):
                trig = c[i]>lastH and c[i-1]<=lastH
                if use_delta: trig = trig and dn[i]>0
            if trig and np.isfinite(A[i]) and A[i]>0 and i+1<len(c):
                s = side
                if shuffle: s = int(rng.choice([-1,1]))
                ent = o[i+1]
                stop = ent - s*ATR_STOP*A[i]; tp = ent + s*TP_ATR*A[i]
                px=None
                for j in range(i+1, min(i+1+EXIT_BAR, len(c))):
                    if s<0 and h[j]>=stop: px=stop; break
                    if s>0 and l[j]<=stop: px=stop; break
                    if s<0 and l[j]<=tp: px=tp; break
                    if s>0 and h[j]>=tp: px=tp; break
                if px is None: px=c[min(i+EXIT_BAR,len(c)-1)]
                k=min(i+EXIT_BAR,len(c)-1)
                trades.append((ts[k], s*(px-ent)/ent - 2*FEE))
                i = i+EXIT_BAR
            i += 1
    if not trades: return None,0
    by={}
    for t,r in trades: by.setdefault(t.date(),[]).append(r)
    cal=pd.date_range(min(by),max(by),freq='D')
    ser=[sum(x*min(1/len(v),CAP) for x in v) if (v:=by.get(dd.date(),[])) else 0.0 for dd in cal]
    return pd.Series(ser,index=cal), len(trades)

def stats(x):
    x=x.dropna()
    if len(x)<50 or x.std()==0: return 0.0,0.0
    return float(x.mean()/x.std()*np.sqrt(365)), float(x.mean()*365*100)

if __name__=='__main__':
    print('BOS DIRECTION TEST — matched-venue delta\n')
    print(f"{'  config':<32}{'Sharpe':>9}{'ctrl':>8}{'GAP':>8}{'ann%':>8}{'1st':>8}{'2nd':>8}{'both':>7}{'n':>7}")
    for lbl,kw in (('SHORT + delta filter',{'side':-1}),
                   ('SHORT, no filter',{'side':-1,'use_delta':False}),
                   ('LONG + delta filter',{'side':1}),
                   ('LONG, no filter',{'side':1,'use_delta':False})):
        r,n = run(**kw)
        if r is None or n<30: print(f'  {lbl:<30}  too few ({n})'); continue
        c,_ = run(**{**kw,'shuffle':True,'seed':11})
        sh,ann = stats(r); cs = stats(c)[0] if c is not None else 0.0
        hh=len(r)//2; s1,s2 = stats(r.iloc[:hh])[0], stats(r.iloc[hh:])[0]
        print(f'  {lbl:<30}{sh:>9.2f}{cs:>8.2f}{sh-cs:>8.2f}{ann:>7.1f}%{s1:>8.2f}{s2:>8.2f}'
              f'{("yes" if s1>0 and s2>0 else "NO"):>7}{n:>7}')
    print('\n  earlier (corrupt delta): SHORT +1.19, LONG -1.21 -- both unreliable')
