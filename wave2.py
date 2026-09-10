"""Elliott wave (docs/wave.txt) -- retested with matched-venue delta available.

IMPORTANT: the detector itself is PURE PRICE (zigzag pivots + fib ratios). It
never reads taker delta, so the data fix that took the delta sleeve from 0.51 to
1.18 cannot change the wave results on its own. What IS new here:
  * clean delta as a CONFIRMATION filter (ride wave 5 only when flow agrees)
  * a timeframe sweep, which was never run
"""
import os, glob
import numpy as np, pandas as pd
TAKER='/tmp/hyro/taker_data'
FEE, CAP, MAX_HOLD, ATR_STOP = 0.00085, 0.15, 15, 2.0
ZZ = 5
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{TAKER}/*_1h.csv')})

def load(sym, tf):
    df=pd.read_csv(f'{TAKER}/{sym}_1h.csv')
    df['ts']=pd.to_datetime(df['open_time'],unit='ms'); df=df.set_index('ts').sort_index()
    r=f'{tf}h' if tf<24 else '1D'
    return pd.DataFrame({'open':df['open'].resample(r).first(),'high':df['high'].resample(r).max(),
        'low':df['low'].resample(r).min(),'close':df['close'].resample(r).last(),
        'delta':df['delta'].resample(r).sum()}).dropna()

def atr(d,n=14):
    h,l,c=d['high'],d['low'],d['close']
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def zigzag(h,l,n=ZZ):
    N=len(h); out=[]
    for i in range(n,N-n):
        if h[i]==h[i-n:i+n+1].max(): out.append((i,h[i],1,i+n))
        if l[i]==l[i-n:i+n+1].min(): out.append((i,l[i],-1,i+n))
    out.sort(key=lambda x:x[3]); return out

def confidence(legs):
    """ewConfidence() from the source: W3 not shortest 30, W2 retrace 25,
    W3 extension 25, W4 retrace 20."""
    if len(legs)<4: return 0.0
    l1,l2,l3,l4=legs[-4:]; sc=0.0
    if l3>=l1 and l3>=l4: sc+=30
    if l1>0:
        r2=l2/l1
        sc += 25 if 0.382<=r2<=0.786 else (12 if 0.236<=r2<=0.886 else 0)
        e3=l3/l1
        sc += 25 if 1.272<=e3<=2.618 else (12 if 1.0<=e3<=3.0 else 0)
    if l3>0:
        r4=l4/l3
        sc += 20 if 0.236<=r4<=0.5 else (10 if 0.146<=r4<=0.618 else 0)
    return sc

def run(tf, mode='ride_w5', min_conf=0, delta_filter=False, shuffle=False, seed=0):
    rng=np.random.default_rng(seed); by={}; ne=0
    for sym in SYMS:
        try: d=load(sym,tf)
        except Exception: continue
        if len(d)<400: continue
        h,l,c,o=(d['high'].values,d['low'].values,d['close'].values,d['open'].values)
        dn=d['delta'].values; A=atr(d).values; n=len(d)
        piv=zigzag(h,l); known=[]
        for (idx,price,dirn,cb) in piv:
            known.append((idx,price,dirn))
            if cb>=n-MAX_HOLD-2 or len(known)<5: continue
            p=[x[1] for x in known[-5:]]; ds=[x[2] for x in known[-5:]]
            legs=[abs(p[i+1]-p[i]) for i in range(4)]
            if confidence(legs)<min_conf: continue
            bull = ds==[-1,1,-1,1,-1]; bear = ds==[1,-1,1,-1,1]
            side=0
            if mode=='ride_w5':  side = 1 if bull else (-1 if bear else 0)
            elif mode=='fade_w5': side = -1 if bull else (1 if bear else 0)
            if side==0: continue
            eb=cb+1
            if eb>=n-MAX_HOLD or not np.isfinite(A[eb]) or A[eb]<=0: continue
            if delta_filter and not ((side>0 and dn[cb]>0) or (side<0 and dn[cb]<0)): continue
            if shuffle: side=int(rng.choice([-1,1]))
            ent=o[eb]; stop=ent-side*ATR_STOP*A[eb]; r=None
            for j in range(eb+1,min(eb+1+MAX_HOLD,n)):
                if side>0 and l[j]<=stop: r=(stop-ent)/ent-2*FEE; break
                if side<0 and h[j]>=stop: r=(ent-stop)/ent-2*FEE; break
            if r is None:
                j=min(eb+MAX_HOLD,n-1); r=side*(c[j]-ent)/ent-2*FEE
            if not np.isfinite(r): continue
            ne+=1; by.setdefault(d.index[eb].date(),[]).append(r)
    if not by: return None,0
    cal=pd.date_range(min(by),max(by),freq='D')
    ser=[sum(x*min(1/len(v),CAP) for x in v) if (v:=by.get(dd.date(),[])) else 0.0 for dd in cal]
    return pd.Series(ser,index=cal), ne

def stats(x):
    x=x.dropna()
    if len(x)<50 or x.std()==0: return 0.0,0.0
    return float(x.mean()/x.std()*np.sqrt(365)), float(x.mean()*365*100)

if __name__=='__main__':
    print('ELLIOTT WAVE — timeframe sweep + clean-delta confirmation (both new)\n')
    print(f"{'  config':<34}{'Sharpe':>9}{'ctrl':>8}{'GAP':>8}{'1st':>8}{'2nd':>8}{'both':>7}{'n':>7}")
    for tf in (4,6,12,24):
        for lbl,kw in ((f'{tf}h ride W5',{'mode':'ride_w5'}),
                       (f'{tf}h ride W5 + delta',{'mode':'ride_w5','delta_filter':True}),
                       (f'{tf}h ride W5 conf>=70',{'mode':'ride_w5','min_conf':70}),
                       (f'{tf}h fade W5',{'mode':'fade_w5'})):
            r,n=run(tf,**kw)
            if r is None or n<30: print(f'  {lbl:<32}  too few ({n})'); continue
            c,_=run(tf,**{**kw,'shuffle':True,'seed':9})
            sh,ann=stats(r); cs=stats(c)[0] if c is not None else 0.0
            hh=len(r)//2; s1,s2=stats(r.iloc[:hh])[0],stats(r.iloc[hh:])[0]
            print(f'  {lbl:<32}{sh:>9.2f}{cs:>8.2f}{sh-cs:>8.2f}{s1:>8.2f}{s2:>8.2f}'
                  f'{("yes" if s1>0 and s2>0 else "NO"):>7}{n:>7}')
        print()
