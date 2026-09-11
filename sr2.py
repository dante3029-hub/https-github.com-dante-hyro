"""ChartPrime S/R volume boxes -- rebuilt from the FULL source.

MY EARLIER VERSION HAD THE VOLUME POLARITY BACKWARDS:
    source: support at a pivot LOW when Vol > vol_hi   (buying pressure)
            resistance at a pivot HIGH when Vol < vol_lo (selling pressure)
    mine:   the reverse, so boxes landed on the wrong bars entirely.
Also: pivots are on CLOSE (calcSupportResistance(close, ...)), not high/low,
and the box width is ATR(200) not ATR(14).
"""
import os, glob, sys
import numpy as np, pandas as pd
sys.path.insert(0,'/tmp/hyro')

PRICE='/tmp/hyro/price_data'; OIDIR='/tmp/hyro/oi_data'
def load(sym, tf):
    df=pd.read_csv(f'{PRICE}/{sym}_1h.csv')
    ts='open_time' if 'open_time' in df.columns else 'timestamp'
    df['ts']=pd.to_datetime(df[ts],unit='ms',utc=True).dt.tz_localize(None)
    df=df.set_index('ts').sort_index(); r=f'{tf}h'
    return pd.DataFrame({'open':df['open'].resample(r).first(),'high':df['high'].resample(r).max(),
        'low':df['low'].resample(r).min(),'close':df['close'].resample(r).last(),
        'volume':df['volume'].resample(r).sum()}).dropna()
def load_oi(sym):
    p=f'{OIDIR}/{sym}_oi_1h.csv'
    if not os.path.exists(p): return None
    df=pd.read_csv(p)
    df['ts']=pd.to_datetime(df['timestamp'],unit='ms',utc=True).dt.tz_localize(None)
    return df.set_index('ts').sort_index()['open_interest']
def stats(x):
    x=x.dropna()
    if len(x)<50 or x.std()==0: return 0.0,0.0,0.0
    eq=x.cumsum()
    return (float(x.mean()/x.std()*np.sqrt(365)), float(x.mean()*365*100), float((eq.cummax()-eq).max()*100))

FEE, CAP, MAX_HOLD, ATR_STOP = 0.00085, 0.15, 15, 2.0
TP_ATR = None   # take-profit in ATR, None = none
LOOKBACK, VOL_LEN, BOX_W = 20, 2, 1.0
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob('/tmp/hyro/price_data/*_1h.csv')})

def delta_vol(d):
    sign = np.where(d['close']>d['open'], 1.0, np.where(d['close']<d['open'], -1.0, np.nan))
    return pd.Series(sign, index=d.index).ffill().fillna(1.0) * d['volume']

def atr(d, n):
    h,l,c = d['high'],d['low'],d['close']
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def levels(d):
    c = d['close'].values; n = len(d); k = LOOKBACK
    # source: vol_hi = ta.highest(Vol/2.5, vol_len) but the test is `Vol > vol_hi`
    # -- only the THRESHOLD is scaled by 2.5, not the value being compared.
    # Dividing both made the condition unsatisfiable.
    Vol = delta_vol(d)
    vol_hi = (Vol/2.5).rolling(VOL_LEN).max().values
    vol_lo = (Vol/2.5).rolling(VOL_LEN).min().values
    V = Vol.values
    W = (atr(d,200)*BOX_W).values
    sup=np.full(n,np.nan); sup1=np.full(n,np.nan)
    res=np.full(n,np.nan); res1=np.full(n,np.nan)
    cs=cs1=cr=cr1=np.nan
    # ta.pivotlow() returns the value from k bars back, but `Vol` on the same
    # line is the CURRENT bar. So the volume test applies to the CONFIRMATION
    # bar, not the pivot. Testing it at the pivot could never fire: a pivot low
    # is a bearish candle, so its delta volume is always negative.
    for i in range(k, n-k):
        j = i + k                     # confirmation bar
        if j >= n:
            continue
        piv_lo = c[i] == c[i-k:i+k+1].min()
        piv_hi = c[i] == c[i-k:i+k+1].max()
        w = W[j] if np.isfinite(W[j]) else 0.0
        if piv_lo and np.isfinite(vol_hi[j]) and V[j] > vol_hi[j]:
            cs, cs1 = c[i], c[i]-w
        if piv_hi and np.isfinite(vol_lo[j]) and V[j] < vol_lo[j]:
            cr, cr1 = c[i], c[i]+w
        sup[j],sup1[j],res[j],res1[j] = cs,cs1,cr,cr1
    for a in (sup,sup1,res,res1):
        a[:] = pd.Series(a).ffill().values
    return sup,sup1,res,res1

def signals(d, which):
    sup,sup1,res,res1 = levels(d)
    h,l = d['high'].values, d['low'].values; n=len(d)
    def xover(a,b):
        o=np.zeros(n,bool)
        for i in range(1,n):
            if np.isfinite(b[i]) and np.isfinite(b[i-1]) and a[i]>b[i] and a[i-1]<=b[i-1]: o[i]=True
        return o
    def xunder(a,b):
        o=np.zeros(n,bool)
        for i in range(1,n):
            if np.isfinite(b[i]) and np.isfinite(b[i-1]) and a[i]<b[i] and a[i-1]>=b[i-1]: o[i]=True
        return o
    br_res = xover(l,res1); br_sup = xunder(h,sup1)
    sp_hld = xover(l,sup);  rs_hld = xunder(h,res)
    L=np.zeros(n,bool); S=np.zeros(n,bool)
    if which=='break_res': L=br_res
    elif which=='break_sup': S=br_sup
    elif which=='sup_holds': L=sp_hld
    elif which=='res_holds': S=rs_hld
    elif which=='all': L,S = br_res|sp_hld, br_sup|rs_hld
    return L,S

def run(tf, which, oi_filter=False, shuffle=False, seed=0):
    rng=np.random.default_rng(seed); by={}; ne=0
    for sym in SYMS:
        try: d=load(sym,tf)
        except Exception: continue
        if len(d)<400: continue
        L,S = signals(d,which)
        A=atr(d,14).values
        o,hi,lo,c = d['open'].values,d['high'].values,d['low'].values,d['close'].values
        n=len(d); oi_up=None
        if oi_filter:
            oi=load_oi(sym)
            if oi is None: continue
            od=oi.resample(f'{tf}h').last(); oi_up=(od-od.shift(1))>0
        for i in range(n-MAX_HOLD-2):
            side = 1 if L[i] else (-1 if S[i] else 0)
            if side==0 or not np.isfinite(A[i]) or A[i]<=0: continue
            eb=i+1
            if eb>=n: continue
            if oi_filter:
                v=oi_up.reindex([d.index[eb]],method='ffill')
                if not (len(v) and bool(v.iloc[0])): continue
            if shuffle: side=int(rng.choice([-1,1]))
            ent=o[eb]; stop=ent-side*ATR_STOP*A[i]; r=None
            tp = None if TP_ATR is None else ent + side*TP_ATR*A[eb]
            for j in range(eb+1,min(eb+1+MAX_HOLD,n)):
                if side>0 and lo[j]<=stop: r=(stop-ent)/ent-2*FEE; break
                if side<0 and hi[j]>=stop: r=(ent-stop)/ent-2*FEE; break
                if tp is not None:
                    if side>0 and hi[j]>=tp: r=(tp-ent)/ent-2*FEE; break
                    if side<0 and lo[j]<=tp: r=(ent-tp)/ent-2*FEE; break
            if r is None:
                j=min(eb+MAX_HOLD,n-1); r=side*(c[j]-ent)/ent-2*FEE
            if not np.isfinite(r): continue
            ne+=1; by.setdefault(d.index[eb].date(),[]).append(r)
    if not by: return None,0
    cal=pd.date_range(min(by),max(by),freq='D')
    ser=[sum(x*min(1/len(v),CAP) for x in v) if (v:=by.get(dd.date(),[])) else 0.0 for dd in cal]
    return pd.Series(ser,index=cal), ne

if __name__=='__main__':
    print('S/R VOLUME BOXES — CORRECTED POLARITY, pivots on close, ATR(200) width\n')
    print(f"{'  config':<30}{'Sharpe':>8}{'ctrl':>7}{'GAP':>7}{'ann%':>8}{'1st':>7}{'2nd':>7}{'both':>6}{'n':>7}")
    for tf in (4,6):
        for w in ('break_res','break_sup','sup_holds','res_holds','all'):
            for oi in (False,True):
                r,n = run(tf,w,oi_filter=oi)
                if r is None or n<40: continue
                c,_ = run(tf,w,oi_filter=oi,shuffle=True,seed=3)
                sh,ann,dd = stats(r); csh = stats(c)[0] if c is not None else 0.0
                hh=len(r)//2; s1,s2 = stats(r.iloc[:hh])[0], stats(r.iloc[hh:])[0]
                lbl=f"{tf}h {w}" + (" +OI" if oi else "")
                print(f'  {lbl:<28}{sh:>8.2f}{csh:>7.2f}{sh-csh:>7.2f}{ann:>7.1f}%{s1:>7.2f}{s2:>7.2f}{("yes" if s1>0 and s2>0 else "NO"):>6}{n:>7}')
        print()
