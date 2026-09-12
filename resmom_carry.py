"""resmom and carry -- two v15 sleeves never rebuilt.

  resmom  beta-hedged residual momentum, 7d, vol-scaled   (spec: 1.42)
  carry   funding carry, N=8, 7d hold                     (spec: 0.92)

Carry is NOT funding z-fade. z-fade shorts extreme funding expecting mean
reversion in PRICE; carry harvests the funding payment itself -- long the coins
paying you, short the coins you pay. Different mechanism, tested separately
because z-fade failed across all 12 configs (best 0.09).
"""
import os, glob
import numpy as np, pandas as pd
TAKER='/tmp/hyro/taker_data'; FUND='/tmp/hyro/funding_data'
FEE, CAP = 0.00085, 0.15
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{TAKER}/*_1h.csv')})

def panel():
    P,F={},{}
    for s in SYMS:
        df=pd.read_csv(f'{TAKER}/{s}_1h.csv')
        df['ts']=pd.to_datetime(df['open_time'],unit='ms'); df=df.set_index('ts').sort_index()
        P[s]=df['close'].resample('1D').last()
        fp=f'{FUND}/{s}_funding.csv'
        if os.path.exists(fp):
            f2=pd.read_csv(fp); f2['ts']=pd.to_datetime(f2['ts'],unit='ms')
            F[s]=f2.set_index('ts')['rate'].resample('1D').sum()
    P={k:v for k,v in P.items() if len(v)>=400}
    idx=None
    for v in P.values(): idx = v.index if idx is None else idx.union(v.index)
    cs=sorted(P)
    PX=pd.DataFrame({c:P[c].reindex(idx) for c in cs})
    FD=pd.DataFrame({c:F[c].reindex(idx) if c in F else pd.Series(np.nan,index=idx) for c in cs})
    return idx,PX,PX.pct_change(),FD

def xs(R,sig,n=5,hold=7,shuffle=False,seed=0,long_low=False):
    rng=np.random.default_rng(seed); idx=R.index; acc=np.zeros(len(idx))
    for ph in range(hold):
        wp=pd.Series(0.0,index=R.columns); out=np.zeros(len(idx))
        for t in range(60,len(idx)):
            r=R.iloc[t].fillna(0.0); carry=float((wp*r).sum())
            if (t-ph)%hold!=0: out[t]=carry; continue
            s=sig(t); v=s.dropna()
            if len(v)<2*n+2: out[t]=carry; continue
            o=v.sort_values(); w=pd.Series(0.0,index=R.columns)
            if shuffle:
                p=list(o.index); rng.shuffle(p); w[p[-n:]]=0.5/n; w[p[:n]]=-0.5/n
            elif long_low:
                w[o.index[:n]]=0.5/n; w[o.index[-n:]]=-0.5/n
            else:
                w[o.index[-n:]]=0.5/n; w[o.index[:n]]=-0.5/n
            out[t]=float((w*r).sum())-float((w-wp).abs().sum())*FEE
            wp=w
        acc+=out
    return pd.Series(acc[60:]/hold,index=idx[60:])

def stats(x):
    x=x.dropna()
    if len(x)<50 or x.std()==0: return 0.0
    return float(x.mean()/x.std()*np.sqrt(365))

if __name__=='__main__':
    idx,PX,R,FD = panel()
    mkt=R.mean(axis=1)
    print(f'{len(R.columns)} coins, {len(idx)} days\n')
    print(f"{'  sleeve':<34}{'Sharpe':>9}{'ctrl':>8}{'GAP':>8}{'1st':>8}{'2nd':>8}{'both':>7}")
    def rep(lbl,sig,**kw):
        r=xs(R,sig,**kw); c=xs(R,sig,shuffle=True,seed=5,**kw)
        sh=stats(r); cs=stats(c); h=len(r)//2
        s1,s2=stats(r.iloc[:h]),stats(r.iloc[h:])
        print(f'  {lbl:<32}{sh:>9.2f}{cs:>8.2f}{sh-cs:>8.2f}{s1:>8.2f}{s2:>8.2f}{("yes" if s1>0 and s2>0 else "NO"):>7}')

    # RESMOM: return minus beta*market, over lookback, scaled by residual vol
    def resmom(t, lb):
        w=R.iloc[t-lb:t]; m=mkt.iloc[t-lb:t]
        if m.std()==0: return pd.Series(np.nan,index=R.columns)
        beta=w.apply(lambda col: col.cov(m)/m.var() if col.notna().sum()>5 else np.nan)
        resid=w.sub(np.outer(m,beta), axis=0)
        cum=resid.sum(); sd=resid.std()
        return cum/sd.replace(0,np.nan)
    for lb in (7,14,21):
        for hold in (7,14):
            rep(f'resmom {lb}d lookback, hold {hold}', lambda t,k=lb: resmom(t,k), hold=hold)

    # CARRY: long the coins PAYING you (negative funding), short those you pay
    print()
    for lb in (1,3,7):
        for hold in (7,14):
            rep(f'carry {lb}d funding, hold {hold}',
                lambda t,k=lb: FD.iloc[t-k:t].sum(), n=8, hold=hold, long_low=True)
    print('\n  spec: resmom 1.42 (beta-hedged 7d vol-scaled), carry 0.92 (N=8, 7d)')
