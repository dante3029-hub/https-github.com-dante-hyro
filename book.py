"""The book on ONE fixed window, with matched-venue taker data.

Every sleeve rebuilt over the same dates so the numbers are comparable.
Previous figures drifted (relvol read 1.16 / 1.18 / 1.36) purely because each
test intersected a different set of dates.

Costs locked: FEE 0.055%/side + SLIP 0.03% = 0.00085.
"""
import os, glob
import numpy as np, pandas as pd
TAKER='/tmp/hyro/taker_data'
FEE, CAP = 0.00085, 0.15
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{TAKER}/*_1h.csv')})

def load_daily(sym):
    df=pd.read_csv(f'{TAKER}/{sym}_1h.csv')
    df['ts']=pd.to_datetime(df['open_time'],unit='ms'); df=df.set_index('ts').sort_index()
    return pd.DataFrame({'close':df['close'].resample('1D').last(),
                         'volume':df['volume'].resample('1D').sum(),
                         'delta':df['delta'].resample('1D').sum()}).dropna()

def panel():
    fr={}
    for s in SYMS:
        try: d=load_daily(s)
        except Exception: continue
        if len(d)>=400: fr[s]=d
    idx=None
    for d in fr.values(): idx = d.index if idx is None else idx.union(d.index)
    cs=sorted(fr)
    PX=pd.DataFrame({c:fr[c]['close'].reindex(idx) for c in cs})
    VOL=pd.DataFrame({c:fr[c]['volume'].reindex(idx) for c in cs})
    DN=pd.DataFrame({c:fr[c]['delta'].reindex(idx) for c in cs})
    return idx,PX,PX.pct_change(),VOL,DN

def xs(R,sig,n=5,hold=7,shuffle=False,seed=0):
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
            else:
                w[o.index[-n:]]=0.5/n; w[o.index[:n]]=-0.5/n
            out[t]=float((w*r).sum())-float((w-wp).abs().sum())*FEE
            wp=w
        acc+=out
    return pd.Series(acc[60:]/hold,index=idx[60:])

def stats(x):
    x=x.dropna()
    if len(x)<50 or x.std()==0: return 0.0,0.0
    return float(x.mean()/x.std()*np.sqrt(365)), float(x.mean()*365*100)

def build():
    idx,PX,R,VOL,DN = panel()
    z=lambda x:(x-x.mean())/x.std() if x.std()>0 else x*0
    out={}
    out['delta']  = xs(R, lambda t: z(DN.iloc[t-3:t].sum()/VOL.iloc[t-3:t].sum().replace(0,np.nan)), hold=5)
    out['relvol'] = xs(R, lambda t: z(VOL.iloc[t-1]/VOL.iloc[t-21:t-1].mean()), n=8)
    out['skew']   = xs(R, lambda t: R.iloc[t-45:t].skew(), hold=45)
    mkt=R.mean(axis=1); pos=pd.Series(0.0,index=R.columns); res=np.zeros(len(idx)); age=0
    for t in range(60,len(idx)):
        r=R.iloc[t].fillna(0.0); res[t]=float((pos*r).sum()); age+=1
        if age<3 and pos.abs().sum()>0: continue
        v=mkt.iloc[t-20:t].std(); w=pd.Series(0.0,index=R.columns)
        if v>0 and mkt.iloc[t]/v<=-2.5:
            worst=R.iloc[t].dropna().sort_values().index[:5]
            if len(worst): w[worst]=1.0/len(worst)
        res[t]-=float((w-pos).abs().sum())*FEE; pos,age=w,0
    out['cascade']=pd.Series(res[60:],index=idx[60:])
    return out, R, VOL, DN

if __name__=='__main__':
    sl,R,VOL,DN = build()
    import sr2
    sl['sr'],_   = sr2.run(6,'break_res',oi_filter=True)
    idx=None
    for v in sl.values(): idx = v.index if idx is None else idx.intersection(v.index)
    S={k:v.reindex(idx).fillna(0.0) for k,v in sl.items()}
    print(f'FIXED WINDOW: {len(idx)} days  ({idx[0].date()} -> {idx[-1].date()})')
    print(f'{len(R.columns)} coins, matched-venue taker data\n')
    print(f"{'  sleeve':<12}{'Sharpe':>9}{'ann%':>8}{'1st':>8}{'2nd':>8}{'both':>7}")
    for k,v in S.items():
        h=len(v)//2; s1,s2=stats(v.iloc[:h])[0],stats(v.iloc[h:])[0]
        sh,ann=stats(v)
        print(f'  {k:<10}{sh:>9.2f}{ann:>7.1f}%{s1:>8.2f}{s2:>8.2f}{("yes" if s1>0 and s2>0 else "NO"):>7}')
    nz=lambda a:a/a.std() if a.std()>0 else a
    print()
    ks=list(S)
    print('CORRELATIONS')
    print('           '+''.join(f'{k[:7]:>9}' for k in ks))
    for a in ks:
        print(f'  {a:<9}'+''.join(f'{np.corrcoef(S[a],S[b])[0,1]:>9.2f}' for b in ks))
    print()
    print(f"{'  blend':<26}{'Sharpe':>9}{'ann%':>8}{'1st':>8}{'2nd':>8}{'both':>7}")
    def rep(l,x):
        h=len(x)//2; s1,s2=stats(x.iloc[:h])[0],stats(x.iloc[h:])[0]
        sh,ann=stats(x)
        print(f'  {l:<24}{sh:>9.2f}{ann:>7.1f}%{s1:>8.2f}{s2:>8.2f}{("yes" if s1>0 and s2>0 else "NO"):>7}')
    rep('all 5 equal', sum(nz(v) for v in S.values())/len(S))
    four={k:S[k] for k in ('delta','relvol','cascade','sr')}
    rep('4 (no skew)', sum(nz(v) for v in four.values())/4)
    three={k:S[k] for k in ('delta','relvol','sr')}
    rep('delta+relvol+sr', sum(nz(v) for v in three.values())/3)
