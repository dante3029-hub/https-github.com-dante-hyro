"""Rebuild the v15 sleeves from repo data (price_data + oi_data, 24 coins).

Costs locked at FEE 0.055%/side + SLIP 0.03% = 0.00085.
strat8 is NOT here -- it needs 5m bars, which live on the server. Run
strat8.py there and push strat8_returns.csv.
"""
import os, glob
import numpy as np, pandas as pd
PRICE='/tmp/hyro/price_data'
FEE, CAP, N_PER_SIDE, HOLD = 0.00085, 0.15, 5, 7
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{PRICE}/*_1h.csv')})

def load_daily(sym):
    df=pd.read_csv(f'{PRICE}/{sym}_1h.csv')
    ts='open_time' if 'open_time' in df.columns else 'timestamp'
    df['ts']=pd.to_datetime(df[ts],unit='ms',utc=True).dt.tz_localize(None)
    df=df.set_index('ts').sort_index()
    cols={'open':df['open'].resample('1D').first(),'high':df['high'].resample('1D').max(),
          'low':df['low'].resample('1D').min(),'close':df['close'].resample('1D').last(),
          'volume':df['volume'].resample('1D').sum()}
    if 'taker_buy_base' in df.columns:
        cols['taker_buy']=pd.to_numeric(df['taker_buy_base'],errors='coerce').resample('1D').sum()
    return pd.DataFrame(cols).dropna(subset=['close'])

def panel():
    frames={}
    for s in SYMS:
        try: d=load_daily(s)
        except Exception: continue
        if len(d)>=400: frames[s]=d
    idx=None
    for d in frames.values(): idx = d.index if idx is None else idx.union(d.index)
    coins=sorted(frames)
    PX=pd.DataFrame({c:frames[c]['close'].reindex(idx) for c in coins})
    VOL=pd.DataFrame({c:frames[c]['volume'].reindex(idx) for c in coins})
    DN=pd.DataFrame({c:(2*frames[c]['taker_buy']-frames[c]['volume']).reindex(idx)
                     if 'taker_buy' in frames[c] else pd.Series(np.nan,index=idx) for c in coins})
    return coins, idx, PX, PX.pct_change(), VOL, DN

def xs_sleeve(R, sig, n=N_PER_SIDE, hold=HOLD, fee=FEE, phases=None):
    phases=phases or hold; idx=R.index; acc=np.zeros(len(idx))
    for ph in range(phases):
        w_prev=pd.Series(0.0,index=R.columns); out=np.zeros(len(idx))
        for t in range(60,len(idx)):
            r=R.iloc[t].fillna(0.0); carry=float((w_prev*r).sum())
            if (t-ph)%hold!=0: out[t]=carry; continue
            s=sig(t); v=s.dropna()
            if len(v)<2*n+2: out[t]=carry; continue
            o=v.sort_values(); w=pd.Series(0.0,index=R.columns)
            w[o.index[-n:]]=0.5/n; w[o.index[:n]]=-0.5/n
            out[t]=float((w*r).sum())-float((w-w_prev).abs().sum())*fee
            w_prev=w
        acc+=out
    return pd.Series(acc[60:]/phases, index=idx[60:])

def build_all():
    coins, idx, PX, R, VOL, DN = panel()
    out={}
    def sig_delta(t):
        d=DN.iloc[t-3:t].sum(); v=VOL.iloc[t-3:t].abs().sum()
        x=d/v.replace(0,np.nan)
        return (x-x.mean())/x.std() if x.std()>0 else x*0
    if DN.notna().any().sum()>=10: out['delta']=xs_sleeve(R,sig_delta)
    def sig_relvol(t):
        x=VOL.iloc[t-1]/VOL.iloc[t-21:t-1].mean()
        return (x-x.mean())/x.std() if x.std()>0 else x*0
    out['relvol']=xs_sleeve(R,sig_relvol,n=8)
    out['skew']=xs_sleeve(R,lambda t: R.iloc[t-45:t].skew(),hold=45,phases=7)
    mkt=R.mean(axis=1)
    pos=pd.Series(0.0,index=R.columns); res=np.zeros(len(idx)); age=0
    for t in range(60,len(idx)):
        r=R.iloc[t].fillna(0.0); res[t]=float((pos*r).sum()); age+=1
        if age<3 and pos.abs().sum()>0: continue
        v=mkt.iloc[t-20:t].std(); w=pd.Series(0.0,index=R.columns)
        if v>0 and mkt.iloc[t]/v<=-2.5:
            worst=R.iloc[t].dropna().sort_values().index[:5]
            if len(worst): w[worst]=1.0/len(worst)
        res[t]-=float((w-pos).abs().sum())*FEE; pos,age=w,0
    out['cascade']=pd.Series(res[60:],index=idx[60:])
    return out, coins, idx

def stats(x):
    x=x.dropna()
    if len(x)<50 or x.std()==0: return 0.0,0.0,0.0
    eq=x.cumsum()
    return float(x.mean()/x.std()*np.sqrt(365)), float(x.mean()*365*100), float((eq.cummax()-eq).max()*100)
