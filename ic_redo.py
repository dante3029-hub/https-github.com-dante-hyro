"""Signal half-life on MATCHED-VENUE delta.

The spec's IC table (1d 0.0276 / 3d 0.0527 / 7d 0.0463 / 14d 0.0640 / 21d 0.0678)
was computed on clean_panel's taker column -- the one where taker volume
exceeded total volume on 60% of bars. Whatever it measured, it was not order
flow, so the conclusion "the edge strengthens to 21 days" needs redoing.

IC = cross-sectional Spearman correlation of the signal with forward return,
averaged over days, with a t-stat on the daily series.
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats as sps
TAKER='/tmp/hyro/taker_data'
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{TAKER}/*_1h.csv')})

def panel():
    F={}
    for s in SYMS:
        df=pd.read_csv(f'{TAKER}/{s}_1h.csv')
        df['ts']=pd.to_datetime(df['open_time'],unit='ms'); df=df.set_index('ts').sort_index()
        d=pd.DataFrame({'close':df['close'].resample('1D').last(),
                        'volume':df['volume'].resample('1D').sum(),
                        'delta':df['delta'].resample('1D').sum()}).dropna()
        if len(d)>=400: F[s]=d
    idx=None
    for v in F.values(): idx = v.index if idx is None else idx.union(v.index)
    cs=sorted(F)
    PX=pd.DataFrame({c:F[c]['close'].reindex(idx) for c in cs})
    VOL=pd.DataFrame({c:F[c]['volume'].reindex(idx) for c in cs})
    DN=pd.DataFrame({c:F[c]['delta'].reindex(idx) for c in cs})
    return idx,PX,PX.pct_change(),VOL,DN

if __name__=='__main__':
    idx,PX,R,VOL,DN = panel()
    print(f'{len(R.columns)} coins, {len(idx)} days, matched-venue delta\n')
    print(f"{'  horizon':<14}{'mean IC':>11}{'t-stat':>10}{'days':>8}")
    sig_all = (DN.rolling(3).sum()/VOL.rolling(3).sum().replace(0,np.nan))
    for hz in (1,3,5,7,14,21,30):
        fwd = PX.shift(-hz)/PX - 1
        ics=[]
        for t in range(60, len(idx)-hz):
            s = sig_all.iloc[t].dropna()
            f = fwd.iloc[t].dropna()
            common = s.index.intersection(f.index)
            if len(common) < 8: continue
            rho = sps.spearmanr(s[common], f[common]).correlation
            if np.isfinite(rho): ics.append(rho)
        a=np.array(ics)
        t_stat = a.mean()/a.std()*np.sqrt(len(a)) if a.std()>0 else 0
        print(f'  {str(hz)+"d":<12}{a.mean():>11.4f}{t_stat:>10.2f}{len(a):>8}')
    print('\n  spec (corrupt column): 1d 0.0276 t3.24 / 3d 0.0527 t6.30')
    print('                         7d 0.0463 t5.99 / 14d 0.0640 t8.68 / 21d 0.0678 t8.72')
