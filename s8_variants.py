"""strat8 variants -- can I reproduce the spec's 1.51?

My rebuild: 349 trades, 0.80 Sharpe.
Spec:       368 trades, 1.24-1.51.
Trade count nearly matches, so detection is right and the difference is in the
exit or the weighting. Spec claims:
    with exhaustion wait      1.51  (76% win)
    without                   1.17  (67% win)
    with confirmation entry   1.42  (removing it IMPROVES to 1.51)
Runs on the SERVER -- the 5m data is 489MB and lives there.
"""
import os, csv, glob, sys
import datetime as dt
import numpy as np

M5 = os.path.expanduser("~/bot_hyrotrader_v1/m5")
FEE = 0.00085

def load(p):
    ts=[];o=[];h=[];l=[];c=[];v=[]
    with open(p) as f:
        r=csv.reader(f); next(r,None)
        for row in r:
            try:
                ts.append(int(row[1])); o.append(float(row[2])); h.append(float(row[3]))
                l.append(float(row[4])); c.append(float(row[5])); v.append(float(row[6]))
            except (ValueError,IndexError): continue
    if len(c)<20000: return None
    return (np.array(ts),np.array(o),np.array(h),np.array(l),np.array(c),np.array(v))

def run_coin(path, wait=True, confirm=False, retrace=0.50, max_hold=144, cap=0.15):
    d=load(path)
    if d is None: return []
    ts,o,h,l,c,v=d; n=len(c)
    vma=np.full(n,np.nan)
    for i in range(288,n): vma[i]=v[i-288:i].mean()
    out=[]; i=300
    while i<n-max_hold-3:
        if not np.isfinite(vma[i]) or vma[i]<=0: i+=1; continue
        wick=(c[i]-l[i])/max(c[i],1e-9)
        if not (v[i]>4.0*vma[i] and wick>=0.05): i+=1; continue
        j=i; g=0
        if wait:
            while j<n-1 and g<12 and l[j+1]<l[j]: j+=1; g+=1
        ei=j+1
        if confirm:
            k=ei
            while k<n-1 and k<ei+12 and not (h[k]>h[k-1]): k+=1
            ei=k+1
        if ei>=n-max_hold: i=j+1; continue
        low=l[i:j+1].min(); hi=h[max(0,i-12):i+1].max()
        ent=o[ei]; tgt=low+retrace*(hi-low); stop=low*0.999
        if tgt<=ent or stop>=ent: i=j+1; continue
        px=None
        for k in range(ei,min(ei+max_hold,n)):
            if l[k]<=stop: px=stop; break
            if h[k]>=tgt: px=tgt; break
        if px is None: px=c[min(ei+max_hold,n-1)]
        out.append((ts[min(ei+max_hold,n-1)], (px-ent)/ent-2*FEE))
        i=ei+max_hold
    return out

def series(tr, cap=0.15):
    by={}
    for exts,r in tr:
        day=dt.datetime.fromtimestamp(exts/1000,tz=dt.timezone.utc).date()
        by.setdefault(day,[]).append(r)
    lo,hi=min(by),max(by); rows=[]; day=lo
    while day<=hi:
        vv=by.get(day,[])
        rows.append(sum(x*min(1/len(vv),cap) for x in vv) if vv else 0.0)
        day+=dt.timedelta(days=1)
    return np.array(rows)

def stats(a):
    if a.std()==0: return 0,0,0
    h=len(a)//2
    f=lambda x: x.mean()/x.std()*np.sqrt(365) if x.std()>0 else 0
    return f(a), f(a[:h]), f(a[h:])

if __name__=='__main__':
    files=sorted(glob.glob(f"{M5}/*_5m.csv"))
    print(f"{'  variant':<34}{'Sharpe':>9}{'1st':>8}{'2nd':>8}{'win%':>7}{'n':>7}")
    for lbl,kw in (('exhaustion wait (spec: 1.51)',{}),
                   ('no wait (spec: 1.17)',{'wait':False}),
                   ('wait + confirm (spec: 1.42)',{'confirm':True}),
                   ('retrace 0.382',{'retrace':0.382}),
                   ('retrace 0.618',{'retrace':0.618}),
                   ('max_hold 72',{'max_hold':72}),
                   ('cap 1.0 (no per-day cap)',{'cap':1.0})):
        tr=[]
        capv=kw.pop('cap',0.15)
        for p in files: tr+=run_coin(p,**kw)
        if len(tr)<50: print(f'  {lbl:<32}  too few ({len(tr)})'); continue
        a=series(tr,capv); sh,s1,s2=stats(a)
        wins=sum(1 for _,r in tr if r>0)/len(tr)*100
        print(f'  {lbl:<32}{sh:>9.2f}{s1:>8.2f}{s2:>8.2f}{wins:>6.0f}%{len(tr):>7}')
