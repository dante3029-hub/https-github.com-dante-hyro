"""Combinatorial purged cross-validation on the current book.

The spec's CPCV (3.10 +/- 1.13, 0 negative partitions) was run on the OLD book
-- core, pattern, w5 -- with the corrupt taker column. It validates nothing we
built since.

Method: split the series into K contiguous groups, take every combination of
`n_test` groups as a test set, PURGE the bars adjacent to each boundary (so a
45-day-hold sleeve cannot leak across), and score the blend on each partition.

What it answers: does the blend hold up on sub-periods it was not chosen on,
or does it rest on a couple of good stretches?
"""
import itertools
import numpy as np, pandas as pd


def cpcv(series_dict, K=8, n_test=2, purge=45, weights=None):
    """series_dict: name -> daily return Series, already aligned."""
    names = list(series_dict)
    idx = series_dict[names[0]].index
    n = len(idx)
    bounds = [int(round(i*n/K)) for i in range(K+1)]
    groups = [(bounds[i], bounds[i+1]) for i in range(K)]
    nz = lambda a: a/a.std() if a.std() > 0 else a
    w = weights or {k: 1.0 for k in names}
    out = []
    for combo in itertools.combinations(range(K), n_test):
        mask = np.zeros(n, bool)
        for g in combo:
            lo, hi = groups[g]
            mask[lo:hi] = True
        # purge `purge` bars either side of every boundary inside the test set
        pm = mask.copy()
        for g in combo:
            lo, hi = groups[g]
            pm[max(0, lo-purge):lo] = False
            pm[hi:min(n, hi+purge)] = False
        sub = {k: v[mask] for k, v in series_dict.items()}
        blend = sum(w[k]*nz(v) for k, v in sub.items())/sum(w.values())
        b = blend.dropna()
        if len(b) < 40 or b.std() == 0:
            continue
        out.append(float(b.mean()/b.std()*np.sqrt(365)))
    return np.array(out)


def report(arr, label):
    if len(arr) == 0:
        return f"  {label:<22}  no valid partitions"
    return (f"  {label:<22}{len(arr):>7}{arr.mean():>8.2f}{np.median(arr):>8.2f}"
            f"{arr.min():>8.2f}{arr.max():>8.2f}"
            f"{(arr > 1.0).mean()*100:>8.0f}%{(arr < 0).mean()*100:>8.0f}%")
