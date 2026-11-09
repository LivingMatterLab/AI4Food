"""Supplementary Table S4: CATA penalty sensitivity."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
spec = importlib.util.spec_from_file_location("S", HERE / "01_primary_statistics.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
df = pd.read_csv(S.DATA, low_memory=False)
rng = np.random.default_rng(20261102)

# Long table: one row per leader rating and term
recs = []
for cat in S.CATEGORIES:
    lik = df[S.cols_for(df, cat, "leader", "Overall Liking")[0]].map(S.LIKE).astype(float)
    for mod in ["Flavor", "Texture Mouthfeel", "Appearance"]:
        lc = S.cols_for(df, cat, "leader", mod, "CATA")
        if not lc:
            continue
        sets = S.parse_cata(df[lc[0]])
        idx = [i for i in range(len(df)) if sets[i] is not None and not np.isnan(lik.iloc[i])]
        terms = sorted({t for i in idx for t in sets[i]})
        for term in terms:
            sel = np.array([term in sets[i] for i in idx])
            if sel.sum() < 10 or sel.sum() == len(sel):
                continue
            for i, s in zip(idx, sel):
                recs.append((df.at[i, "Respondent"], cat, mod, term, int(s), lik.iloc[i]))
L = pd.DataFrame(recs, columns=["resp", "cat", "mod", "term", "sel", "y"])
resp_ids = L.resp.unique()
rid = {r: k for k, r in enumerate(resp_ids)}
L["rid"] = L.resp.map(rid)


rows = []
for (term, mod), g in L.groupby(["term", "mod"]):
    if g.sel.sum() < 25:
        continue
    g = g.reset_index(drop=True)
    cats = g.cat.to_numpy()
    sel = g.sel.to_numpy().astype(bool)
    y = g.y.to_numpy()
    r = g.rid.to_numpy()
    ucat = np.unique(cats)

    def compute(wt):
        raw = (np.sum(wt * y * ~sel) / np.sum(wt * ~sel)
               - np.sum(wt * y * sel) / np.sum(wt * sel))
        num = den = 0.0
        pc = []
        for c in ucat:
            m = cats == c
            a, b = np.sum(wt * (m & sel)), np.sum(wt * (m & ~sel))
            if a == 0 or b == 0:
                continue
            p = np.sum(wt * y * (m & ~sel)) / b - np.sum(wt * y * (m & sel)) / a
            h = a * b / (a + b)
            num += h * p
            den += h
            pc.append(p)
        return raw, num / den, pc

    raw, adj, pc = compute(np.ones(len(y)))
    # Respondent-clustered bootstrap via multinomial weights
    nr = len(resp_ids)
    braw, badj = np.empty(NB), np.empty(NB)
    for b in range(NB):
        cnt = np.bincount(rng.integers(0, nr, nr), minlength=nr)
        wt = cnt[r].astype(float)
        try:
            braw[b], badj[b], _ = compute(wt)
        except ZeroDivisionError:
            braw[b] = badj[b] = np.nan
    lo_r, hi_r = np.nanpercentile(braw, [2.5, 97.5])
    lo_a, hi_a = np.nanpercentile(badj, [2.5, 97.5])
    rows.append(dict(term=term, modality=mod, n_selected=int(sel.sum()),
                     n_categories=len(pc), pooled=raw, pooled_lo=lo_r, pooled_hi=hi_r,
                     adjusted=adj, adj_lo=lo_a, adj_hi=hi_a,
                     cat_min=min(pc), cat_max=max(pc),
                     n_same_sign=int(sum(np.sign(p) == np.sign(adj) for p in pc))))

R = pd.DataFrame(rows).sort_values("adjusted", ascending=False)
R.to_csv(f"{OUT}/cata_pooled_clustered.csv", index=False)

# Compare with the original analysis
orig = pd.read_csv(f"{OUT}/rev_cata_penalty_pooled.csv")
M = R.merge(orig[["term", "modality", "penalty", "ci_lo", "ci_hi"]],
            on=["term", "modality"], how="left")
M["orig_excl0"] = (M.ci_lo > 0) | (M.ci_hi < 0)
M["clus_excl0"] = (M.pooled_lo > 0) | (M.pooled_hi < 0)
M["adj_excl0"] = (M.adj_lo > 0) | (M.adj_hi < 0)
pd.set_option("display.width", 250)
print(M[["term", "modality", "n_selected", "n_categories", "penalty", "ci_lo", "ci_hi",
         "pooled", "pooled_lo", "pooled_hi", "adjusted", "adj_lo", "adj_hi",
         "cat_min", "cat_max", "n_same_sign", "orig_excl0", "clus_excl0", "adj_excl0"]]
      .round(2).to_string())
M.to_csv(f"{OUT}/cata_pooled_comparison.csv", index=False)
