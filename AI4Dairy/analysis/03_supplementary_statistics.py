"""Supplementary Tables S1 to S3: equivalence and age."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "responsesdairy_public.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CATS = ["Barista Milk", "Butter", "Cheddar", "Cream Cheese", "Creamer",
        "Ice Cream", "Milk", "Mozzarella", "Sour Cream", "Yogurt"]
LIKE = {"Dislike very much": 1, "Dislike": 2, "Dislike somewhat": 3,
        "Neither like nor dislike": 4, "Like somewhat": 5, "Like": 6,
        "Like very much": 7}
MARGINS = [0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]
AGES = ["18-25", "26-35", "36-45", "46-55", "Greater than 55"]

df = pd.read_csv(DATA, low_memory=False)
age_col = [c for c in df.columns if "[MODALITY: Age]" in c][0]


def col(cat, grp):
    tag = f"{cat} Animal Benchmark" if grp == "b" else f"{cat} Plant-based Leader"
    return [c for c in df.columns if f"[RELATED_PRODUCT: {tag}]" in c
            and "[MODALITY: Overall Liking]" in c
            and "[QUESTION_TYPE: Likert]" in c][0]


# Long table of paired differences
rows = []
for cat in CATS:
    b = df[col(cat, "b")].map(LIKE)
    l = df[col(cat, "l")].map(LIKE)
    keep = b.notna() & l.notna()
    for i in df.index[keep]:
        rows.append(dict(resp=df.at[i, "Respondent"], cat=cat,
                         age=df.at[i, age_col], d=l[i] - b[i]))
long = pd.DataFrame(rows)


# Equivalence table
def tost(d, m):
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    lo = stats.t.sf((d.mean() + m) / se, n - 1)
    hi = stats.t.cdf((d.mean() - m) / se, n - 1)
    return max(lo, hi)


eq = []
for cat in CATS:
    d = long.loc[long.cat == cat, "d"].to_numpy(float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    ci90 = stats.t.interval(0.90, n - 1, loc=d.mean(), scale=se)
    r = dict(cat=cat, n=n, mean=d.mean(), lo90=ci90[0], hi90=ci90[1])
    for m in MARGINS:
        r[f"p_{m}"] = tost(d, m)
    eq.append(r)
eq = pd.DataFrame(eq)
for m in MARGINS:
    p = eq[f"p_{m}"]
    eq[f"holm_{m}"] = multipletests(p, method="holm")[1]
    eq[f"bonf_{m}"] = multipletests(p, method="bonferroni")[1]
eq = eq.sort_values("mean").reset_index(drop=True)
eq.to_csv(f"{OUT}/supp_equivalence.csv", index=False)

# Age analysis: OLS, respondent-clustered SEs
long = long[long.age.isin(AGES)].copy()
long["age"] = pd.Categorical(long.age, AGES)
g = long.resp.astype("category").cat.codes
fit_main = smf.ols("d ~ C(cat) + C(age)", long).fit(
    cov_type="cluster", cov_kwds={"groups": g})
fit_int = smf.ols("d ~ C(cat) * C(age)", long).fit(
    cov_type="cluster", cov_kwds={"groups": g})


def wald(fit, key, excl=None):
    names = [n for n in fit.params.index if key in n
             and (excl is None or excl not in n)]
    if excl is None and key == "C(age)":
        names = [n for n in names if ":" not in n]
    R = np.zeros((len(names), len(fit.params)))
    for k, nm in enumerate(names):
        R[k, list(fit.params.index).index(nm)] = 1
    w = fit.wald_test(R, use_f=True, scalar=True)
    return float(w.statistic), float(w.pvalue), len(names), float(w.df_denom)


age_main = wald(fit_main, "C(age)")
inter = wald(fit_int, ":")

# Mixed model check
mm = smf.mixedlm("d ~ C(cat) + C(age)", long, groups=long.resp).fit(reml=False)
mm0 = smf.mixedlm("d ~ C(cat)", long, groups=long.resp).fit(reml=False)
lr = 2 * (mm.llf - mm0.llf)
p_lr = stats.chi2.sf(lr, 4)
mmi = smf.mixedlm("d ~ C(cat) * C(age)", long, groups=long.resp).fit(reml=False)
lri = 2 * (mmi.llf - mm.llf)
p_lri = stats.chi2.sf(lri, 36)

# Within-category age tests (one-way ANOVA on d)
wc = []
for cat in CATS:
    s = long[long.cat == cat]
    grp = [s.loc[s.age == a, "d"].to_numpy(float) for a in AGES]
    F, p = stats.f_oneway(*grp)
    kw = stats.kruskal(*grp).pvalue
    r = dict(cat=cat, n=len(s), F=F, p=p, p_kw=kw)
    for a, x in zip(AGES, grp):
        r[f"n_{a}"] = len(x)
        r[f"m_{a}"] = x.mean() if len(x) else np.nan
    wc.append(r)
wc = pd.DataFrame(wc)
wc["p_holm"] = multipletests(wc.p, method="holm")[1]
wc.to_csv(f"{OUT}/supp_age_within.csv", index=False)

summ = dict(n_pairs=len(long), n_resp=long.resp.nunique(),
            multi=int((long.groupby("resp").size() > 1).sum()),
            ols_age=age_main, ols_int=inter,
            mm_age=(lr, p_lr), mm_int=(lri, p_lri),
            sub_min=int(wc[[f"n_{a}" for a in AGES]].min().min()),
            sub_max=int(wc[[f"n_{a}" for a in AGES]].max().max()))
json.dump(summ, open(f"{OUT}/supp_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
print(eq.round(3).to_string())
print(wc.round(3).to_string())
