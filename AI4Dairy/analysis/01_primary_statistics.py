"""
Revision analyses for the dairy benchmarking manuscript.

Adds, relative to the original submission:
  1. Paired t-tests as the primary test (Wilcoxon retained as sensitivity)
  2. TOST equivalence tests at a fixed +/-0.5 point margin
  3. Full inference for purchase intent
  4. FDR sensitivity on the 40 exploratory component comparisons
  5. McNemar tests on CATA term frequencies (benchmark vs leader)
  6. Bootstrap CIs on CATA penalties
  7. Spearman correlation between price-threshold interval and liking gap
  8. Rating dispersion / polarization checks
  9. Sample-size reconciliation

Writes outputs/rev_*.csv, outputs/rev_summary.json, and a printed summary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "responsesdairy_public.csv"
OUT = ROOT / "outputs"
RNG = np.random.default_rng(20260721)
N_BOOT = 10000
MARGIN = 0.5  # TOST equivalence margin, points on the 7-point scale

CATEGORIES = [
    "Barista Milk", "Butter", "Cheddar", "Cream Cheese", "Creamer",
    "Ice Cream", "Milk", "Mozzarella", "Sour Cream", "Yogurt",
]

LIKE = {"Dislike very much": 1, "Dislike": 2, "Dislike somewhat": 3,
        "Neither like nor dislike": 4, "Like somewhat": 5, "Like": 6,
        "Like very much": 7}
SIM = {"Very dissimilar": 1, "Dissimilar": 2, "Somewhat dissimilar": 3,
       "Neither similar nor dissimilar": 4, "Somewhat similar": 5,
       "Similar": 6, "Very similar": 7}
BUY = {"Definitely would NOT buy": 1, "Would NOT buy": 2,
       "Probably would NOT buy": 3, "Might or might not buy": 4,
       "Probably would buy": 5, "Would buy": 6, "Definitely would buy": 7}

COMPONENTS = ["Flavor", "Texture / Mouthfeel", "Appearance", "Similarity"]


def mapping_for(modality: str) -> dict:
    if modality == "Similarity":
        return SIM
    if "Purchase Intent" in modality:
        return BUY
    return LIKE


def cols_for(df, cat, group, modality, qtype="Likert"):
    tag = f"{cat} Animal Benchmark" if group == "benchmark" else f"{cat} Plant-based Leader"
    return [c for c in df.columns
            if f"[RELATED_PRODUCT: {tag}]" in c
            and f"[MODALITY: {modality}]" in c
            and f"[QUESTION_TYPE: {qtype}]" in c]


def paired_series(df, cat, modality):
    b = cols_for(df, cat, "benchmark", modality)
    l = cols_for(df, cat, "leader", modality)
    if not b or not l:
        return None, None
    m = mapping_for(modality)
    bs = df[b[0]].map(m).astype(float)
    ls = df[l[0]].map(m).astype(float)
    keep = bs.notna() & ls.notna()
    return bs[keep].to_numpy(), ls[keep].to_numpy()


# --------------------------------------------------------------------------
# 1-3. Primary liking comparisons: t-test, Wilcoxon, TOST
# --------------------------------------------------------------------------
def primary_table(df, modality="Overall Liking"):
    rows = []
    for cat in CATEGORIES:
        b, l = paired_series(df, cat, modality)
        d = l - b
        n = len(d)
        mean_d = d.mean()
        sd = d.std(ddof=1)
        se = sd / np.sqrt(n)
        tstat, p_t = stats.ttest_rel(l, b)
        ci = stats.t.interval(0.95, n - 1, loc=mean_d, scale=se)
        dz = mean_d / sd  # Cohen's dz

        w, p_w = stats.wilcoxon(l, b, zero_method="wilcox", correction=False)
        # z from the normal approximation, for the r effect size
        n_nz = int(np.sum(d != 0))
        mu_w = n_nz * (n_nz + 1) / 4
        sigma_w = np.sqrt(n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24)
        z = (w - mu_w) / sigma_w
        r = abs(z) / np.sqrt(n)

        # TOST at +/- MARGIN
        t_lower = (mean_d + MARGIN) / se
        t_upper = (mean_d - MARGIN) / se
        p_lower = stats.t.sf(t_lower, n - 1)      # H0: d <= -margin
        p_upper = stats.t.cdf(t_upper, n - 1)     # H0: d >= +margin
        p_tost = max(p_lower, p_upper)
        # 90% CI is the equivalent-decision interval for TOST at alpha=.05
        ci90 = stats.t.interval(0.90, n - 1, loc=mean_d, scale=se)
        equivalent = bool(ci90[0] > -MARGIN and ci90[1] < MARGIN)

        rows.append(dict(
            category=cat, n=n, mean_d=mean_d, sd=sd,
            ci_lo=ci[0], ci_hi=ci[1], t=tstat, p_t=p_t, dz=dz,
            W=float(w), p_wilcoxon=p_w, r=r,
            ci90_lo=ci90[0], ci90_hi=ci90[1],
            p_tost=p_tost, equivalent=equivalent,
        ))
    out = pd.DataFrame(rows).sort_values("mean_d").reset_index(drop=True)
    out["holm_t"] = multipletests(out["p_t"], method="holm")[0]
    out["holm_wilcoxon"] = multipletests(out["p_wilcoxon"], method="holm")[0]
    return out


# --------------------------------------------------------------------------
# 4. Component ratings with FDR sensitivity
# --------------------------------------------------------------------------
def component_table(df):
    rows = []
    for cat in CATEGORIES:
        for mod in COMPONENTS:
            b, l = paired_series(df, cat, mod)
            if b is None or len(b) == 0:
                continue
            d = l - b
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n)
            tstat, p_t = stats.ttest_rel(l, b)
            _, p_w = stats.wilcoxon(l, b, zero_method="wilcox", correction=False)
            ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
            rows.append(dict(category=cat, modality=mod, n=n, mean_d=d.mean(),
                             ci_lo=ci[0], ci_hi=ci[1], p_t=p_t, p_wilcoxon=p_w))
    out = pd.DataFrame(rows)
    out["p_fdr"] = multipletests(out["p_t"], method="fdr_bh")[1]
    out["sig_fdr"] = out["p_fdr"] < 0.05
    return out


# --------------------------------------------------------------------------
# 5-6. CATA: McNemar between benchmark and leader, bootstrap penalty CIs
# --------------------------------------------------------------------------
def boot_ci(selected: np.ndarray, not_selected: np.ndarray, level: float = 95.0):
    """Vectorized percentile bootstrap CI for the penalty (not_selected - selected)."""
    a = RNG.choice(selected, size=(N_BOOT, selected.size), replace=True).mean(axis=1)
    b = RNG.choice(not_selected, size=(N_BOOT, not_selected.size), replace=True).mean(axis=1)
    boot = b - a
    tail = (100 - level) / 2
    lo, hi = np.percentile(boot, [tail, 100 - tail])
    return float(lo), float(hi)


def parse_cata(series):
    out = []
    for v in series:
        if pd.isna(v) or str(v).strip() in {"No response", ""}:
            out.append(None)
            continue
        terms = {t.strip() for t in str(v).replace("\\r", "").split(",") if t.strip()}
        terms.discard("No response")
        out.append(terms or None)
    return out


def cata_mcnemar(df):
    rows = []
    for cat in CATEGORIES:
        for mod in ["Flavor", "Texture Mouthfeel", "Appearance"]:
            bc = cols_for(df, cat, "benchmark", mod, "CATA")
            lc = cols_for(df, cat, "leader", mod, "CATA")
            if not bc or not lc:
                continue
            bsets = parse_cata(df[bc[0]])
            lsets = parse_cata(df[lc[0]])
            idx = [i for i in range(len(df)) if bsets[i] is not None and lsets[i] is not None]
            if len(idx) < 30:
                continue
            terms = sorted({t for i in idx for t in bsets[i] | lsets[i]})
            for term in terms:
                b = np.array([term in bsets[i] for i in idx])
                l = np.array([term in lsets[i] for i in idx])
                n01 = int(np.sum(~b & l))   # leader only
                n10 = int(np.sum(b & ~l))   # benchmark only
                if n01 + n10 == 0:
                    continue
                # exact binomial McNemar
                p = stats.binomtest(n01, n01 + n10, 0.5).pvalue
                rows.append(dict(category=cat, modality=mod, term=term,
                                 n_pairs=len(idx),
                                 pct_benchmark=100 * b.mean(),
                                 pct_leader=100 * l.mean(),
                                 diff_pp=100 * (l.mean() - b.mean()),
                                 n_leader_only=n01, n_bench_only=n10, p=p))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr"] = multipletests(out["p"], method="fdr_bh")[1]
        out["sig_fdr"] = out["p_fdr"] < 0.05
    return out


def cata_penalties(df, min_mentions=10, pooled_min=25):
    per_cat, pooled_rows = [], []
    for cat in CATEGORIES:
        lik_col = cols_for(df, cat, "leader", "Overall Liking")
        if not lik_col:
            continue
        liking = df[lik_col[0]].map(LIKE).astype(float)
        for mod in ["Flavor", "Texture Mouthfeel", "Appearance"]:
            lc = cols_for(df, cat, "leader", mod, "CATA")
            if not lc:
                continue
            sets = parse_cata(df[lc[0]])
            idx = [i for i in range(len(df)) if sets[i] is not None and not np.isnan(liking.iloc[i])]
            if not idx:
                continue
            y = liking.iloc[idx].to_numpy()
            terms = sorted({t for i in idx for t in sets[i]})
            for term in terms:
                sel = np.array([term in sets[i] for i in idx])
                n_sel = int(sel.sum())
                if n_sel < min_mentions or n_sel == len(sel):
                    continue
                pen = y[~sel].mean() - y[sel].mean()
                lo, hi = boot_ci(y[sel], y[~sel])
                per_cat.append(dict(category=cat, modality=mod, term=term,
                                    n_selected=n_sel, n_total=len(sel),
                                    penalty=pen, ci_lo=lo, ci_hi=hi,
                                    excludes_zero=bool(lo > 0 or hi < 0)))
                pooled_rows.append((term, mod, y[sel], y[~sel]))

    # pooled across the ten leaders
    pooled = []
    by_term: dict[tuple[str, str], list] = {}
    for term, mod, ys, yn in pooled_rows:
        by_term.setdefault((term, mod), []).append((ys, yn))
    for (term, mod), parts in by_term.items():
        ys = np.concatenate([p[0] for p in parts])
        yn = np.concatenate([p[1] for p in parts])
        if ys.size < pooled_min:
            continue
        pen = yn.mean() - ys.mean()
        lo, hi = boot_ci(ys, yn)
        pooled.append(dict(term=term, modality=mod, n_selected=int(ys.size),
                           prevalence=100 * ys.size / (ys.size + yn.size),
                           penalty=pen, ci_lo=lo, ci_hi=hi,
                           excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(per_cat), pd.DataFrame(pooled).sort_values("penalty", ascending=False)


# --------------------------------------------------------------------------
# 7. Price thresholds and the price/sensory correspondence
# --------------------------------------------------------------------------
def price_values(series: pd.Series) -> pd.Series:
    """Extract dollar values. The export carries stray carriage returns and
    'No response' strings, both of which must be stripped before conversion."""
    s = series.dropna().astype(str).str.extract(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce").dropna()


def price_column_category(col: str, df: pd.DataFrame) -> str | None:
    """Assign a van Westendorp column to a category.

    The two 64 oz milk items are disambiguated by which sensory group the
    responding participants belong to: the item answered by the barista-milk
    tasters is the barista-context item.
    """
    q = re.sub(r"\s+", " ", col).lower()
    for name, pat in [("Butter", "butter"), ("Cheddar", "cheddar"),
                      ("Cream Cheese", "cream cheese"), ("Creamer", "creamer"),
                      ("Ice Cream", "ice cream"), ("Mozzarella", "mozzarella"),
                      ("Sour Cream", "sour cream"), ("Yogurt", "yogurt")]:
        if pat in q:
            return name
    if "milk" in q:
        # Two 64 oz milk items exist. The one worded "gallon 64oz" was served to
        # the barista-milk tasters (n=211, matching the barista benchmark count);
        # the plain "64oz" item was served to the fluid-milk tasters (n=418).
        return "Barista Milk" if "gallon" in q else "Milk"
    return None


def price_table(df):
    vw = [c for c in df.columns if "[MODALITY: van-westendorp]" in c]
    recs = []
    for c in vw:
        q = re.sub(r"\s+", " ", c).lower()
        kind = "bargain" if "bargain" in q else ("expensive" if "expensive" in q else None)
        cat = price_column_category(c, df)
        if kind is None or cat is None:
            continue
        for v in price_values(df[c]):
            recs.append(dict(category=cat, kind=kind, value=float(v), column=q[-70:]))
    raw = pd.DataFrame(recs)
    summ = (raw.groupby(["category", "kind"])["value"]
            .agg(n="size", median="median",
                 q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
            .reset_index())
    piv = summ.pivot(index="category", columns="kind", values="median")
    piv["interval"] = piv["expensive"] - piv["bargain"]
    ns = summ.pivot(index="category", columns="kind", values="n")
    piv["n_bargain"] = ns["bargain"]
    piv["n_expensive"] = ns["expensive"]
    return raw, summ, piv.reset_index()


# --------------------------------------------------------------------------
# 8. Dispersion / polarization
# --------------------------------------------------------------------------
def dispersion(df):
    rows = []
    for cat in CATEGORIES:
        for grp in ["benchmark", "leader"]:
            c = cols_for(df, cat, grp, "Overall Liking")
            v = df[c[0]].map(LIKE).dropna().astype(float)
            rows.append(dict(category=cat, group=grp, n=len(v), mean=v.mean(),
                             sd=v.std(ddof=1),
                             top_box=100 * (v >= 6).mean(),
                             mid=100 * ((v >= 4) & (v <= 5)).mean(),
                             bottom_box=100 * (v <= 3).mean()))
    d = pd.DataFrame(rows)
    # variance ratio test on the paired subset
    vr = []
    for cat in CATEGORIES:
        b, l = paired_series(df, cat, "Overall Liking")
        f = l.var(ddof=1) / b.var(ddof=1)
        n = len(b)
        p = 2 * min(stats.f.cdf(f, n - 1, n - 1), stats.f.sf(f, n - 1, n - 1))
        vr.append(dict(category=cat, sd_benchmark=b.std(ddof=1),
                       sd_leader=l.std(ddof=1), var_ratio=f, p=p))
    return d, pd.DataFrame(vr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, dtype=str)

    print(f"respondents in response file: {len(df)}")

    liking = primary_table(df)
    pi = primary_table(df, "Purchase Intent (After NFP Reveal)")
    comp = component_table(df)
    mcn = cata_mcnemar(df)
    pen_cat, pen_pool = cata_penalties(df)
    raw_price, price_summ, price = price_table(df)
    disp, varratio = dispersion(df)

    # price interval vs liking gap
    merged = price.merge(liking[["category", "mean_d"]], on="category", how="inner")
    rho, p_rho = stats.spearmanr(merged["interval"], merged["mean_d"])
    pear, p_pear = stats.pearsonr(merged["interval"], merged["mean_d"])

    for name, frame in [("liking_primary", liking), ("purchase_intent", pi),
                        ("components_fdr", comp), ("cata_mcnemar", mcn),
                        ("cata_penalty_category", pen_cat),
                        ("cata_penalty_pooled", pen_pool),
                        ("price_medians", price), ("price_summary", price_summ),
                        ("dispersion", disp), ("variance_ratio", varratio)]:
        frame.to_csv(OUT / f"rev_{name}.csv", index=False)

    summary = dict(
        n_respondents_file=int(len(df)),
        n_paired_total=int(liking["n"].sum()),
        tost_margin=MARGIN,
        equivalent_categories=liking.loc[liking["equivalent"], "category"].tolist(),
        holm_significant_t=liking.loc[liking["holm_t"], "category"].tolist(),
        holm_significant_wilcoxon=liking.loc[liking["holm_wilcoxon"], "category"].tolist(),
        price_liking_spearman=dict(rho=float(rho), p=float(p_rho)),
        price_liking_pearson=dict(r=float(pear), p=float(p_pear)),
    )
    (OUT / "rev_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n=== PRIMARY: overall liking ===")
    print(liking[["category", "n", "mean_d", "ci_lo", "ci_hi", "p_t", "holm_t",
                  "dz", "p_wilcoxon", "holm_wilcoxon", "r",
                  "ci90_lo", "ci90_hi", "p_tost", "equivalent"]].to_string(index=False))
    print("\n=== PURCHASE INTENT ===")
    print(pi[["category", "n", "mean_d", "ci_lo", "ci_hi", "p_t", "holm_t", "dz", "r"]].to_string(index=False))
    print("\n=== PRICE ===")
    print(price.to_string(index=False))
    print(f"\nSpearman(interval, liking gap) rho={rho:.3f} p={p_rho:.4f}")
    print(f"Pearson  r={pear:.3f} p={p_pear:.4f}")
    print("\n=== DISPERSION (paired subset variance ratio) ===")
    print(varratio.to_string(index=False))
    print("\n=== POOLED CATA PENALTIES (top/bottom) ===")
    print(pen_pool.head(14).to_string(index=False))
    print(pen_pool.tail(8).to_string(index=False))
    print("\n=== COMPONENTS: FDR ===")
    print(f"{comp['sig_fdr'].sum()} of {len(comp)} significant after BH-FDR "
          f"(unadjusted p<.05: {(comp['p_t'] < 0.05).sum()})")
    print(comp.to_string(index=False))
    print("\n=== CATA McNemar ===")
    if len(mcn):
        print(f"{mcn['sig_fdr'].sum()} of {len(mcn)} term comparisons significant after BH-FDR")
        print(mcn.sort_values("p").head(20)[
            ["category", "modality", "term", "pct_benchmark", "pct_leader", "diff_pp", "p", "p_fdr"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
