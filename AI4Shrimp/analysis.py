"""Table-only analysis for the deidentified COAST release."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def paired_results(data):
    rows = []
    measures = [
        "appearance", "flavor", "texture", "overall", "purchase",
        "shrimp_intensity", "firmness", "chewiness",
    ]
    for measure in measures:
        paired = data.pivot(
            index="participant_id", columns="product", values=measure
        ).dropna()
        conventional = paired["conventional"]
        plant = paired["plant_based"]
        difference = plant - conventional
        t_stat, t_p = stats.ttest_rel(plant, conventional)
        try:
            w_stat, w_p = stats.wilcoxon(plant, conventional)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        rows.append({
            "measure": measure,
            "n_pairs": len(paired),
            "conventional_mean": conventional.mean(),
            "conventional_sd": conventional.std(),
            "plant_based_mean": plant.mean(),
            "plant_based_sd": plant.std(),
            "mean_difference_plant_minus_conventional": difference.mean(),
            "paired_t_statistic": t_stat,
            "paired_t_p": t_p,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_p": w_p,
            "paired_cohens_d": (
                difference.mean() / difference.std(ddof=1)
                if difference.std(ddof=1) > 0 else np.nan
            ),
        })
    result = pd.DataFrame(rows)
    result["wilcoxon_p_fdr"] = bh_adjust(result["wilcoxon_p"].fillna(1))
    return result


def cata_results(data):
    rows = []
    for column in [c for c in data.columns if c.startswith("cata_")]:
        paired = data.pivot(
            index="participant_id", columns="product", values=column
        ).dropna()
        conventional = paired["conventional"].astype(int)
        plant = paired["plant_based"].astype(int)
        conventional_only = int(((conventional == 1) & (plant == 0)).sum())
        plant_only = int(((conventional == 0) & (plant == 1)).sum())
        discordant = conventional_only + plant_only
        p_value = (
            stats.binomtest(
                min(conventional_only, plant_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
            if discordant else 1.0
        )
        rows.append({
            "attribute": column.removeprefix("cata_"),
            "n_pairs": len(paired),
            "conventional_percent": 100 * conventional.mean(),
            "plant_based_percent": 100 * plant.mean(),
            "conventional_only": conventional_only,
            "plant_based_only": plant_only,
            "exact_mcnemar_p": p_value,
        })
    result = pd.DataFrame(rows)
    result["exact_mcnemar_p_fdr"] = bh_adjust(result["exact_mcnemar_p"])
    return result


def tpa_results(data):
    metrics = [
        "stiffness", "hardness", "cohesiveness",
        "springiness", "resilience", "chewiness",
    ]
    summary = data.groupby("product")[metrics].agg(["count", "mean", "std", "median"])
    summary.columns = ["_".join(column) for column in summary.columns]
    tests = []
    for metric in metrics:
        groups = [
            group[metric].dropna().to_numpy()
            for _, group in data.groupby("product", sort=True)
        ]
        statistic, p_value = stats.kruskal(*groups)
        tests.append({"measure": metric, "kruskal_h": statistic, "p_value": p_value})
    return summary.reset_index(), pd.DataFrame(tests)


def main():
    sensory = pd.read_csv(HERE / "data" / "sensory.csv")
    tpa = pd.read_csv(HERE / "data" / "tpa.csv")
    paired_results(sensory).to_csv(RESULTS / "sensory_paired_tests.csv", index=False)
    cata_results(sensory).to_csv(RESULTS / "cata_paired_tests.csv", index=False)
    tpa_summary, tpa_tests = tpa_results(tpa)
    tpa_summary.to_csv(RESULTS / "tpa_summary.csv", index=False)
    tpa_tests.to_csv(RESULTS / "tpa_tests.csv", index=False)
    print(f"Wrote table outputs to {RESULTS}")


if __name__ == "__main__":
    main()
