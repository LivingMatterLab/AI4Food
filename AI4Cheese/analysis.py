"""Table-only analysis for the deidentified cheese study release."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

CHEESES = ["cheddar", "feta", "mozzarella"]
SENSORY = [
    "overall_liking", "chewiness", "hardness", "moisture", "graininess",
    "dairy_flavor", "fattiness", "tastiness", "softness",
]
JAR = [
    "jar_moisture", "jar_firmness", "jar_saltiness",
    "jar_fattiness", "jar_sharpness", "jar_creaminess",
]


def paired_sensory(data):
    rows = []
    for cheese in CHEESES:
        subset = data[data["cheese_type"] == cheese]
        for measure in SENSORY:
            paired = subset.pivot(
                index="participant_id", columns="product", values=measure
            ).dropna()
            dairy = paired["dairy"]
            plant = paired["plant_based"]
            statistic, p_value = stats.wilcoxon(plant, dairy)
            rows.append({
                "cheese_type": cheese, "measure": measure, "n_pairs": len(paired),
                "plant_mean": plant.mean(), "plant_sd": plant.std(),
                "dairy_mean": dairy.mean(), "dairy_sd": dairy.std(),
                "dairy_minus_plant": dairy.mean() - plant.mean(),
                "wilcoxon_statistic": statistic, "p_value": p_value,
            })
    return pd.DataFrame(rows)


def jar_summary(data):
    rows = []
    for (cheese, product), subset in data.groupby(["cheese_type", "product"]):
        for measure in JAR:
            values = pd.to_numeric(subset[measure], errors="coerce").dropna()
            rows.append({
                "cheese_type": cheese, "product": product,
                "measure": measure, "n": len(values),
                "too_little_percent": 100 * (values <= 2).mean(),
                "just_right_percent": 100 * (values == 3).mean(),
                "too_much_percent": 100 * (values >= 4).mean(),
            })
    return pd.DataFrame(rows)


def tpa_results(data):
    metrics = [
        "stiffness", "hardness", "cohesiveness",
        "springiness", "resilience", "chewiness",
    ]
    summary = data.groupby("product")[metrics].agg(["count", "mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    tests = []
    for cheese in CHEESES:
        dairy = data[data["product"] == f"dairy_{cheese}"]
        plant = data[data["product"] == f"plant_{cheese}"]
        for metric in metrics:
            statistic, p_value = stats.mannwhitneyu(
                dairy[metric].dropna(), plant[metric].dropna(),
                alternative="two-sided",
            )
            tests.append({
                "cheese_type": cheese, "measure": metric,
                "mann_whitney_u": statistic, "p_value": p_value,
            })
    return summary.reset_index(), pd.DataFrame(tests)


def rheology_results(data):
    rows = []
    target = 2 * np.pi
    for (product, sample_id), group in data.groupby(["product", "sample_id"]):
        ordered = group.sort_values("angular_frequency")
        at_one_hz = ordered.iloc[
            (ordered["angular_frequency"] - target).abs().argmin()
        ]
        positive = ordered[
            (ordered["angular_frequency"] > 0)
            & (ordered["storage_modulus"] > 0)
        ]
        slope = (
            stats.linregress(
                np.log10(positive["angular_frequency"]),
                np.log10(positive["storage_modulus"]),
            ).slope
            if len(positive) >= 2 else np.nan
        )
        rows.append({
            "product": product, "sample_id": sample_id,
            "storage_modulus_at_1hz": at_one_hz["storage_modulus"],
            "loss_modulus_at_1hz": at_one_hz["loss_modulus"],
            "tan_delta_at_1hz": at_one_hz["tan_delta"],
            "log_storage_modulus_slope": slope,
        })
    replicate_results = pd.DataFrame(rows)
    summary = replicate_results.groupby("product").agg(
        n=("sample_id", "nunique"),
        storage_modulus_at_1hz_mean=("storage_modulus_at_1hz", "mean"),
        storage_modulus_at_1hz_sd=("storage_modulus_at_1hz", "std"),
        loss_modulus_at_1hz_mean=("loss_modulus_at_1hz", "mean"),
        loss_modulus_at_1hz_sd=("loss_modulus_at_1hz", "std"),
        tan_delta_at_1hz_mean=("tan_delta_at_1hz", "mean"),
        tan_delta_at_1hz_sd=("tan_delta_at_1hz", "std"),
        log_storage_modulus_slope_mean=("log_storage_modulus_slope", "mean"),
        log_storage_modulus_slope_sd=("log_storage_modulus_slope", "std"),
    )
    replicate_results.to_csv(RESULTS / "rheology_by_sample.csv", index=False)
    return summary.reset_index()


def main():
    sensory = pd.read_csv(HERE / "data" / "sensory.csv")
    tpa = pd.read_csv(HERE / "data" / "tpa.csv")
    rheology = pd.read_csv(HERE / "data" / "rheology_frequency_sweep.csv")
    paired_sensory(sensory).to_csv(
        RESULTS / "sensory_paired_tests.csv", index=False
    )
    jar_summary(sensory).to_csv(RESULTS / "jar_summary.csv", index=False)
    tpa_summary, tpa_tests = tpa_results(tpa)
    tpa_summary.to_csv(RESULTS / "tpa_summary.csv", index=False)
    tpa_tests.to_csv(RESULTS / "tpa_tests.csv", index=False)
    rheology_results(rheology).to_csv(
        RESULTS / "rheology_summary.csv", index=False
    )
    print(f"Wrote table outputs to {RESULTS}")


if __name__ == "__main__":
    main()
