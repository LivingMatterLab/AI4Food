"""Table-only analysis for the deidentified Stanford Burger Experiment."""

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

BURGER_ORDER = ["Beef", "Turkey", "Bean", "Beyond"]
SENSORY = [
    "overall_liking", "sensory_hardness", "sensory_chewiness",
    "sensory_moistness", "sensory_fibrousness", "sensory_meatiness",
    "sensory_fattiness", "sensory_tastiness", "sensory_softness",
]
JAR = [
    "jar_moistness", "jar_chewiness", "jar_savoriness",
    "jar_fattiness", "jar_fibrousness",
]


def sensory_results(data):
    summaries = []
    tests = []
    for measure in SENSORY:
        for burger in BURGER_ORDER:
            values = pd.to_numeric(
                data.loc[data["burger_name"] == burger, measure], errors="coerce"
            ).dropna()
            summaries.append({
                "measure": measure, "burger": burger, "n": len(values),
                "mean": values.mean(), "sd": values.std(), "median": values.median(),
            })
        groups = [
            pd.to_numeric(
                data.loc[data["burger_name"] == burger, measure], errors="coerce"
            ).dropna()
            for burger in BURGER_ORDER
        ]
        h_stat, p_value = stats.kruskal(*groups)
        tests.append({
            "measure": measure, "comparison": "all_burgers",
            "test": "kruskal_wallis", "statistic": h_stat,
            "p_value": p_value, "p_adjusted": p_value,
        })
        pairs = list(combinations(BURGER_ORDER, 2))
        for first, second in pairs:
            first_values = pd.to_numeric(
                data.loc[data["burger_name"] == first, measure], errors="coerce"
            ).dropna()
            second_values = pd.to_numeric(
                data.loc[data["burger_name"] == second, measure], errors="coerce"
            ).dropna()
            statistic, p_value = stats.mannwhitneyu(
                first_values, second_values, alternative="two-sided"
            )
            tests.append({
                "measure": measure,
                "comparison": f"{first}_vs_{second}",
                "test": "mann_whitney_u",
                "statistic": statistic,
                "p_value": p_value,
                "p_adjusted": min(p_value * len(pairs), 1.0),
            })
    return pd.DataFrame(summaries), pd.DataFrame(tests)


def jar_results(data):
    rows = []
    for burger in BURGER_ORDER:
        subset = data[data["burger_name"] == burger]
        for measure in JAR:
            values = pd.to_numeric(subset[measure], errors="coerce").dropna()
            rows.append({
                "burger": burger, "measure": measure, "n": len(values),
                "not_enough_percent": 100 * (values == 1).mean(),
                "just_right_percent": 100 * (values == 2).mean(),
                "too_much_percent": 100 * (values == 3).mean(),
            })
    return pd.DataFrame(rows)


def treatment_results(data):
    rows = []
    for burger_type in ["Animal", "Plant"]:
        subset = data[data["burger_type"] == burger_type]
        for measure in SENSORY:
            control = pd.to_numeric(
                subset.loc[subset["treatment_condition"] == "Control", measure],
                errors="coerce",
            ).dropna()
            treatment = pd.to_numeric(
                subset.loc[subset["treatment_condition"] == "Treatment", measure],
                errors="coerce",
            ).dropna()
            statistic, p_value = stats.mannwhitneyu(
                control, treatment, alternative="two-sided"
            )
            rows.append({
                "burger_type": burger_type, "measure": measure,
                "control_n": len(control), "control_mean": control.mean(),
                "treatment_n": len(treatment), "treatment_mean": treatment.mean(),
                "mann_whitney_u": statistic, "p_value": p_value,
            })
    return pd.DataFrame(rows)


def tpa_results(data):
    metrics = [
        "stiffness_kPa", "hardness_N", "cohesiveness",
        "springiness", "resilience", "chewiness_N",
    ]
    summary = data.groupby("burger_type")[metrics].agg(["count", "mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    tests = []
    for metric in metrics:
        groups = [
            group[metric].dropna().to_numpy()
            for _, group in data.groupby("burger_type", sort=True)
        ]
        statistic, p_value = stats.kruskal(*groups)
        tests.append({"measure": metric, "kruskal_h": statistic, "p_value": p_value})
    return summary.reset_index(), pd.DataFrame(tests)


def main():
    consumer = pd.read_csv(HERE / "data" / "consumer.csv")
    tpa = pd.read_csv(HERE / "data" / "tpa.csv")
    summary, tests = sensory_results(consumer)
    summary.to_csv(RESULTS / "sensory_summary.csv", index=False)
    tests.to_csv(RESULTS / "sensory_tests.csv", index=False)
    jar_results(consumer).to_csv(RESULTS / "jar_summary.csv", index=False)
    treatment_results(consumer).to_csv(RESULTS / "treatment_tests.csv", index=False)
    tpa_summary, tpa_tests = tpa_results(tpa)
    tpa_summary.to_csv(RESULTS / "tpa_summary.csv", index=False)
    tpa_tests.to_csv(RESULTS / "tpa_tests.csv", index=False)
    print(f"Wrote table outputs to {RESULTS}")


if __name__ == "__main__":
    main()
