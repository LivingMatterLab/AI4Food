"""
Meatball Study - Sensory Analysis
Clean, sequential analysis code for the de-identified GitHub dataset.
"""

import pandas as pd
import numpy as np
from scipy import stats

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_excel("Meatball_Sensory_Data_GitHub.xlsx")
print(f"Loaded {len(df)} participants\n")

# Define products and attributes
products = ["Soy", "Soy-Wheat", "Beef", "Beef-Mushroom"]
sensory_attrs = ["Chewiness", "Hardness", "Moistness", "Fibrousness",
                 "Meatiness", "Fattiness", "Tastiness", "Softness"]
jar_attrs = ["JAR_Moistness", "JAR_Chewiness", "JAR_Savoriness",
             "JAR_Fattiness", "JAR_Fibrousness"]

# Identify participants who tasted all 4 (no NaN in Beef columns)
df_all4 = df[df["Beef_Tastiness"].notna()].copy()
print(f"Participants who tasted all 4: {len(df_all4)}")
print(f"Plant-only participants: {len(df) - len(df_all4)}\n")

# =============================================================================
# 1. DEMOGRAPHICS SUMMARY
# =============================================================================

print("=" * 60)
print("1. DEMOGRAPHICS")
print("=" * 60)

# Age distribution
age_labels = {1: "18-24", 2: "25-30", 3: "31-35", 4: "36-40", 5: "41-45",
              6: "46-50", 7: "51-55", 8: "56-60", 9: "61-65", 10: "66-70",
              11: "71-75", 12: "75-80"}
print("\nAge:")
for code, label in age_labels.items():
    n = (df["Age"] == code).sum()
    if n > 0:
        print(f"  {label}: {n} ({n/len(df)*100:.1f}%)")

# Gender distribution
gender_labels = {1: "Male", 2: "Female", 3: "Non-binary", 4: "Not listed"}
print("\nGender:")
for code, label in gender_labels.items():
    n = (df["Gender"] == code).sum()
    if n > 0:
        print(f"  {label}: {n} ({n/len(df)*100:.1f}%)")

# Ethnicity distribution
ethnicity_labels = {1: "Asian", 2: "White", 3: "Hispanic/Latino", 4: "Other/Multiracial"}
print("\nEthnicity:")
for code, label in ethnicity_labels.items():
    n = (df["Ethnicity"] == code).sum()
    if n > 0:
        print(f"  {label}: {n} ({n/len(df)*100:.1f}%)")

# Diet type distribution
diet_labels = {1: "Omnivore", 2: "Pescatarian", 3: "Flexitarian",
               4: "Vegetarian", 5: "Vegan"}
print("\nDiet Type:")
for code, label in diet_labels.items():
    n = (df["DietType"] == code).sum()
    if n > 0:
        print(f"  {label}: {n} ({n/len(df)*100:.1f}%)")

# =============================================================================
# 2. SENSORY RATINGS BY PRODUCT (Mean, SD, SE)
# =============================================================================

print("\n" + "=" * 60)
print("2. SENSORY RATINGS BY PRODUCT")
print("=" * 60)

# Use all available data for each product (n=128 for plants, n=116 for animals)
sensory_summary = []
for product in products:
    print(f"\n{product}:")
    for attr in sensory_attrs:
        col = f"{product}_{attr}"
        vals = df[col].dropna()
        n = len(vals)
        mean = vals.mean()
        sd = vals.std()
        se = sd / np.sqrt(n)
        print(f"  {attr:12s}: M={mean:.2f}, SD={sd:.2f}, SE={se:.2f}, n={n}")
        sensory_summary.append({
            "Product": product, "Attribute": attr,
            "Mean": mean, "SD": sd, "SE": se, "n": n
        })

# =============================================================================
# 3. PLANT VS ANIMAL COMPARISON (Aggregate)
# =============================================================================

print("\n" + "=" * 60)
print("3. PLANT VS ANIMAL COMPARISON")
print("=" * 60)

print("\nUsing only participants who tasted all 4 (n={})".format(len(df_all4)))
print("\nAttribute        Plant Mean  Animal Mean  Diff (P-A)  t-stat    p-value")
print("-" * 75)

for attr in sensory_attrs:
    # Plant average (Soy + Soy-Wheat) / 2 per person, then mean
    plant_vals = df_all4[[f"Soy_{attr}", f"Soy-Wheat_{attr}"]].mean(axis=1)
    animal_vals = df_all4[[f"Beef_{attr}", f"Beef-Mushroom_{attr}"]].mean(axis=1)

    plant_mean = plant_vals.mean()
    animal_mean = animal_vals.mean()
    diff = plant_mean - animal_mean

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(plant_vals, animal_vals)

    sig = ""
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"

    print(f"{attr:12s}     {plant_mean:6.2f}      {animal_mean:6.2f}       {diff:+5.2f}     {t_stat:6.2f}    {p_val:.4f} {sig}")

# =============================================================================
# 4. TASTINESS COMPARISON ACROSS ALL 4 PRODUCTS
# =============================================================================

print("\n" + "=" * 60)
print("4. TASTINESS BY PRODUCT")
print("=" * 60)

print("\nProduct           Mean    SD      SE      Median   n")
print("-" * 55)
for product in products:
    vals = df[f"{product}_Tastiness"].dropna()
    print(f"{product:15s}  {vals.mean():5.2f}   {vals.std():5.2f}   {vals.std()/np.sqrt(len(vals)):5.2f}   {vals.median():5.1f}    {len(vals)}")

# Pairwise comparisons (Wilcoxon signed-rank for paired data)
print("\nPairwise Wilcoxon tests (using all-4 participants):")
from itertools import combinations
for p1, p2 in combinations(products, 2):
    v1 = df_all4[f"{p1}_Tastiness"]
    v2 = df_all4[f"{p2}_Tastiness"]
    stat, p = stats.wilcoxon(v1, v2)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {p1:12s} vs {p2:12s}: W={stat:6.0f}, p={p:.4f} {sig}")

# =============================================================================
# 5. JAR DISTRIBUTIONS
# =============================================================================

print("\n" + "=" * 60)
print("5. JAR (JUST-ABOUT-RIGHT) DISTRIBUTIONS")
print("=" * 60)

jar_labels = ["Moistness", "Chewiness", "Savoriness", "Fattiness", "Fibrousness"]

for product in products:
    print(f"\n{product}:")
    print("  Attribute      Not Enough   Just Right   Too Much     n")
    print("  " + "-" * 55)
    for attr, label in zip(jar_attrs, jar_labels):
        col = f"{product}_{attr}"
        vals = df[col].dropna()
        n = len(vals)
        if n > 0:
            pct_low = (vals == 1).sum() / n * 100
            pct_jr = (vals == 2).sum() / n * 100
            pct_high = (vals == 3).sum() / n * 100
            print(f"  {label:12s}     {pct_low:5.1f}%       {pct_jr:5.1f}%       {pct_high:5.1f}%    {n}")

# =============================================================================
# 6. PENALTY ANALYSIS (JAR vs Tastiness)
# =============================================================================

print("\n" + "=" * 60)
print("6. PENALTY ANALYSIS")
print("=" * 60)

print("\nPenalty = Mean Tastiness(JAR) - Mean Tastiness(Not JAR)")
print("Actionable = |Penalty| >= 0.5 AND p < 0.05\n")

for product in products:
    print(f"\n{product}:")
    print("  Attribute      Group         n    Mean    Penalty   t       p")
    print("  " + "-" * 65)

    tast_col = f"{product}_Tastiness"

    for attr, label in zip(jar_attrs, jar_labels):
        jar_col = f"{product}_{attr}"

        # Get just-right reference
        jr_mask = df[jar_col] == 2
        jr_tast = df.loc[jr_mask, tast_col].dropna()
        jr_mean = jr_tast.mean() if len(jr_tast) > 0 else np.nan
        jr_n = len(jr_tast)

        for code, group in [(2, "Just right"), (1, "Not enough"), (3, "Too much")]:
            mask = df[jar_col] == code
            grp_tast = df.loc[mask, tast_col].dropna()
            n = len(grp_tast)
            mean = grp_tast.mean() if n > 0 else np.nan

            if code == 2:
                print(f"  {label:12s}   {group:12s}  {n:3d}   {mean:.2f}    (ref)")
            else:
                if n >= 3 and jr_n >= 3:
                    penalty = jr_mean - mean
                    t_stat, p_val = stats.ttest_ind(jr_tast, grp_tast)
                    flag = "***" if (abs(penalty) >= 0.5 and p_val < 0.05) else ""
                    print(f"  {''*12}   {group:12s}  {n:3d}   {mean:.2f}    {penalty:+.2f}    {t_stat:5.2f}  {p_val:.4f} {flag}")
                elif n > 0:
                    penalty = jr_mean - mean if not np.isnan(jr_mean) else np.nan
                    print(f"  {''*12}   {group:12s}  {n:3d}   {mean:.2f}    {penalty:+.2f}    (n<3)")

# =============================================================================
# 7. INDIVIDUAL PREFERENCE DISTRIBUTIONS
# =============================================================================

print("\n" + "=" * 60)
print("7. INDIVIDUAL PREFERENCE DISTRIBUTIONS (Plant vs Animal)")
print("=" * 60)

print(f"\nUsing participants who tasted all 4 (n={len(df_all4)})")
print("Difference = (Plant avg) - (Animal avg) per person")
print("\nAttribute      % Prefer Animal  % No Pref  % Prefer Plant  Mean Diff")
print("-" * 70)

for attr in sensory_attrs:
    # Calculate per-person difference
    plant_avg = df_all4[[f"Soy_{attr}", f"Soy-Wheat_{attr}"]].mean(axis=1)
    animal_avg = df_all4[[f"Beef_{attr}", f"Beef-Mushroom_{attr}"]].mean(axis=1)
    diff = plant_avg - animal_avg

    pct_animal = (diff < 0).sum() / len(diff) * 100
    pct_neutral = (diff == 0).sum() / len(diff) * 100
    pct_plant = (diff > 0).sum() / len(diff) * 100
    mean_diff = diff.mean()

    print(f"{attr:12s}      {pct_animal:5.1f}%          {pct_neutral:4.1f}%        {pct_plant:5.1f}%        {mean_diff:+.2f}")

# =============================================================================
# 8. CORRELATION MATRIX (Sensory attributes within product)
# =============================================================================

print("\n" + "=" * 60)
print("8. ATTRIBUTE CORRELATIONS (Example: Beef)")
print("=" * 60)

beef_cols = [f"Beef_{attr}" for attr in sensory_attrs]
corr_matrix = df_all4[beef_cols].corr()

print("\nCorrelation matrix (Beef meatball):")
print("             " + "  ".join([a[:4] for a in sensory_attrs]))
for i, attr in enumerate(sensory_attrs):
    row = [f"{corr_matrix.iloc[i, j]:.2f}" for j in range(len(sensory_attrs))]
    print(f"{attr:12s} " + "  ".join(row))


# =============================================================================
# 9. TPA TEXTURE PROFILE ANALYSIS (Instrumental Data, n=37)
# =============================================================================

print("\n" + "=" * 60)
print("9. TPA TEXTURE PROFILE ANALYSIS")
print("=" * 60)

# TPA data from double-compression tests (25%/s loading rate, 50% strain)
# Order: Beef, Beef-Mushroom, Soy-Wheat, Soy
tpa_products = ["Beef", "Beef-Mushroom", "Soy-Wheat", "Soy"]
tpa_data = {
    "Stiffness (N/mm)":  [91.7, 101.7, 56.2, 45.4],
    "Hardness (N)":      [2.31, 2.56, 1.41, 1.14],
    "Cohesiveness":      [0.43, 0.43, 0.32, 0.29],
    "Springiness":       [0.55, 0.52, 0.34, 0.27],
    "Resilience":        [0.37, 0.40, 0.27, 0.22],
    "Chewiness (N)":     [0.56, 0.57, 0.19, 0.09],
}

print("\nTPA Parameters (n=37 samples per product):")
print("\nParameter          Beef    Beef-Mush  Soy-Wheat    Soy")
print("-" * 60)
for param, values in tpa_data.items():
    print(f"{param:18s} {values[0]:6.2f}    {values[1]:6.2f}      {values[2]:6.2f}    {values[3]:6.2f}")

# =============================================================================
# 10. TPA-SENSORY CORRELATIONS (Kendall's tau)
# =============================================================================

print("\n" + "=" * 60)
print("10. TPA-SENSORY CORRELATIONS (Kendall's tau)")
print("=" * 60)

# Get sensory means in same product order as TPA
sensory_means = {}
for attr in sensory_attrs:
    means = []
    for prod in tpa_products:
        col = f"{prod}_{attr}"
        means.append(df[col].dropna().mean())
    sensory_means[attr] = means

# TPA metrics for correlation
tpa_metrics = {
    "TPA_Hardness": tpa_data["Hardness (N)"],
    "TPA_Stiffness": tpa_data["Stiffness (N/mm)"],
    "TPA_Cohesiveness": tpa_data["Cohesiveness"],
    "TPA_Springiness": tpa_data["Springiness"],
    "TPA_Resilience": tpa_data["Resilience"],
    "TPA_Chewiness": tpa_data["Chewiness (N)"],
}

print("\nKendall's tau correlation matrix (TPA vs Sensory, n=4 products):")
print("\n" + " " * 16 + "  ".join([a[:6] for a in sensory_attrs]))
print("-" * 80)

kendall_matrix = []
for tpa_name, tpa_vals in tpa_metrics.items():
    row_vals = []
    for attr in sensory_attrs:
        tau, p = stats.kendalltau(tpa_vals, sensory_means[attr])
        row_vals.append((tau, p))
    kendall_matrix.append((tpa_name, row_vals))

    row_str = f"{tpa_name.replace('TPA_', ''):14s}"
    for tau, p in row_vals:
        sig = "*" if p < 0.05 else " "
        row_str += f" {tau:+5.2f}{sig}"
    print(row_str)

print("\n* p < 0.05")

# Key correlations
print("\nKey TPA-Sensory Correlations:")
print("-" * 50)

key_pairs = [
    ("TPA_Hardness", "Hardness"),
    ("TPA_Chewiness", "Chewiness"),
    ("TPA_Chewiness", "Meatiness"),
    ("TPA_Springiness", "Softness"),
]

for tpa_name, sensory_name in key_pairs:
    tpa_vals = tpa_metrics[tpa_name]
    sensory_vals = sensory_means[sensory_name]
    tau, p = stats.kendalltau(tpa_vals, sensory_vals)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {tpa_name.replace('TPA_', ''):12s} vs {sensory_name:12s}: tau = {tau:+.2f}, p = {p:.3f} {sig}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Tastiness ranking
print("\nTastiness ranking:")
tast_means = [(p, df[f"{p}_Tastiness"].dropna().mean()) for p in products]
tast_means.sort(key=lambda x: x[1], reverse=True)
for i, (p, m) in enumerate(tast_means, 1):
    print(f"  {i}. {p}: {m:.2f}")

# Biggest Plant-Animal gaps
print("\nLargest Plant-Animal differences:")
gaps = []
for attr in sensory_attrs:
    plant = df_all4[[f"Soy_{attr}", f"Soy-Wheat_{attr}"]].mean(axis=1).mean()
    animal = df_all4[[f"Beef_{attr}", f"Beef-Mushroom_{attr}"]].mean(axis=1).mean()
    gaps.append((attr, plant - animal))
gaps.sort(key=lambda x: abs(x[1]), reverse=True)
for attr, gap in gaps:
    direction = "Plant higher" if gap > 0 else "Animal higher"
    print(f"  {attr}: {gap:+.2f} ({direction})")


