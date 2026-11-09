"""Table 1: demographic composition of respondents."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "responsesdairy_public.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
ITEMS = ["Gender", "Age", "Diet", "Education", "Kids in HH"]

df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
rows = []
for item in ITEMS:
    col = [c for c in df.columns if f"[MODALITY: {item}]" in c][0]
    v = df[col].str.strip()
    v = v[(v != "") & (v != "No response")]
    pct = (100 * v.value_counts() / len(v)).round(1)
    for level, p in pct.items():
        rows.append(dict(characteristic=item, level=level, n=int((v == level).sum()),
                         percent=p, n_answered=len(v)))

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "table1_demographics.csv", index=False)
print(f"respondents: {len(df)}")
print(tab.to_string(index=False))
