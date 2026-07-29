# AI4Shrimp

Deidentified analysis data for the Comparative Organoleptic Analysis of Shrimp
Texture study. Participants evaluated conventional and plant-based breaded
shrimp in a counterbalanced, within-participant design.

## Files

- `data/sensory.csv`: paired sensory, JAR, purchase-intent, and CATA responses
- `data/tpa.csv`: instrumental texture-profile measurements
- `analysis.py`: statistical analyses that write CSV tables only

Run from this folder:

```bash
python analysis.py
```

Requirements: Python 3.10+, pandas, NumPy, and SciPy.

