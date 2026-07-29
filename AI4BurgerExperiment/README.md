# AI4BurgerExperiment

Deidentified consumer-sensory and texture-profile data for beef-mushroom,
turkey, bean, and pea-protein burgers.

This release covers the primary experiment only. Demographic, diet, behavioral,
and subgroup-analysis fields are intentionally excluded.

## Files

- `data/consumer.csv`: burger assignment, treatment condition, sensory ratings,
  and JAR responses
- `data/tpa.csv`: instrumental texture-profile measurements
- `analysis.py`: statistical analyses that write CSV tables only

Run from this folder:

```bash
python analysis.py
```

Requirements: Python 3.10+, pandas, NumPy, and SciPy.

## Privacy

Direct identifiers, survey metadata, dates, free text, demographics, behavioral
fields, source record numbers, and stable participant IDs are excluded. The
consumer rows are randomly ordered. Instrument sample IDs are newly randomized.
