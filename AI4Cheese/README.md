# AI4Cheese

Deidentified paired sensory and instrumental data comparing dairy and
plant-based cheddar, feta, and mozzarella.

## Files

- `data/sensory.csv`: paired consumer sensory and JAR responses
- `data/tpa.csv`: texture-profile measurements
- `data/rheology_frequency_sweep.csv`: oscillatory frequency-sweep measurements
- `analysis.py`: statistical analyses that write CSV tables only

Run from this folder:

```bash
python analysis.py
```

Requirements: Python 3.10+, pandas, NumPy, and SciPy.

## Privacy

Direct identifiers, survey metadata, dates, free text, demographics, attitudes,
and source record numbers are excluded. 

This release includes completed sensory, TPA, and frequency-sweep rheology
work. Unfinished CANN, tension, and additional shear/compression work is not
included.
