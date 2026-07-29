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

## Privacy

Direct identifiers, survey metadata, dates, free text, demographics, and source
record numbers are excluded. Participant IDs are new random values used only to
preserve the paired design; the mapping to source rows was not retained. Rows
are randomly ordered.

The public data are suitable for reproducing primary sensory, CATA, and
instrumental analyses, not participant-level demographic analyses.
