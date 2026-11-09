# Dairy and dairy-free sensory benchmark: public data

This repository holds the de-identified consumer data and analysis code behind "Open-source benchmarking of dairy and dairy-free products" (Koosis, Cotto, and Kuhl; under revision at Future Foods, manuscript FUFO-D-26-00951).

## Source

NECTAR (Food System Innovations) designed the Taste of the Industry 2026 study. Palate Insights collected the data in blinded, randomized, in-person tastings in San Francisco, CA, and New York City, NY, between September and November 2025. This file covers the ten dairy benchmarks and the ten dairy-free leaders that NECTAR selected, one pair per category.

## Repository contents

| Path | Contents |
| --- | --- |
| `data/responsesdairy_public.csv` | One row per respondent (1,437 rows, 247 columns) |
| `data/data_dictionary.csv` | Product tag, question type, modality, question text, and response count for each column |
| `analysis/01_primary_statistics.py` | Paired tests, equivalence tests, component ratings, purchase intent, CATA penalties, price thresholds, and rating distributions (Tables 2 and 3; Figures 2 to 4 and S1 to S4) |
| `analysis/02_demographics.py` | Demographic composition (Table 1) |
| `analysis/03_supplementary_statistics.py` | Equivalence corrections, alternative margins, and age analysis (Supplementary Tables S1 to S3) |
| `analysis/04_cata_sensitivity.py` | Respondent-clustered and category-adjusted CATA penalties (Supplementary Table S4) |
| `requirements.txt` | Python packages |

## What we removed

The file contains no brand names, nutrition panels, or ingredient lists. To meet this rule, we removed 92 columns from the original export:

- 61 free-text answers (likes, dislikes, off-flavors), because some respondents named brands
- 18 event-feedback questions about Palate Insights events
- 11 questions about balanced-dairy products, which the study excluded
- 2 personal details that the analysis does not use (ethnicity and ages of children)

Products appear only as generic tags, such as `Barista Milk Animal Benchmark` and `Barista Milk Plant-based Leader`. The three-digit numbers in the question text (for example, "latte 334") are blinding codes.

## Column format

Each column header keeps the export's tags so the analysis code can find it:

```
[RELATED_PRODUCT: <category> <Animal Benchmark | Plant-based Leader>]
[QUESTION_TYPE: <Likert | CATA | Multiple Choice | Demographic | Conceptual>]
[MODALITY: <Overall Liking | Flavor | Texture / Mouthfeel | Appearance | Similarity | ...>]
<question text>
```

- **Likert** answers are text labels on seven-point scales. The analysis maps them to 1 to 7 (for example, "Dislike very much" = 1 and "Like very much" = 7).
- **CATA** answers list the selected terms, separated by commas.
- **van-westendorp** answers give stated prices in US dollars for a stated package size.
- "No response" marks questions that a respondent did not answer, usually because they tasted a different category.
- **Purchase Intent (After NFP Reveal)** was asked after respondents viewed the nutrition panel and ingredient list. The panel and list themselves are not included.

Category `Milk` is fluid milk. Category `Yogurt` is plain Greek-style yogurt.

## Reproducing the paper

Install Python 3.10 or later and the packages in `requirements.txt`, then run the scripts in order from the `AI4Dairy` folder:

```
pip install -r requirements.txt
python analysis/01_primary_statistics.py
python analysis/02_demographics.py
python analysis/03_supplementary_statistics.py
python analysis/04_cata_sensitivity.py
```

Each script writes its results to `outputs/`. Script 04 reads the pooled penalties from script 01, so run 01 first. The bootstraps use fixed random seeds, so repeated runs give identical results. We checked that every output matches the results from the original export exactly.

## License and citation

The data are released under CC BY-NC 4.0, the license of the NECTAR Taste of the Industry 2026 report, with approval from Food System Innovations.

Please cite the paper and the NECTAR Taste of the Industry 2026 report when you use these data.
