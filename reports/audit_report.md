# Audit report

Scope: the first version of the notebook and report (`census_income_classification.ipynb`,
`Census_Income_Classification_Report.pdf`, `results.json`, `requirements.txt`) and the UCI Adult files. Every statement below was checked by
running code; items I could not check are listed at the end.

## 1. What was verified

The original notebook was re-executed as-is in a clean virtual environment built from the original
`requirements.txt` (Python 3.12.3, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4). The new
`results.json` matches the original one except for three values:

| Field | Original | Re-run |
|---|---|---|
| L2 baseline CV F1 | 0.6559 | 0.6558 (the PDF already says 0.6558) |
| Test ROC AUC | 0.92983 | 0.92983 (differs in the 6th decimal) |
| Test average precision | 0.83096 | 0.83096 (differs in the 6th decimal) |

Precision (0.692), recall (0.766), F1 (0.728), accuracy (0.863), the confusion matrix
(TN 6,635 / FP 796 / FN 546 / TP 1,792) and the threshold (0.35) are identical. The refactored pipeline in this
repository reproduces the same test confusion matrix and F1 (0.7276).

Data checks (48,842 rows; 32,561 + 16,281 in the official files): class counts 37,155 / 11,687 (23.9%
positive); missing values 2,799 (`workclass`), 2,809 (`occupation`), 857 (`native-country`); 52 exact
duplicate rows; 18 test rows identical to a training row.

## 2. Findings

Severity: **Critical** = wrong results or serious leakage; **High** = reproducibility / evaluation
correctness; **Medium** = maintainability or methodological weakness; **Low** = polish.

| # | Sev. | Finding | Evidence | Fix | Verification |
|---|---|---|---|---|---|
| 1 | - | **No data leakage found.** Preprocessing, L1 selection and the model sit in one `Pipeline`; the threshold is chosen on out-of-fold training predictions; the test set is used once. | Code review of every cell; unit tests (`tests/test_pipeline_integrity.py`) | none needed; tests added to keep it that way | tests pass; random labels give chance-level CV AUC |
| 2 | High | Dependency pins contradict the recorded run. `requirements.txt` says the results came from Python 3.12.3 / scikit-learn 1.9.1, but the stored notebook output reports a different interpreter and scikit-learn build. | Stored output of the notebook's first cell vs `requirements.txt` | Pin only direct dependencies, document the tested versions, record the environment inside `results.json` | clean-venv install and re-run reproduces the numbers |
| 3 | High | Hard-coded data folder `adult/` with no helpful error; `adult.test` read with `skiprows=1`, which silently drops a real row if the comment line is absent. | `read_csv` cell | `load_adult` searches `data/raw/`, uses `comment="|"`, validates columns and labels, raises an error that says where to get the files | `tests/test_data.py` |
| 4 | High | Narrative numbers are typed by hand in markdown and in a static reflection string; they can drift from the code output (the L2 baseline appears as 0.6558 in text and 0.6559 in `results.json`). | text vs `results.json` | Notebook consistency cell asserts the key numbers against `results/results.json` | notebook run |
| 5 | Medium | Everything lives in one notebook: no importable code, no tests, results written to the working directory. | repo layout | `src/census_income/` package, `scripts/run_pipeline.py`, `tests/` | 21 tests; full script run |
| 6 | Medium | The same 5 folds drive model comparison, tuning, threshold selection and the ablation, and there is no nested CV, so CV numbers are slightly optimistic. | code review | documented; the held-out test (used once) is the unbiased estimate. Test F1 0.728 vs OOF 0.727 shows no sign of a problem | seed / split sensitivity in section 10 |
| 7 | Medium | The L1 selector in front of gradient boosting sees unscaled inputs, so which columns survive depends on feature scale. | `SelectFromModel` on `preprocessor(scale=False)` | sensitivity check with standardised inputs | out-of-fold F1 0.727 both ways |
| 8 | Medium | L1 selection adds nothing measurable for gradient boosting (F1 0.7102 vs 0.7107) and its chosen columns vary between resamples (23 always, 16 sometimes, 3 never selected out of 42). | `fs_gb`, `selection_stability` | kept (it was part of the design and costs nothing) but described as a simplification, not an improvement | `results.json` |
| 9 | Medium | Fairness section had no uncertainty estimates. | tables only | bootstrap 95% CIs for group rates and for the male-female / white-black gaps; small groups flagged | `tests/test_fairness.py` |
| 10 | Medium | The text said that dropping `relationship` "left the gap unchanged". Out-of-fold, the male/female flagging ratio went from 3.2x to 3.6x. | ablation table | wording corrected | `results.json` |
| 11 | Low | The CV standard deviations are population std (ddof=0) over 5 non-independent folds. Differences smaller than about one std (e.g. 0.7107 vs 0.7109) are not meaningful. | `summarise_folds` | documented | - |
| 12 | Low | Not reported in the original: 57 rows duplicated on features, 5 feature groups with conflicting labels, no precision-recall curve, no missing-value plot, unused imports. | data audit | added to data report and figures | - |
| 13 | Low | Comparison with published results is not like for like: the project pools the official train and test files and re-splits them. | design | official split is a random 2/3-1/3 split, so pooling is valid; added an official-split check (F1 0.720, ROC AUC 0.928) | section 10 |

## 3. Robustness checks added

See `results/results.json` and notebook section 10: duplicates excluded (F1 unchanged), scaled selector
(no change), `HistGradientBoostingClassifier` benchmark (OOF F1 0.728, no better than the tuned model),
official split (F1 0.720), five other split seeds (F1 0.732 ± 0.007).
Hyper-parameters were tuned on the seed-42 training data, so the last two are split-variability checks,
not unbiased estimates.

## 4. Fairness audit

Sex and race are not model inputs (tested: flipping them does not change a prediction), but the model is
not neutral: male-female gaps on the test set are recall 0.137 (95% CI 0.086 to 0.191), false positive rate
0.104 (0.092 to 0.117), FPR ratio 3.4 (2.8 to 4.1). White-Black: recall gap 0.122 (0.038 to 0.210), FPR ratio
1.8 (1.4 to 2.5). The groups "Amer-Indian-Eskimo" (104) and "Other" (69) are too small to read. The
ablation shows `marital-status` and `relationship` carry the sex signal; removing both costs 0.07 F1.
These are measured disparities; their causes (labels reflecting 1994 pay gaps, household effects of
marital status) are interpretations, not results.

## 5. Not verified

- Exact interpreter / package build used for the original notebook output (only the pinned
  `requirements.txt` environment could be rebuilt).
- Behaviour of the full run on scikit-learn versions other than 1.9.1 (the 21 unit tests do pass on
  scikit-learn 1.8.0).
- Any claim about the UCI dataset licence: the repository does not redistribute the data and points to
  the UCI page for terms.

## Update (October 2026)

Results, figures and the notebook were regenerated on Python 3.14.7 (same pinned versions:
scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4). Test metrics and the confusion matrix are identical
(F1 0.7276; TN 6,635 / FP 796 / FN 546 / TP 1,792). Differences are limited to the 6th decimal of
ROC AUC and average precision, and to one row moving between TN and FP in the official-split check
(F1 0.7197 vs 0.7198). Sections 1 to 4 describe the original audit, which was run on Python 3.12.3.