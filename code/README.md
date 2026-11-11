# LASSO on medical abstract text

Second test of forecast **T3v3-2026-0041 — LASSO regression in Health Informatics**. This time the model classifies broad topics in medical abstracts, where a sparse vocabulary has a clear purpose.

## Result

After removing normalized duplicate texts whose labels conflicted, 8,298 unique, unambiguous abstracts remained. On a stratified 20% holdout, LASSO macro-F1 was 0.791 and ridge macro-F1 was 0.758. The paired difference was 0.033 (class-stratified bootstrap 95% interval 0.014–0.053). Ridge's macro AUC was slightly higher (0.957 vs 0.952). LASSO kept 2,201 of 250,000 class-specific coefficients non-zero (0.88%); mean pairwise overlap of selected vocabularies across three training folds was 0.358. This is a text categorization result, not clinical diagnosis.

## Files

- `main.pdf`, `main.tex`: paper and LaTeX source.
- `experiment.py`: complete analysis, including corpus audit, training-only tuning, holdout evaluation, and bootstrap.
- `protocol.md`: pre-analysis question, split, models, metrics, and execution notes.
- `results.json`: complete machine-readable summary.
- `metrics_holdout.csv`, `metrics_cv.csv`, `confusion_matrices.csv`, `lasso_selections.csv`: score and coefficient tables.
- `fig_results.png`: figure used in the paper.
- `data/`: source data and CC BY-SA 3.0 license text.

## Reproduce

From this folder, with Python 3.13 and `numpy`, `pandas`, `matplotlib`, and `scikit-learn` installed:

```powershell
py -3.13 experiment.py
```

The corpus and its attribution are described in [the authors' repository](https://github.com/sebischair/Medical-Abstracts-TC-Corpus) and paper ([DOI](https://doi.org/10.1145/3582768.3582795)). This research remains private and has not been added to the project website.
