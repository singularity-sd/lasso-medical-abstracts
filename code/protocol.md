Forecast under test: T3v3-2026-0041 — LASSO regression in Health Informatics.

Question: can an L1-penalized linear classifier provide a compact and useful representation of medical abstracts when the input is high-dimensional text? This is a second, independent attempt after the EHR readmission analysis. It tests a setting where sparsity has a direct purpose: selecting a small vocabulary from bag-of-words features.

Data: Medical Abstracts Text Classification Corpus, 14,438 rows and five condition labels, from the authors' public GitHub repository. Before modelling, we will normalize case and whitespace and inspect exact-text duplicates and label conflicts. To prevent a repeated abstract appearing on both sides of evaluation, the analysis set will contain one normalized abstract per group only when all copies carry the same single label. Groups whose identical abstract has conflicting labels will be excluded. Remaining unique abstracts will be split once, stratified 80/20, with seed 20261107.

Representation and models: TF-IDF unigrams and bigrams (minimum document frequency 2, at most 50,000 features); no stop-word removal, so negation remains available. Compare one-vs-rest logistic regression with L1 (LASSO) and L2 (ridge) penalties. For each penalty, choose C from {0.03, 0.3, 3} by 3-fold stratified training-only cross-validation maximizing macro-F1. Refit once on the full training split and evaluate the held-out split. The held-out labels will not be used for tuning.

Primary metric: macro-F1 on held-out unique abstracts. Secondary metrics: accuracy, balanced accuracy, weighted-F1, macro one-vs-rest ROC AUC, and log loss. Report the L1 non-zero feature count. Estimate a paired, class-stratified bootstrap 95% interval for the difference in macro-F1 (L1 minus L2) on the fixed holdout. Also quantify the overlap of selected vocabulary across the three training folds at the selected L1 C.

Interpretation: this is a text-routing benchmark, not diagnosis, patient-level prediction, clinical decision support, or a test of disease causation. The corpus is curated and the labels describe broad abstract topics. No claim of novelty or future forecast confirmation will be made.

Execution: the initial run stopped before any metric because scikit-learn 1.8 requires an explicit One-vs-Rest wrapper for multiclass liblinear. The code was corrected accordingly; the models and the statistical settings of the protocol do not change.

Technical note: in the first complete run the training finished, but the AUC computation required normalised probabilities, while the OvR model returns per-class margins. No metric had been saved or looked at. The computation was corrected to the mean of the five binary one-vs-rest AUCs on the margins, without changing the split, the models or the hyperparameter selection.

Reproducibility update: after reading the complete run, the configuration was moved from the deprecated penalty argument to the equivalent l1_ratio form indicated by scikit-learn 1.8 (1 for L1, 0 for L2). To deliver consistent files and metrics, the script is re-run from scratch with unchanged data, seed, grid and split.
