"""Second LASSO/Health Informatics test: sparse categorization of medical abstracts.

Run from this directory: py -3.13 experiment.py
The official corpus is bundled in data/ under its CC BY-SA 3.0 terms.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20261107
LABEL_NAMES = {
    1: "Neoplasms",
    2: "Digestive system diseases",
    3: "Nervous system diseases",
    4: "Cardiovascular diseases",
    5: "General pathological conditions",
}
C_GRID = [0.03, 0.3, 3.0]


def load_unique_unambiguous():
    train = pd.read_csv(DATA / "medical_tc_train.csv")
    test = pd.read_csv(DATA / "medical_tc_test.csv")
    raw = pd.concat([train, test], ignore_index=True)
    raw["normalized"] = raw["medical_abstract"].astype(str).map(
        lambda s: " ".join(s.casefold().split())
    )
    groups = raw.groupby("normalized", sort=False, dropna=False)
    label_counts = groups["condition_label"].nunique()
    clean_texts = label_counts[label_counts == 1].index
    clean = raw[raw["normalized"].isin(clean_texts)].drop_duplicates("normalized").copy()
    clean["condition_label"] = clean["condition_label"].astype(int)
    duplicate_groups = int((groups.size() > 1).sum())
    conflict_groups = int((label_counts > 1).sum())
    cross_split_texts = set(train["medical_abstract"].astype(str).map(lambda s: " ".join(s.casefold().split()))) & set(
        test["medical_abstract"].astype(str).map(lambda s: " ".join(s.casefold().split()))
    )
    return raw, clean, {
        "raw_rows": int(len(raw)),
        "raw_train_rows": int(len(train)),
        "raw_test_rows": int(len(test)),
        "normalized_unique_texts": int(len(label_counts)),
        "duplicate_text_groups": duplicate_groups,
        "ambiguous_label_groups_excluded": conflict_groups,
        "same_label_duplicate_groups_collapsed": int(((groups.size() > 1) & (label_counts == 1)).sum()),
        "cross_split_normalized_texts_in_original_split": int(len(cross_split_texts)),
        "analysis_rows": int(len(clean)),
        "analysis_class_counts": {str(int(k)): int(v) for k, v in clean.condition_label.value_counts().sort_index().items()},
    }


def make_model(penalty: str) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=50_000,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("logit", OneVsRestClassifier(LogisticRegression(
            solver="liblinear",
            l1_ratio=1.0 if penalty == "l1" else 0.0,
            C=0.3,
            max_iter=2000,
            tol=1e-4,
            random_state=SEED,
        ))),
    ])


def metrics_for(model, X, y, labels):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    score = model.decision_function(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "macro_ovr_auc": float(np.mean([
            roc_auc_score((np.asarray(y) == cls).astype(int), score[:, j])
            for j, cls in enumerate(labels)
        ])),
        "log_loss": float(log_loss(y, proba, labels=labels)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "classification_report": classification_report(y, pred, labels=labels, target_names=[LABEL_NAMES[int(x)] for x in labels], output_dict=True, zero_division=0),
    }


def paired_stratified_bootstrap(y, pred_l1, pred_l2, labels, n_boot=2000):
    rng = np.random.default_rng(SEED + 1)
    y = np.asarray(y)
    pred_l1 = np.asarray(pred_l1)
    pred_l2 = np.asarray(pred_l2)
    by_class = [np.flatnonzero(y == c) for c in labels]
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in by_class])
        diffs[b] = f1_score(y[idx], pred_l1[idx], labels=labels, average="macro", zero_division=0) - f1_score(
            y[idx], pred_l2[idx], labels=labels, average="macro", zero_division=0
        )
    return {
        "n_bootstrap": n_boot,
        "difference_l1_minus_l2": float(f1_score(y, pred_l1, labels=labels, average="macro") - f1_score(y, pred_l2, labels=labels, average="macro")),
        "ci95_percentile": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
    }


def main():
    raw, clean, data_audit = load_unique_unambiguous()
    labels = np.array(sorted(LABEL_NAMES))
    X_train, X_test, y_train, y_test = train_test_split(
        clean.medical_abstract.astype(str).to_numpy(),
        clean.condition_label.to_numpy(),
        test_size=0.20,
        random_state=SEED,
        stratify=clean.condition_label.to_numpy(),
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fitted = {}
    tuning = {}
    warnings_seen = []
    for name, penalty in [("lasso_l1", "l1"), ("ridge_l2", "l2")]:
        search = GridSearchCV(
            make_model(penalty),
            {"logit__estimator__C": C_GRID},
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X_train, y_train)
        warnings_seen.extend([str(w.message) for w in caught])
        fitted[name] = search.best_estimator_
        tuning[name] = {
            "best_C": float(search.best_params_["logit__estimator__C"]),
            "best_train_cv_macro_f1": float(search.best_score_),
            "cv_results": [
                {"C": float(p["logit__estimator__C"]), "mean_macro_f1": float(m), "std_macro_f1": float(s)}
                for p, m, s in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"])
            ],
        }

    scores = {name: metrics_for(model, X_test, y_test, labels) for name, model in fitted.items()}
    pred_l1 = fitted["lasso_l1"].predict(X_test)
    pred_l2 = fitted["ridge_l2"].predict(X_test)
    comparison_ci = paired_stratified_bootstrap(y_test, pred_l1, pred_l2, labels)

    l1_model = fitted["lasso_l1"].named_steps["logit"]
    feature_names = fitted["lasso_l1"].named_steps["tfidf"].get_feature_names_out()
    coef = np.vstack([est.coef_[0] for est in l1_model.estimators_])
    selected_per_class = {}
    top_terms = {}
    for i, class_id in enumerate(l1_model.classes_):
        selected = np.flatnonzero(coef[i] != 0)
        selected_per_class[str(int(class_id))] = int(len(selected))
        positive = np.argsort(coef[i])[-12:][::-1]
        top_terms[str(int(class_id))] = [
            {"term": str(feature_names[j]), "coefficient": float(coef[i, j])}
            for j in positive if coef[i, j] > 0
        ]
    l1_nonzero = int(np.count_nonzero(coef))
    l1_total = int(coef.size)

    # Measure vocabulary selection on the same 3 training folds used for tuning;
    # this is training-only and leaves the held-out test split untouched.
    fold_sets = []
    fold_nonzero_by_class = []
    selected_C = tuning["lasso_l1"]["best_C"]
    for train_idx, _ in cv.split(X_train, y_train):
        model = make_model("l1").set_params(logit__estimator__C=selected_C)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train[train_idx], y_train[train_idx])
        warnings_seen.extend([str(w.message) for w in caught])
        names = model.named_steps["tfidf"].get_feature_names_out()
        coefs = np.vstack([est.coef_[0] for est in model.named_steps["logit"].estimators_])
        union = set()
        counts = {}
        for i, cls in enumerate(model.named_steps["logit"].classes_):
            idx = set(np.flatnonzero(coefs[i] != 0).tolist())
            union.update(idx)
            counts[str(int(cls))] = int(len(idx))
        fold_sets.append({str(names[j]) for j in union})
        fold_nonzero_by_class.append(counts)
    jaccards = []
    for i in range(len(fold_sets)):
        for j in range(i + 1, len(fold_sets)):
            union = fold_sets[i] | fold_sets[j]
            jaccards.append(len(fold_sets[i] & fold_sets[j]) / len(union) if union else 1.0)

    results = {
        "forecast_id": "T3v3-2026-0041",
        "forecast": "LASSO regression -> Health Informatics",
        "study": "L1 logistic regression on medical abstract text classification",
        "seed": SEED,
        "data_audit": data_audit,
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "test_fraction": 0.20, "stratified": True},
        "preprocessing": {"normalization": "casefold and whitespace collapse for deduplication only", "tfidf": "unigrams+bigrams, min_df=2, max_df=0.98, max_features=50000, sublinear_tf=True, no stop-word removal", "solver": "liblinear one-vs-rest", "C_grid": C_GRID, "training_cv": "3-fold stratified, seed 20261107, macro-F1"},
        "tuning": tuning,
        "test_metrics": scores,
        "paired_macro_f1_bootstrap": comparison_ci,
        "lasso_sparsity": {"nonzero_coefficients_all_classes": l1_nonzero, "total_coefficients": l1_total, "fraction_nonzero": float(l1_nonzero / l1_total), "nonzero_features_per_class": selected_per_class},
        "lasso_selected_vocabulary_stability": {"fold_nonzero_features_per_class": fold_nonzero_by_class, "mean_pairwise_jaccard_union_vocabulary": float(np.mean(jaccards)), "pairwise_jaccard": [float(x) for x in jaccards], "note": "three training folds; overlap describes feature selection, not causal stability"},
        "lasso_top_positive_terms_by_class": top_terms,
        "warnings": sorted(set(warnings_seen)),
        "interpretation_limit": "Curated broad-topic abstract classification only; not patient diagnosis or clinical decision support.",
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.DataFrame([
        {"model": name, **{k: v[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_ovr_auc", "log_loss"]}}
        for name, v in scores.items()
    ]).to_csv(HERE / "metrics_holdout.csv", index=False)
    confusion_rows = []
    for name, values in scores.items():
        for i, actual in enumerate(labels):
            for j, predicted in enumerate(labels):
                confusion_rows.append({"model": name, "actual": LABEL_NAMES[int(actual)], "predicted": LABEL_NAMES[int(predicted)], "count": values["confusion_matrix"][i][j]})
    pd.DataFrame(confusion_rows).to_csv(HERE / "confusion_matrices.csv", index=False)
    selected_rows = []
    for i, class_id in enumerate(l1_model.classes_):
        for j in np.flatnonzero(coef[i] != 0):
            selected_rows.append({"class_id": int(class_id), "class_name": LABEL_NAMES[int(class_id)], "term": str(feature_names[j]), "coefficient": float(coef[i, j])})
    pd.DataFrame(selected_rows).sort_values(["class_id", "coefficient"], ascending=[True, False]).to_csv(HERE / "lasso_selections.csv", index=False)
    cv_rows = []
    for model_name, detail in tuning.items():
        for row in detail["cv_results"]:
            cv_rows.append({"model": model_name, **row})
    pd.DataFrame(cv_rows).to_csv(HERE / "metrics_cv.csv", index=False)

    # Figures intentionally show held-out scores and the distinct sparsity profile.
    metric_names = ["macro_f1", "accuracy", "balanced_accuracy"]
    model_names = ["lasso_l1", "ridge_l2"]
    display = ["LASSO (L1)", "Ridge (L2)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(metric_names))
    width = 0.34
    for k, (name, label) in enumerate(zip(model_names, display)):
        vals = [scores[name][m] for m in metric_names]
        axes[0].bar(x + (k - 0.5) * width, vals, width, label=label)
    axes[0].set_xticks(x, ["Macro F1", "Accuracy", "Balanced\naccuracy"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Held-out score")
    axes[0].set_title("Predictive performance")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    cls_ids = list(l1_model.classes_)
    cls_labels = [LABEL_NAMES[int(c)] for c in cls_ids]
    counts = [selected_per_class[str(int(c))] for c in cls_ids]
    axes[1].barh(cls_labels, counts, color="#3b7898")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Non-zero terms")
    axes[1].set_title("LASSO vocabulary by class")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(HERE.parent / "fig_results.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"data_audit": data_audit, "tuning": tuning, "test_metrics": {k: {m: round(v[m], 4) for m in ["macro_f1", "accuracy", "balanced_accuracy", "macro_ovr_auc"]} for k, v in scores.items()}, "bootstrap": comparison_ci, "lasso_sparsity": results["lasso_sparsity"], "stability": results["lasso_selected_vocabulary_stability"], "warnings": results["warnings"]}, indent=2))


if __name__ == "__main__":
    main()


