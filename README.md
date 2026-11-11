# Sparse L1 classification of medical abstracts after removing duplicated and conflicting texts

From a v3 forecast: LASSO beats ridge on medical abstracts using under 1% of the coefficients.

**Paper:** [paper.pdf](paper.pdf) · **Page:** [singularitysd.com/research/medical-text.html](https://singularitysd.com/research/medical-text.html) · Preprint, not yet peer reviewed.

## Abstract

We ask whether an L1-penalised (LASSO) linear classifier gives a compact and accurate representation of medical abstracts, a genuinely wide and sparse input. An audit of a public five-class corpus of 14,438 abstracts found 988 normalised texts shared between its supplied training and test files and 2,929 groups of identical texts, all with conflicting labels. Keeping only unique, unambiguous abstracts left 8,298, which we split once (80/20). On the 1,660 held-out abstracts, LASSO reached a macro-F1 of 0.791 against 0.758 for ridge (L2) regression with the same features and tuning; the paired difference was 0.033 (class-stratified bootstrap 95% interval 0.014 to 0.053). LASSO used 2,201 of 250,000 class-specific coefficients (0.9%), a few hundred readable terms per class. Probability ranking was similar (macro AUC 0.952 against 0.956), so the gain lies in the class decisions.

## How it was made

This study started from the Singularity forecast **T3v3-2026-0041**: a method predicted to enter a new field. It was then tested once, with the plan fixed before looking at the results.

## What is in this repository

| file | what it is |
|---|---|
| `code/` | full code, protocol and intermediate results |
| `experiment.py` | reproduces the study end to end |
| `fig_results.png` | figure |
| `main.tex` | LaTeX source of the paper |
| `paper.pdf` | the paper |
| `results.json` | every number reported in the paper |

## Reproduce

```bash
pip install -r code/requirements.txt
python experiment.py
```

It rewrites the results and figures.

## Cite

```bibtex
@misc{singularity_lasso_medical_abstracts_2026,
  title  = {Sparse L1 classification of medical abstracts after removing duplicated and conflicting texts},
  author = {Singularity},
  year   = {2026},
  url    = {https://github.com/singularity-sd/lasso-medical-abstracts}
}
```

## License

Code: MIT (`LICENSE`). Paper, text and figures: CC BY 4.0 (`LICENSE-paper.md`). Third-party data keep their own licence where stated.

---

Part of [Singularity](https://singularitysd.com): forecasting where science goes next, then testing it.
