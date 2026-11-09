# Results

All tables below are **paired** comparisons: for each seed, the no-TDA and
+TDA variants are evaluated on the identical train/test split (see
[METHODOLOGY.md §5](METHODOLOGY.md#5-paired-evaluation-design)), so
`mean_delta`, the paired t-test, and the "seeds improved" sign-test count are
all computed seed-by-seed, not from independent samples.

Dataset for all three tables below: 3000 malicious (Phishing.Database) + 3000
benign (Tranco, skip top 2000) seed domains, 3984 resolved after active-DNS
lookup, radius=3 ego graphs capped to the top-50 PageRank nodes. This is the
exact feature table bundled at `experiments/sample_data/features_m3000_b3000_skip2000_seed0_r3_n3984_topk50.csv`,
so every number below is directly regenerable (see
[REPRODUCING.md](REPRODUCING.md)).

## Random Forest, 10 seeds

| metric | no_tda | tda | mean_delta | t | p | seeds improved |
|---|---|---|---|---|---|---|
| auc | 0.908 | 0.909 | +0.001 | — | 0.1752 | 6/10 |
| precision | — | — | -0.004 | — | 0.1129 | 5/10 |
| recall | — | — | +0.010 | — | 0.0002 | 10/10 |
| f1 | — | — | +0.003 | — | 0.0218 | 8/10 |
| fpr | — | — | +0.003 | — | 0.0754 | 3/10 |

AUC direction is positive but not individually significant. The one clearly
significant effect is recall (p=0.0002, 10/10 seeds improved) — TDA features
help Random Forest catch more malicious domains, at a small, marginally
significant cost in false-positive rate.

## XGBoost, 10 seeds — the strongest result in this project

| metric | no_tda | tda | mean_delta | t | p | seeds improved |
|---|---|---|---|---|---|---|
| auc | — | — | **+0.004** | — | **< 0.0001** | **10/10** |
| precision | — | — | -0.001 | — | 0.8338 | 5/10 |
| recall | — | — | +0.003 | — | 0.0381 | 7/10 |
| f1 | — | — | +0.002 | — | 0.2800 | 7/10 |
| fpr | — | — | +0.001 | — | 0.7986 | 3/10 |

AUC improves in **every single one of the 10 seeds**, with p<0.0001 — the
single most statistically robust finding across the whole project. Precision
and FPR are essentially unchanged (not significant); the gain is concentrated
in AUC and, more weakly, recall.

## GNN (GraphSAGE), 10 seeds, epochs=200 (the paper's own stated training budget)

| metric | mean_delta | p (paired t-test / sign-test) | seeds improved |
|---|---|---|---|
| auc | +0.032 | p = 0.083 | 6/10 |
| precision / fpr | — | sign-test p ≈ 0.011 | 9/10 |

AUC direction is strongly positive but falls short of the conventional 0.05
significance threshold at n=10 seeds. The more robust, individually
significant effect for the GNN is actually a **precision/false-positive-rate
improvement** (9/10 seeds, p≈0.011) traded against a small amount of recall —
a different kind of win than a pure AUC lift, and worth stating precisely
rather than folding into the AUC headline.

## GNN (GraphSAGE), 10 seeds, epochs=800 — robustness confirmation

| metric | mean_delta | p | seeds improved |
|---|---|---|---|
| auc | +0.022 | p = 0.106 | 8/10 |

Run as a follow-up robustness check after a 3-seed grid search over
epochs/weight-decay appeared to show a stronger, config-dependent AUC effect.
**That apparent finding did not replicate** under this clean 10-seed re-run —
the effect size and significance both came back closer to the epochs=200
result, not stronger. This is kept in the record deliberately as a worked
methodology lesson: a result built on 3 seeds (`grid_screen.py`'s original
run, not shipped in this repository) is not trustworthy evidence on its own,
even when it looks clean. The fix was simply to re-run at the standard 10-seed
budget before believing it.

## Cross-cutting summary

| Model / config | mean ΔAUC | AUC p-value | Individually significant? |
|---|---|---|---|
| Random Forest | +0.001 | 0.175 | No |
| XGBoost | **+0.004** | **<0.0001** | **Yes** |
| GNN, epochs=200 | +0.032 | 0.083 | No (AUC); yes for precision/FPR |
| GNN, epochs=800 | +0.022 | 0.106 | No |

**AUC direction is positive in all four model/configuration combinations
tested.** Magnitude and statistical significance vary by model, reaching
clear significance only for XGBoost. The honest claim this project supports
is: *adding TDA features consistently nudges AUC upward across model
families; the effect is strong and significant for the best-performing
model (XGBoost), directionally consistent but not individually significant
for Random Forest and the GNN.* This is not a claim that TDA improves every
model significantly, and it should not be read or cited as one.

## Scale disclosure and the grid-search lesson

This project's hosting graphs are ~15-20K nodes; the MANTIS paper's own
PDNS-based graphs are 700K-900K nodes (50-300x larger), a direct structural
consequence of active vs. passive DNS (see
[METHODOLOGY.md §2](METHODOLOGY.md#2-active-dns-vs-the-papers-passive-dns)).
This gap is not closeable by more compute, more seeds, or hyperparameter
tuning — the grid-search non-replication above is a concrete demonstration of
that boundary, not just a disclaimer.
