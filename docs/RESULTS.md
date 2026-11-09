# Results

All tables below are **paired** comparisons: for each seed, the no-TDA and
+TDA variants are evaluated on the identical train/test split (see
[METHODOLOGY.md §5](METHODOLOGY.md#5-paired-evaluation-design)), so
`mean_delta`, the paired t-test, and the "seeds improved" sign-test count are
all computed seed-by-seed, not from independent samples.

Dataset for all four tables below: 3000 malicious (Phishing.Database) + 3000
benign (Tranco, skip top 2000) seed domains, 3977 resolved after active-DNS
lookup, radius=3 ego graphs capped to the top-50 PageRank nodes — **2358
benign-labeled and 1619 malicious-labeled domains** (the hosting graph itself
has ~16.9K nodes total once IP/subnet/ASN nodes are counted, but only the
3977 domain nodes carry labels and are evaluated). This is the exact
feature table + hosting graph bundled at
`experiments/sample_data/features_m3000_b3000_skip2000_seed0_r3_n3977_topk50.csv`
and `experiments/sample_data/graph_m3000_b3000_skip2000_seed0_r3_n3977_topk50.pkl.gz`
(one matched snapshot — see [REPRODUCING.md](REPRODUCING.md)), so every
number below is directly regenerable. Each seed's evaluation uses a
stratified 70/30 train/test split of the 3977 labeled domains; at seed=0 that
is 2783 train (1650 benign / 1133 malicious) and 1194 test (708 benign / 486
malicious) — sizes shift slightly seed to seed but stay in that range.

Variability below is the **standard deviation of the per-seed paired delta**
(tda − no_tda), derived from the reported t-statistic and mean via
`sd = mean_delta · sqrt(10) / t`— consistent with the paired-per-seed design
used throughout, and not the same thing as the raw metric's spread across
seeds.

## Random Forest, 10 seeds

| metric | no_tda | tda | mean_delta | sd(delta) | t | p | seeds improved |
|---|---|---|---|---|---|---|---|
| auc | 0.965 | 0.966 | +0.001 | 0.0016 | 1.98 | 0.0786 | 7/10 |
| precision | 0.945 | 0.943 | -0.002 | 0.0036 | -1.76 | 0.1128 | 1/10 |
| recall | 0.872 | 0.876 | +0.003 | 0.0035 | 2.68 | 0.0250 | 7/10 |
| f1 | 0.907 | 0.908 | +0.001 | 0.0044 | 0.72 | 0.4897 | 5/10 |
| fpr | 0.035 | 0.036 | +0.002 | 0.0032 | 1.96 | 0.0811 | 1/10 |

AUC direction is positive but not individually significant (p=0.079). The
one clearly significant effect is recall (p=0.025, 7/10 seeds improved) — the
added topological+structural features help Random Forest catch more
malicious domains, at a small, borderline-significant cost in false-positive
rate (p=0.081).

## XGBoost, 10 seeds — the strongest result in this project

| metric | no_tda | tda | mean_delta | sd(delta) | t | p | seeds improved |
|---|---|---|---|---|---|---|---|
| auc | 0.965 | 0.968 | **+0.003** | 0.0014 | 6.74 | **0.0001** | **10/10** |
| precision | 0.936 | 0.935 | -0.001 | 0.0052 | -0.61 | 0.5590 | 4/10 |
| recall | 0.876 | 0.877 | +0.001 | 0.0047 | 0.67 | 0.5217 | 4/10 |
| f1 | 0.905 | 0.905 | -0.000 | ~0.004 | -0.16 | 0.8795 | 4/10 |
| fpr | 0.041 | 0.042 | +0.001 | 0.0054 | 0.59 | 0.5688 | 4/10 |

AUC improves in **every single one of the 10 seeds**, with p=0.0001 — the
single most statistically robust finding across the whole project. Precision,
recall, f1 and FPR are all essentially unchanged (not significant); the gain
is concentrated in AUC alone.

## GNN (GraphSAGE), 10 seeds, epochs=200 (the paper's own stated training budget)

| metric | no_tda | tda | mean_delta | sd(delta) | t | p | seeds improved |
|---|---|---|---|---|---|---|---|
| auc | 0.859 | 0.947 | +0.088 | 0.192 | 1.45 | 0.1804 | 8/10 |
| precision | 0.854 | 0.900 | +0.046 | 0.103 | 1.41 | 0.1911 | 8/10 |
| recall | 0.790 | 0.871 | +0.081 | 0.215 | 1.19 | 0.2661 | 7/10 |
| f1 | 0.789 | 0.883 | +0.094 | 0.196 | 1.52 | 0.1626 | 8/10 |
| fpr | 0.105 | 0.070 | -0.035 | 0.104 | -1.06 | 0.3161 | 7/10 |

AUC direction is strongly positive (0.859 → 0.947) but, with the much larger
seed-to-seed variability typical of GNN training, falls well short of the
conventional 0.05 significance threshold at n=10 seeds — and so does every
other metric here. An earlier draft of this document, based on a different
(superseded) data snapshot, claimed precision/FPR as a separately robust,
individually significant effect for the GNN; **that does not hold on this
snapshot** (precision p=0.19, fpr p=0.32). Read this table as: direction
consistently favorable, magnitude large, nothing here clears significance on
its own at this epoch budget.

## GNN (GraphSAGE), 10 seeds, epochs=800 — robustness follow-up

| metric | no_tda | tda | mean_delta | sd(delta) | t | p | seeds improved |
|---|---|---|---|---|---|---|---|
| auc | 0.946 | 0.961 | +0.016 | 0.028 | 1.81 | 0.1039 | 8/10 |
| precision | 0.920 | 0.949 | +0.029 | 0.070 | 1.31 | 0.2214 | 6/10 |
| recall | 0.878 | 0.872 | -0.006 | 0.029 | -0.66 | 0.5278 | 5/10 |
| f1 | 0.896 | 0.909 | +0.012 | 0.023 | 1.63 | 0.1378 | 8/10 |
| fpr | 0.057 | 0.032 | -0.025 | 0.058 | -1.36 | 0.2057 | 6/10 |

Originally run as a follow-up check after a 3-seed grid search over
epochs/weight-decay appeared to show a stronger, config-dependent AUC effect.
**That apparent finding did not replicate** under this clean 10-seed re-run —
the effect size and significance both came back closer to the epochs=200
result, not stronger (this is kept in the record deliberately as a worked
methodology lesson: a result built on 3 seeds, `grid_screen.py`'s original
run, not shipped in this repository, is not trustworthy evidence on its own,
even when it looks clean — the fix was simply to re-run at the standard
10-seed budget before believing it).

One notable pattern across the two GNN budgets: both the no_tda baseline AUC
(0.859 → 0.946) and the mean AUC delta (+0.088 → +0.016) move substantially
between epochs=200 and epochs=800 — the longer-trained baseline closes most
of the gap. That is consistent with, but does not on its own prove, the
hypothesis that part of the TDA-feature benefit observed at the paper's
stated epoch budget is a "better features compensate for an undertrained
model" effect rather than a persistent one. No metric is individually
significant at either budget on this snapshot, so this is offered as an
observation worth further runs, not a conclusion.

## Cross-cutting summary

| Model / config | mean ΔAUC | AUC p-value | Individually significant (any metric)? |
|---|---|---|---|
| Random Forest | +0.001 | 0.079 | Yes — recall (p=0.025) |
| XGBoost | **+0.003** | **0.0001** | **Yes — AUC (p=0.0001)** |
| GNN, epochs=200 | +0.088 | 0.180 | No |
| GNN, epochs=800 | +0.016 | 0.104 | No |

**AUC direction is positive in all four model/configuration combinations
tested.** Magnitude and statistical significance vary by model, reaching
clear significance only for XGBoost's AUC and Random Forest's recall. The
honest claim this project supports is: *adding the combined topological and
structural feature block (curvature + persistent homology + neighborhood-size
counts) consistently nudges AUC upward across model families; the effect is
strong and significant for the best-performing model (XGBoost), directionally
consistent but not individually significant for Random Forest's AUC or either
GNN configuration.* This is not a claim that TDA (specifically persistent
homology or curvature, as opposed to the neighborhood-size features bundled
alongside them) improves every model significantly, and it should not be read
or cited as one — see the note on attribution below.

### A note on attribution: "topological features," not yet "TDA" specifically

The feature block tested as `+tda` in every table above actually bundles
three distinct signals: Ollivier-Ricci curvature, persistent homology, and
plain neighborhood-size counts (`topo_ego_n_nodes`, `topo_ego_n_edges`,
`topo_diagram_cardinality` — see `_TOPO_SIZE_COLS` in
`src/mantis/classify.py`). `feature_columns(df, topo_mode="shape_only")`
already exists to drop the size columns and isolate curvature+persistence
alone, but that ablation has not been run and reported in this document.
Until it is, the gains above should be attributed to "added topological and
structural features" as a block, not specifically to persistent homology or
to TDA in the narrower sense.

## Scale disclosure and the grid-search lesson

This project's bundled hosting graph has 16,860 *nodes* — 3977 domain, 4186
IP, 5188 subnet, and 3509 ASN nodes combined, not labeled domains; see the
per-table counts above for the actual labeled-domain and test-set sizes. The
MANTIS paper's own PDNS-based graphs are 700K-900K nodes, roughly **40-55x
larger**, which
likely reflects active vs. passive DNS coverage (see
[METHODOLOGY.md §2](METHODOLOGY.md#2-active-dns-vs-the-papers-passive-dns)).

That scale gap is a plausible hypothesis for part of the difference between
this project's numbers and the paper's reported performance — it has not
been demonstrated to be the explanation. The grid-search non-replication
above shows that a 3-seed "finding" about epochs/weight-decay wasn't
trustworthy evidence; it does not, by itself, show that graph scale is the
cause of anything. Treat the scale gap as an open question flagged for the
reader, not a settled causal account.
