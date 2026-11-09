# Reproducing these results

## Setup

Requires Python 3.10-3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`networkit`, `GraphRicciCurvature`, and `ripser` (the TDA dependencies) are
pure pip installs, no system packages or compiled toolchains needed on
macOS/Linux.

Verify the install:

```bash
pytest
```

## Fast path: using the bundled sample feature table + graph

Both significance scripts default to loading a matched pair of bundled
files: `experiments/sample_data/features_m3000_b3000_skip2000_seed0_r3_n3977_topk50.csv`
and, for the GNN script, `graph_m3000_b3000_skip2000_seed0_r3_n3977_topk50.pkl.gz`
— a feature table and its exact hosting graph, built together in one pass
from 3000 malicious + 3000 benign seed domains (3977 resolved), radius=3 ego
graphs, top-50 PageRank nodes. Loading the pair this way skips the
active-DNS resolution, Team Cymru ASN lookups, and TDA feature computation
(the expensive steps, tens of minutes total) entirely — **and**, because the
graph is loaded from the same frozen snapshot as the feature table rather
than re-fetched, guarantees the edges/labels the GNN trains on actually match
the features in the CSV.

(The bundled snapshot itself was produced by `experiments/freeze_sample_data.py`,
which fetches/resolves one domain set and builds the graph and feature table
from that same graph in a single pass, so they can never drift apart. Re-run
it yourself — `python experiments/freeze_sample_data.py` — to regenerate a
fresh matched snapshot under a new filename key if you want one.)

```bash
# Random Forest / XGBoost, 10-seed paired comparison (~30-60s)
python experiments/significance_classify.py --n-seeds 10

# GraphSAGE GNN, 10-seed paired comparison (~30 min on CPU, 2 processes)
python experiments/significance_gnn.py --n-seeds 10 --epochs 200

# GNN robustness confirmation at a longer training budget (~2x+ as long)
python experiments/significance_gnn.py --n-seeds 10 --epochs 800
```

Both scripts print per-metric paired statistics (mean values, delta, paired
t-test p-value, and seeds-improved sign-test count) matching the tables in
[RESULTS.md](RESULTS.md).

## Full path: rebuilding from scratch (live DNS)

Pass `--rebuild` to either script to re-fetch domain lists, re-resolve via
active DNS, and rebuild the hosting graph + feature table from zero, instead
of loading the bundled CSV:

```bash
python experiments/significance_classify.py --n-seeds 10 --rebuild
python experiments/significance_gnn.py --n-seeds 10 --epochs 200 --rebuild
```

This takes materially longer (DNS resolution of ~6000 domains, Team Cymru
whois lookups, then per-domain curvature + persistence computation) — the
build step alone took ~21 minutes when this project's bundled snapshot was
generated, on top of whichever script's own training/evaluation time (see
the runtime table below) — and results will differ slightly from the bundled
table since Phishing.Database's live feed changes over time.
`experiments/mantis_poc.py` and
`experiments/mantis_gnn.py` expose the same fetch/resolve/build pipeline with
more granular CLI flags (`--malicious-n`, `--benign-n`, `--radius`,
`--top-k-centrality`, etc.) if you want to sweep those parameters directly.

## A note on GNN parallelism

`experiments/significance_gnn.py` runs its 20 (2 variants × 10 seeds) train/
eval jobs through a small `multiprocessing.Pool`, defaulting to **2
processes** (`--n-processes` to override). This default is deliberately
conservative: an earlier run of this project's GNN significance testing at
higher parallelism triggered a thermal shutdown on a laptop. Each individual
run is short (a few seconds to a couple minutes depending on `--epochs`), so
2 processes is already a meaningful speedup without pushing sustained load
on a non-dedicated machine.

## Expected runtime summary

Measured on a 12-core 2026 machine, `significance_gnn.py` at its default
`--n-processes 2`. The epochs=800 fast-path figure is extrapolated from the
measured epochs=200 run (training time scales roughly linearly with
`--epochs`), not independently timed — treat it as an estimate, not a
measurement.

| Script | Fast path (bundled CSV+graph) | Full rebuild |
|---|---|---|
| `significance_classify.py` | ~30-60s (measured ~25s) | ~20-25 min (dominated by DNS/TDA build) |
| `significance_gnn.py --epochs 200` | ~30 min (measured) | ~50-55 min |
| `significance_gnn.py --epochs 800` | ~60-120 min (estimated) | ~80-145 min |

These are well past the "a few minutes" impression a reader might get from
the script names alone — the GNN significance scripts are the slow part of
this project's reproducibility story, not the DNS/feature-build step.
