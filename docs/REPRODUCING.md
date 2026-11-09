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

## Fast path: using the bundled sample feature table

Both significance scripts default to loading
`experiments/sample_data/features_m3000_b3000_skip2000_seed0_r3_n3984_topk50.csv`
— a feature table already built from 3000 malicious + 3000 benign seed
domains (3984 resolved), radius=3 ego graphs, top-50 PageRank nodes. This
skips the active-DNS resolution and TDA feature computation (the two
expensive steps, tens of minutes total) entirely.

```bash
# Random Forest / XGBoost, 10-seed paired comparison (~1-2 minutes)
python experiments/significance_classify.py --n-seeds 10

# GraphSAGE GNN, 10-seed paired comparison (~5-10 minutes on CPU)
python experiments/significance_gnn.py --n-seeds 10 --epochs 200

# GNN robustness confirmation at a longer training budget
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
whois lookups, then per-domain curvature + persistence computation) — expect
on the order of 15-30 minutes depending on network conditions, and results
will differ slightly from the bundled table since Phishing.Database's live
feed changes over time. `experiments/mantis_poc.py` and
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

| Script | Fast path (bundled CSV) | Full rebuild |
|---|---|---|
| `significance_classify.py` | ~1-2 min | ~15-30 min (dominated by DNS/TDA build) |
| `significance_gnn.py --epochs 200` | ~5-10 min | ~20-35 min |
| `significance_gnn.py --epochs 800` | ~15-25 min | ~30-45 min |
