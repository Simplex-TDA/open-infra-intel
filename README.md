# Open Infrastructure Intelligence for Malicious Domain Detection

An exploratory, fully reproducible replication of the MANTIS malicious-domain
detection pipeline (IEEE S&P 2025) built entirely on **free, public data** —
Active DNS resolution, [Phishing.Database](https://phishing.database.aa419.org/),
[Tranco](https://tranco-list.eu/), and [Team Cymru](https://team-cymru.com/)
bulk ASN whois — instead of the paper's paid Farsight Passive DNS and
VirusTotal Enterprise feeds. On top of the paper's own lexical+hosting
feature set and GraphSAGE GNN encoder, this project adds topological features
(Ollivier-Ricci curvature + persistent homology computed over each domain's
local hosting neighborhood) and asks a narrow, testable question: **does
adding topological data analysis (TDA) features improve malicious-domain
classification?**

This repository is a grant-application artifact. It is explicitly an
*exploratory* project, not a claim of parity with the original paper — see
[Scale disclosure](#scale-disclosure) below for why that distinction matters.

## What this is

[MANTIS](https://arxiv.org/abs/2502.03255) (Rahman et al., IEEE S&P 2025)
detects malicious domains from the "hosting graph" — the bipartite/multipartite
structure of which domains resolve to which IPs, which IPs sit in which
subnets, and which subnets belong to which ASN — plus lexical features of the
domain name itself. The paper's Table 4 ablates a lexical+hosting feature set
against a semi-supervised GraphSAGE GNN that also sees graph structure
directly.

This project mirrors that ablation methodology on a differently-sourced
dataset, and extends it with a third variant: lexical+hosting features plus
TDA features extracted from each domain's local hosting neighborhood (its
"ego graph" out to a fixed radius). The TDA features summarize two
structural properties of that neighborhood that raw hosting features (degree
counts, ASN membership, etc.) don't directly capture:

- **Ollivier-Ricci curvature** — a measure of how tightly connected a node's
  neighborhood is (negative curvature ~ hub/tree-like structure; positive ~
  clustered/redundant structure).
- **Persistent homology** — the shape of the neighborhood's connected
  components and cycles across a filtration, summarized as total persistence
  and cycle ("H1") counts.

The hypothesis: malicious infrastructure is deliberately provisioned (bulk
registration, shared bulletproof hosting, fast-flux rotation), which may
leave a different topological signature in its hosting neighborhood than
organically-grown benign infrastructure — a signature that's cheaper to
compute per-domain than training a full GNN.

## The key scoping decision: Active DNS instead of Passive DNS

The original MANTIS paper uses Farsight Passive DNS (PDNS) — a paid feed that
records every DNS resolution ever observed across Farsight's global sensor
network, going back years. This lets the paper's hosting graph include every
domain ever seen sharing an IP with a seed domain, even domains that resolved
there yesterday and moved today.

This project instead uses **Active DNS**: direct, on-demand resolution of a
fixed, known seed set of malicious and benign domains. This is a deliberate
feasibility choice, not an oversight — Farsight PDNS costs thousands of
dollars a month and VirusTotal Enterprise is similarly gated; Active DNS is
free and requires nothing but network access. The tradeoff is explicit:
"shared infrastructure" edges in this project's hosting graph are limited to
co-occurrences *within the seed set itself*, not a full passive crawl of
everything ever hosted on an IP. That's a real signal, just a weaker and
narrower one than the paper's.

This tradeoff is also the project's main selling point for a grant
application: **the entire pipeline is reproducible by anyone with no data
budget.** No paid feed, no special access, no institutional agreement — just
public lists and DNS queries.

## Headline result: TDA features raise AUC, consistently, across every model tested

Across four model/configuration combinations — Random Forest, XGBoost, and a
GraphSAGE GNN at two training budgets — adding TDA features to lexical+hosting
features **raised mean AUC in all four**, with the clearest, most individually
significant effect in XGBoost:

| Model | config | mean ΔAUC | p-value (paired t-test) | seeds improved |
|---|---|---|---|---|
| XGBoost | 10-seed | **+0.004** | **p < 0.0001** | 10/10 |
| Random Forest | 10-seed | +0.001 | p = 0.175 | 6/10 |
| GNN (GraphSAGE) | epochs=200 (paper's value) | +0.032 | p = 0.083 | 6/10 |
| GNN (GraphSAGE) | epochs=800 (confirmation run) | +0.022 | p = 0.106 | 8/10 |

The honest framing: this is **"consistently positive, individually
significant only for the strongest/best-performing model (XGBoost),"** not
"TDA improves every model significantly." For the GNN specifically, the more
statistically robust effect of adding TDA features is actually a
precision/false-positive-rate trade-off against recall, not the AUC shift
itself — see [docs/RESULTS.md](docs/RESULTS.md) for the full per-metric
tables and discussion.

## Scale disclosure

The hosting graphs built here top out around 15-20K nodes (seed count ×
resolved IPs/subnets/ASNs). The MANTIS paper's own PDNS-based graphs are
700K-900K nodes — 50-300x larger, because passive DNS surfaces every domain
ever co-hosted, not just the seed set. This is a structural consequence of
the active-vs-passive DNS choice, not something closeable by more compute or
hyperparameter tuning (confirmed directly: a grid search over GNN training
epochs/weight-decay that appeared to find a config-dependent effect did not
replicate under a clean 10-seed re-run — see docs/RESULTS.md for that
worked example). A reviewer should read all AUC/significance numbers above
with this scale gap in mind.

## Reproducibility

Everything needed to regenerate every number in this README and in
[docs/RESULTS.md](docs/RESULTS.md) is in this repository: the data-fetching
code (`src/mantis/ingest.py`), graph construction (`src/mantis/graph.py`),
feature extraction including the vendored TDA code (`src/mantis/classify.py`,
`src/mantis/topo_features.py`, `src/mantis/_topology/`), the GNN
(`src/mantis/gnn.py`), and the exact driver scripts used to produce the
headline table above (`experiments/significance_classify.py`,
`experiments/significance_gnn.py`). See [docs/REPRODUCING.md](docs/REPRODUCING.md)
for exact setup and run instructions, including how to run against the
bundled sample feature table (no DNS resolution needed, runs in minutes) vs.
a full from-scratch rebuild (involves live DNS queries, takes longer).

## Documentation

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — data pipeline, graph
  construction, feature sets, and how this maps onto the MANTIS paper's own
  methodology.
- [docs/RESULTS.md](docs/RESULTS.md) — full 10-seed result tables for
  RF/XGBoost/GNN, including the honest trade-off discussion and the
  grid-search non-replication lesson.
- [docs/REPRODUCING.md](docs/REPRODUCING.md) — exact setup and run
  instructions, expected runtime.

## License and citation

This code is released under the [MIT License](LICENSE).

This project replicates and extends methodology from:

> M. Rahman et al., "MANTIS: Detecting Malicious Domains via Graph-based
> Semi-supervised Learning," *IEEE Symposium on Security and Privacy (S&P)*,
> 2025.

The original paper (`MantisIEEE.pdf`) is not bundled here as it is the
authors' copyrighted work; please obtain it through IEEE or the authors'
own channels.
