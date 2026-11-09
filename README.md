# Open Infrastructure Intelligence for Malicious Domain Detection

An exploratory, fully reproducible **MANTIS-inspired active-DNS
implementation** of a malicious-domain detection pipeline, built entirely on
**free, public data** — Active DNS resolution,
[Phishing.Database](https://phishing.database.aa419.org/),
[Tranco](https://tranco-list.eu/), and [Team Cymru](https://team-cymru.com/)
bulk ASN whois — instead of the paper's paid Farsight Passive DNS and
VirusTotal Enterprise feeds. On top of a lexical+hosting feature set and
GraphSAGE GNN encoder modeled on the paper's own design, this project adds
topological features (Ollivier-Ricci curvature + persistent homology computed
over each domain's local hosting neighborhood) and asks a narrow, testable
question: **does adding topological data analysis (TDA) features improve
malicious-domain classification?**

This repository is a grant-application artifact. Given a different data
source, independently-built graph construction, and an independent model
implementation, it is explicitly framed as **MANTIS-inspired**, not a direct
reproduction of or a claim of parity with the original paper — see
[Scale disclosure](#scale-disclosure) below for why that distinction matters.

## What this is

[MANTIS](https://arxiv.org/abs/2502.09788) (Fatih Deniz, Mohamed Nabeel, Ting
Yu, and Issa Khalil, *IEEE Symposium on Security and Privacy*, 2025) detects
malicious domains from the "hosting graph" — the bipartite/multipartite
structure of which domains resolve to which IPs, which IPs sit in which
subnets, and which subnets belong to which ASN — plus lexical features of the
domain name itself. The paper's Table 4 ablates a lexical+hosting feature set
against a semi-supervised GraphSAGE GNN that also sees graph structure
directly.

This project takes inspiration from that ablation methodology on a
differently-sourced dataset, with its own graph construction and model
implementation, and extends it with a third variant: lexical+hosting features
plus topological features extracted from each domain's local hosting
neighborhood (its "ego graph" out to a fixed radius). Those topological
features summarize two structural properties of that neighborhood that raw
hosting features (degree counts, ASN membership, etc.) don't directly
capture:

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

The original MANTIS paper uses Farsight Passive DNS (PDNS) — a paid feed
recording DNS resolutions observed across Farsight's global sensor network,
going back years. This gives the paper's hosting graph much broader
*observed* coverage of domains that have shared an IP with a seed domain than
an active-DNS crawl of a fixed seed set can — though it is coverage, not
completeness: Farsight's sensors only see what resolvers route through them,
and MANTIS itself bounds how far it expands the graph rather than crawling
without limit.

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

## Headline result: topological+structural features raise AUC, consistently, across every model tested

Across four model/configuration combinations — Random Forest, XGBoost, and a
GraphSAGE GNN at two training budgets — adding the combined topological and
structural feature block (curvature + persistent homology + neighborhood-size
features; see [Scale disclosure](#scale-disclosure) and
[docs/RESULTS.md](docs/RESULTS.md) for why these aren't yet credited
separately) to lexical+hosting features **raised mean AUC in all four**, with
the clearest, most individually significant effect in XGBoost:

| Model | config | mean ΔAUC | p-value (paired t-test) | seeds improved |
|---|---|---|---|---|
| XGBoost | 10-seed | **+0.003** | **p = 0.0001** | 10/10 |
| Random Forest | 10-seed | +0.001 | p = 0.079 | 7/10 |
| GNN (GraphSAGE) | epochs=200 (paper's value) | +0.088 | p = 0.180 | 8/10 |
| GNN (GraphSAGE) | epochs=800 (confirmation run) | +0.016 | p = 0.104 | 8/10 |

The honest framing: this is **"consistently positive, individually
significant only for the strongest/best-performing model (XGBoost's AUC and,
separately, Random Forest's recall),"** not "TDA improves every model
significantly." The GNN's AUC gain is directionally the largest of any model
at epochs=200, but with far higher seed-to-seed variability — it does not
reach significance at either training budget tested, and neither does any
other GNN metric — see [docs/RESULTS.md](docs/RESULTS.md) for the full
per-metric tables and discussion.

## Scale disclosure

The bundled hosting graph has 16,860 nodes total (3977 domain + 4186 IP +
5188 subnet + 3509 ASN) — note this is *graph nodes*, not labeled domains;
see [docs/RESULTS.md](docs/RESULTS.md) for the actual labeled-domain and
test-set counts. The MANTIS paper's own PDNS-based graphs are 700K-900K
nodes, roughly **40-55x larger**. Passive DNS gives Farsight's sensors much
broader *observed* coverage of co-hosted domains than an active-DNS seed-set
crawl can — it is not literally every domain ever co-hosted (MANTIS itself
also bounds its own graph expansion; it isn't an unlimited crawl either).
This scale gap is a plausible, but not demonstrated, explanation for part of
the performance gap between this project and the paper's reported numbers.
A 3-seed grid search over GNN training epochs/weight-decay appeared to find a
stronger, config-dependent AUC effect; that did not replicate under a clean
10-seed re-run (see docs/RESULTS.md for that worked example). That
non-replication shows the original 3-seed grid-search finding wasn't
reliable — it does not, by itself, demonstrate that graph scale is what
explains the remaining gap. A reviewer should read all AUC/significance
numbers above with this scale gap in mind, as an open question rather than a
settled explanation.

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

This project takes methodological inspiration from:

> Fatih Deniz, Mohamed Nabeel, Ting Yu, and Issa Khalil, "MANTIS: Detection of
> Zero-Day Malicious Domains Leveraging Low Reputed Hosting Infrastructure,"
> *IEEE Symposium on Security and Privacy (S&P)*, 2025.
> [arXiv:2502.09788](https://arxiv.org/abs/2502.09788)

The original paper (`MantisIEEE.pdf`) is not bundled here as it is the
authors' copyrighted work; please obtain it through IEEE or the authors'
own channels.
