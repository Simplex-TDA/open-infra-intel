# Methodology

## 1. Data sources (all free, public)

| Role | Source | Notes |
|---|---|---|
| Malicious seed domains | [Phishing.Database](https://github.com/mitchellkrogza/Phishing.Database) | Free, no-auth, continuously-updated aggregate of OpenPhish/PhishTank/etc. Apex domains only (`src/mantis/ingest.py`). |
| Benign seed domains | [Tranco top-1M](https://tranco-list.eu/) | Sampled with an offset skipping the very top ranks, so the benign set isn't pure global brands — the MANTIS paper makes the same call (Sec 4.1), to avoid a classifier that just learns "popular = benign." |
| Hosting resolution | Active DNS (`socket.gethostbyname_ex`) | On-demand resolution of the seed domains only — see [the Active vs Passive DNS tradeoff](#2-active-dns-vs-the-papers-passive-dns) below. |
| IP → subnet/ASN mapping | [Team Cymru](https://team-cymru.com/) bulk whois (`whois.cymru.com`, port 43) | One bulk round-trip for all resolved IPs, no auth, no key, no rate limit encountered. |

## 2. Active DNS vs. the paper's Passive DNS

MANTIS builds its hosting graph from Farsight Passive DNS (PDNS): a paid feed
recording DNS resolutions Farsight's global sensors have observed. This gives
the paper's graph much broader *observed* coverage of domains sharing an IP
with a seed domain than this project's active-DNS crawl can reach — a
sprawling, historically-accumulated "who else lives here" view. It is
observed coverage, not a guarantee of completeness (Farsight only sees what
its sensors are positioned to see), and MANTIS itself bounds how far its own
graph expansion goes rather than crawling without limit.

This project has no PDNS access, so it resolves only the fixed seed set
(malicious + benign domains) via direct Active DNS queries. The hosting
graph's "shared infrastructure" edges are therefore limited to domains that
are *both in the seed set and* resolve to the same IP/subnet/ASN — a real but
narrower signal than the paper's passive crawl.

This is the project's central scoping decision, made deliberately for
reproducibility: anyone can run this pipeline with zero data budget. There is
no equivalent free substitute for PDNS's historical breadth; the honest
response is to disclose the resulting scale gap (see
[RESULTS.md's scale section](RESULTS.md#scale-disclosure-and-the-grid-search-lesson))
rather than pretend it's not there.

## 3. Hosting graph construction

`src/mantis/graph.py` builds a multipartite graph: `domain —resolves→ ip
—within→ subnet —within→ asn`, matching MANTIS Figure 5 minus the apex/
subdomain split and domain-domain "similar" edges (both need PDNS-scale data
this project doesn't have).

One deliberate deviation from a naive implementation: hyperscale CDN/cloud
ASNs (Cloudflare, AWS, Google Cloud, Azure, Akamai, etc.) are *scoped
per-domain* rather than shared as single hub nodes. Verified directly against
live data: AS13335 (Cloudflare) alone touched 44% of resolved domains in one
pull. Letting such ASNs be shared hubs wires most of the dataset together on
pure infrastructure-provider coincidence, not real shared hosting. Dedicated/
boutique hosting providers are left as genuine shared hubs — co-location there
is still real signal.

## 4. Feature sets (mirrors the paper's Table 4 ablation)

For each domain, three feature groups are available (`src/mantis/features.py`,
`src/mantis/classify.py`, `src/mantis/topo_features.py`):

1. **Lexical features** — domain name length, character entropy, digit/hyphen
   ratios, TLD, and similar string-level features (`lexical_features`).
2. **Hosting features** — degree/centrality-style features computed directly
   on the hosting graph (number of resolved IPs, subnet/ASN membership
   counts, etc.) (`hosting_features`).
3. **TDA features** — computed over each domain's local hosting neighborhood
   ("ego graph," `domain_ego_graph`, radius=3 by default, capped to the
   top-50 PageRank nodes for tractability at scale):
   - **Ollivier-Ricci curvature** per edge, summarized (mean/std/etc.) over
     the neighborhood (vendored from `GraphRicciCurvature`, `src/mantis/_topology/curvature.py`).
   - **Persistent homology** (H0/H1 total persistence, H1 cycle count) over
     a distance matrix built from the neighborhood (`ripser`,
     `src/mantis/_topology/persistence.py`).
   - Both a "full" and a "shape-only" variant are tracked — shape-only drops
     pure neighborhood-size columns (node/edge counts, diagram cardinality)
     to isolate the size-independent topological signal from a domain simply
     having a larger or smaller neighborhood.

Two model families are compared against these feature sets, following the
paper's own ablation structure:

- **Classical ML** (Random Forest, XGBoost) trained on flat feature vectors
  — lexical+hosting alone vs. lexical+hosting+TDA (`src/mantis/classify.py`).
- **GraphSAGE GNN** (`src/mantis/gnn.py`), an independent implementation
  modeled on the paper's own semi-supervised graph neural network, trained
  transductively over the whole hosting graph with per-domain feature
  vectors as node inputs — again lexical+hosting-only inputs vs.
  lexical+hosting+TDA inputs.

Note: the combined TDA feature block mixes three distinct signals —
curvature, persistence, and raw neighborhood-size counts (`_TOPO_SIZE_COLS`
in `src/mantis/classify.py`). The "full" vs. "shape-only" split above exists
to let someone isolate them, but that ablation hasn't actually been run and
reported yet (see [RESULTS.md](RESULTS.md)), so any AUC gain attributed to
"TDA" in this project's results should currently be read as coming from the
combined topological+structural block, not specifically from persistent
homology or curvature alone.

## 5. Paired evaluation design

For a fixed random seed, `sklearn.train_test_split`'s resulting split depends
only on the label vector and the seed — not on which feature columns are
selected. This means that for a given seed, the no-TDA and +TDA runs are
evaluated on the *exact same* train/test split, making every no-TDA-vs-+TDA
comparison a **paired** comparison across seeds (the same statistical design
used for both the classical-ML and GNN tracks). This is why results are
reported as per-seed deltas with a paired t-test, not independent-sample
statistics — see [RESULTS.md](RESULTS.md).
