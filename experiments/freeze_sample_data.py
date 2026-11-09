"""Generates the frozen fast-path snapshot used by significance_classify.py
and significance_gnn.py: fetches+resolves one domain set, builds the hosting
graph and feature table from that *same* graph, and writes both to
experiments/sample_data/ under a matching filename key. The graph and the
feature table are never built independently of each other, so the fast path
of the significance scripts can load both and be guaranteed consistent --
this is what makes "the fast path needs no DNS resolution" true.

Usage: python experiments/freeze_sample_data.py
"""
from __future__ import annotations

import argparse
import gzip
import pathlib
import pickle

from mantis.classify import build_feature_table
from mantis.graph import build_hosting_graph, graph_summary
from mantis.ingest import cymru_bulk_asn, fetch_phishing_db_domains, fetch_tranco_domains, resolve_many

SAMPLE_DIR = pathlib.Path(__file__).parent / "sample_data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--malicious-n", type=int, default=3000)
    parser.add_argument("--benign-n", type=int, default=3000)
    parser.add_argument("--benign-skip", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--top-k-centrality", type=int, default=50)
    args = parser.parse_args()

    print(f"Fetching {args.malicious_n} malicious + {args.benign_n} benign domains...", flush=True)
    malicious = fetch_phishing_db_domains(n=args.malicious_n, seed=args.seed)
    benign = fetch_tranco_domains(n=args.benign_n, skip=args.benign_skip)
    labels = {d: 1 for d in malicious} | {d: 0 for d in benign}
    domains = list(labels)

    print(f"Resolving {len(domains)} domains via active DNS...", flush=True)
    domain_ips = resolve_many(domains)
    print(f"  {len(domain_ips)}/{len(domains)} resolved", flush=True)

    all_ips = sorted({ip for ips in domain_ips.values() for ip in ips})
    print(f"Resolving {len(all_ips)} IPs to ASN/subnet via Team Cymru...", flush=True)
    ip_info = cymru_bulk_asn(all_ips, timeout=60.0)
    g = build_hosting_graph(domain_ips, ip_info, labels=labels)
    print("Hosting graph:", graph_summary(g), flush=True)

    resolved_domains = [d for d in domains if d in domain_ips]
    n = len(resolved_domains)
    key = (
        f"m{args.malicious_n}_b{args.benign_n}_skip{args.benign_skip}_seed{args.seed}"
        f"_r{args.radius}_n{n}_topk{args.top_k_centrality}"
    )

    print(f"Building feature table for {n} resolved domains...", flush=True)
    df = build_feature_table(
        resolved_domains, labels, g, radius=args.radius, top_k_centrality=args.top_k_centrality, progress_every=200
    )

    SAMPLE_DIR.mkdir(exist_ok=True)
    csv_path = SAMPLE_DIR / f"features_{key}.csv"
    graph_path = SAMPLE_DIR / f"graph_{key}.pkl.gz"
    df.to_csv(csv_path, index=False)
    with gzip.open(graph_path, "wb") as f:
        pickle.dump(g, f)
    print(f"\nWrote {csv_path.name} and {graph_path.name} -- bundle both, they are one snapshot.")


if __name__ == "__main__":
    main()
