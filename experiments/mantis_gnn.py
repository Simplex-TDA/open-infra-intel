"""Reproduce the MANTIS paper's own GNN result (semi-supervised GraphSAGE,
Sec 4.4/5.1) on this replication's domain sets, then test whether adding
TDA topology features (mantis.topo_features) as extra domain-node inputs
improves it -- Task 2 of docs/CURRENT_PLAN.md.

Usage: python experiments/mantis_gnn.py --malicious-n 3000 --benign-n 3000
"""
from __future__ import annotations

import argparse

from mantis.classify import feature_columns
from mantis.gnn import build_node_features, train_and_evaluate
from mantis.graph import build_hosting_graph, graph_summary
from mantis.ingest import cymru_bulk_asn, fetch_phishing_db_domains, fetch_tranco_domains, resolve_many
from mantis_poc import _load_or_build_feature_table  # reuses mantis_poc's feature-table cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--malicious-n", type=int, default=3000)
    parser.add_argument("--benign-n", type=int, default=3000)
    parser.add_argument("--benign-skip", type=int, default=2000)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--top-k-centrality", type=int, default=50)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--max-degree", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=1, help="average metrics over this many train/test splits")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    print(f"Fetching {args.malicious_n} Phishing.Database malicious seeds...")
    malicious = fetch_phishing_db_domains(n=args.malicious_n, seed=args.seed)
    print(f"  {len(malicious)} malicious apex domains")

    print(f"Fetching {args.benign_n} Tranco benign domains (skip top {args.benign_skip})...")
    benign = fetch_tranco_domains(n=args.benign_n, skip=args.benign_skip)
    print(f"  {len(benign)} benign domains")

    labels = {d: 1 for d in malicious} | {d: 0 for d in benign}
    domains = list(labels)

    print(f"Resolving {len(domains)} domains via active DNS (parallel)...")
    domain_ips = resolve_many(domains)
    print(f"  {len(domain_ips)}/{len(domains)} resolved")

    all_ips = sorted({ip for ips in domain_ips.values() for ip in ips})
    print(f"Resolving {len(all_ips)} IPs to ASN/subnet via Team Cymru...")
    ip_info = cymru_bulk_asn(all_ips, timeout=60.0)

    g = build_hosting_graph(domain_ips, ip_info, labels=labels)
    print("Hosting graph:", graph_summary(g))

    resolved_domains = [d for d in domains if d in domain_ips]
    print(f"Building/loading feature table (lex+host+topo) for {len(resolved_domains)} domains...")
    df = _load_or_build_feature_table(resolved_domains, labels, g, args.radius, args, args.top_k_centrality)
    domain_features = df.set_index("domain").to_dict("index")

    variants = [
        (feature_columns(df, include_topo=False), "GNN (lexical+hosting node features, no TDA)"),
        (feature_columns(df, include_topo=True), "GNN + TDA node features"),
    ]
    for cols, label in variants:
        nodes, X, edge_index = build_node_features(g, cols, domain_features)
        print(f"\n--- {label} ({len(cols)} domain-feature cols, {X.shape[0]} nodes, {edge_index.shape[1]} directed edges) ---")
        runs = [
            train_and_evaluate(nodes, X, edge_index, labels, seed=s, epochs=args.epochs)
            for s in range(args.seed, args.seed + args.n_seeds)
        ]
        means = {k: round(sum(r[k] for r in runs) / len(runs), 3) for k in runs[0]}
        if len(runs) > 1:
            stds = {k: round((sum((r[k] - means[k]) ** 2 for r in runs) / len(runs)) ** 0.5, 3) for k in runs[0]}
            print(f"    mean={means} std={stds}")
        else:
            print(f"    {means}")


if __name__ == "__main__":
    main()
