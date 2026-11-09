"""MANTIS-TDA PoC driver: pull Phishing.Database (malicious) + Tranco
(benign) domains, resolve hosting via active DNS + Team Cymru whois, build
the hosting graph, and compare lexical+hosting features alone vs with added
TDA topology features (mantis.topo_features) -- same ablation style
as the paper's Table 4.

Usage: python experiments/mantis_poc.py --malicious-n 3000 --benign-n 3000
"""
from __future__ import annotations

import argparse
import pathlib
import time

import pandas as pd

from mantis.classify import build_feature_table, evaluate, feature_columns, feature_importances
from mantis.graph import build_hosting_graph, graph_summary
from mantis.ingest import (
    cymru_bulk_asn,
    fetch_phishing_db_domains,
    fetch_tranco_domains,
    resolve_many,
)

# Building the feature table (Ricci curvature + persistence per domain) is
# the expensive step -- tens of minutes per radius -- while the ablation/
# importance comparisons below it are cheap. Cache per radius so iterating
# on feature-set variants doesn't re-pay that cost.
CACHE_DIR = pathlib.Path(__file__).parent / ".cache"

# (include_topo, topo_mode, label) -- the two "leads": drop pure-size topo
# columns (shape_only) and/or rely on the size-normalized persistence
# features added alongside them (both included in "full").
_VARIANTS = [
    (False, "full", "lexical+hosting only"),
    (True, "full", "lexical+hosting+TDA (full)"),
    (True, "shape_only", "lexical+hosting+TDA (shape-only, no size cols)"),
]


def _cache_path(n_domains: int, radius: int, args, top_k: int | None) -> pathlib.Path:
    key = f"m{args.malicious_n}_b{args.benign_n}_skip{args.benign_skip}_seed{args.seed}_r{radius}_n{n_domains}"
    if args.max_edges is not None:
        key += f"_maxedges{args.max_edges}"
    if args.max_degree is not None:
        key += f"_maxdeg{args.max_degree}"
    if top_k is not None:
        key += f"_topk{top_k}"
    return CACHE_DIR / f"features_{key}.csv"


def _load_or_build_feature_table(resolved_domains, labels, g, radius: int, args, top_k: int | None) -> pd.DataFrame:
    path = _cache_path(len(resolved_domains), radius, args, top_k)
    if path.exists():
        print(f"  (loaded cached feature table: {path.name})")
        return pd.read_csv(path)
    start = time.monotonic()
    df = build_feature_table(
        resolved_domains,
        labels,
        g,
        radius=radius,
        max_edges=args.max_edges,
        max_degree=args.max_degree,
        top_k_centrality=top_k,
        progress_every=args.progress_every,
    )
    print(f"  feature table build took {time.monotonic() - start:.1f}s", flush=True)
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _run_ablation(df, args) -> None:
    for include_topo, topo_mode, label in _VARIANTS:
        cols = feature_columns(df, include_topo=include_topo, topo_mode=topo_mode)
        print(f"  --- {label} ({len(cols)} features) ---")

        per_model_runs: dict[str, list[dict]] = {}
        for seed in range(args.seed, args.seed + args.n_seeds):
            for model, metrics in evaluate(df, cols, seed=seed).items():
                per_model_runs.setdefault(model, []).append(metrics)

        for model, runs in per_model_runs.items():
            means = {k: round(sum(r[k] for r in runs) / len(runs), 3) for k in runs[0]}
            if len(runs) > 1:
                stds = {k: round((sum((r[k] - means[k]) ** 2 for r in runs) / len(runs)) ** 0.5, 3) for k in runs[0]}
                print(f"    {model}: mean={means} std={stds}")
            else:
                print(f"    {model}: {means}")


def _print_feature_importances(df, args, topo_mode: str, label: str) -> None:
    cols = feature_columns(df, include_topo=True, topo_mode=topo_mode)
    print(f"  --- feature importances: {label} ---")
    for model, importances in feature_importances(df, cols, seed=args.seed).items():
        print(f"    {model} (top {args.top_features}):")
        for feat, score in importances.head(args.top_features).items():
            tag = "TDA" if feat.startswith("topo_") else "lex/host"
            print(f"      {score:.4f}  [{tag}]  {feat}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--malicious-n", type=int, default=3000, help="random sample from Phishing.Database")
    parser.add_argument("--benign-n", type=int, default=3000)
    parser.add_argument("--benign-skip", type=int, default=2000)
    parser.add_argument("--radius", type=int, nargs="+", default=[2], help="one or more ego-graph radii to sweep")
    parser.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="keep only the N cheapest edges (lowest deg(u)*deg(v)) of the ego-graph and the "
        "subgraph induced on just their endpoints (see docs/CURRENT_PLAN.md task 1); default: uncapped",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=None,
        help="drop any ego-graph node (other than the domain itself) whose degree exceeds this",
    )
    parser.add_argument(
        "--top-k-centrality",
        type=int,
        nargs="+",
        default=[50],
        help="keep only the N highest-PageRank nodes of the ego-graph (plus the domain itself); "
        "one or more values to sweep, reusing the same fetched/resolved data. Default 50 is the "
        "current standard (see docs/archive/experiment_log.md, 2026-11-08 K-sweep: ~27x faster "
        "than uncapped radius=3 with matching/better AUC/F1).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="print elapsed time + curvature/persistence time breakdown every N domains while building "
        "the feature table (0 to disable)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=1, help="average metrics over this many train/test splits")
    parser.add_argument("--top-features", type=int, default=15)
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
    print(f"Building feature table for {len(resolved_domains)} resolved domains...")

    for radius in args.radius:
        for top_k in (args.top_k_centrality or [None]):
            label = f"radius={radius}" + (f" top_k={top_k}" if top_k is not None else "")
            print(f"\n=== {label} ===")
            df = _load_or_build_feature_table(resolved_domains, labels, g, radius, args, top_k)
            _run_ablation(df, args)
            _print_feature_importances(df, args, "full", "full topo set")
            _print_feature_importances(df, args, "shape_only", "shape-only (no size cols) topo set")


if __name__ == "__main__":
    main()
