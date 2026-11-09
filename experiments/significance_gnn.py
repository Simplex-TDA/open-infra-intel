"""Paired multi-seed significance test: does adding TDA features as extra
GraphSAGE node inputs change GNN performance? Paired per seed, same reasoning
as significance_classify.py (fixed seed -> identical train/test split for
both variants).

N_PROCESSES defaults to 2: an earlier session ran this at higher parallelism
on a laptop and triggered a thermal shutdown. Training is CPU-bound and
short-lived per run, so 1-2 processes is plenty and safe on any machine.

Usage: python experiments/significance_gnn.py --n-seeds 10 --epochs 200
       python experiments/significance_gnn.py --n-seeds 10 --epochs 800 --weight-decay 1e-4
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing
import pathlib
import pickle

import pandas as pd
from scipy import stats

from mantis.classify import feature_columns
from mantis.gnn import build_node_features, train_and_evaluate
from mantis.graph import build_hosting_graph, graph_summary
from mantis.ingest import cymru_bulk_asn, fetch_phishing_db_domains, fetch_tranco_domains, resolve_many
from mantis_poc import _load_or_build_feature_table

_SAMPLE_DATA = pathlib.Path(__file__).parent / "sample_data"
DEFAULT_CACHE = _SAMPLE_DATA / "features_m3000_b3000_skip2000_seed0_r3_n3977_topk50.csv"
DEFAULT_GRAPH_CACHE = _SAMPLE_DATA / "graph_m3000_b3000_skip2000_seed0_r3_n3977_topk50.pkl.gz"
N_PROCESSES = 2


def _build_feature_table(args) -> pd.DataFrame:
    print(f"Fetching {args.malicious_n} malicious + {args.benign_n} benign domains...", flush=True)
    malicious = fetch_phishing_db_domains(n=args.malicious_n, seed=args.seed)
    benign = fetch_tranco_domains(n=args.benign_n, skip=args.benign_skip)
    labels = {d: 1 for d in malicious} | {d: 0 for d in benign}
    domains = list(labels)

    print(f"Resolving {len(domains)} domains via active DNS...", flush=True)
    domain_ips = resolve_many(domains)
    print(f"  {len(domain_ips)}/{len(domains)} resolved", flush=True)

    all_ips = sorted({ip for ips in domain_ips.values() for ip in ips})
    ip_info = cymru_bulk_asn(all_ips, timeout=60.0)
    g = build_hosting_graph(domain_ips, ip_info, labels=labels)
    print("Hosting graph:", graph_summary(g), flush=True)

    resolved_domains = [d for d in domains if d in domain_ips]

    class _Args:
        malicious_n = args.malicious_n
        benign_n = args.benign_n
        benign_skip = args.benign_skip
        seed = args.seed
        max_edges = None
        max_degree = None
        progress_every = 200

    return _load_or_build_feature_table(resolved_domains, labels, g, args.radius, _Args(), args.top_k_centrality), g, labels


def _run_one(task) -> tuple[str, int, dict]:
    variant_name, seed, nodes, X, edge_index, labels, epochs, weight_decay = task
    metrics = train_and_evaluate(nodes, X, edge_index, labels, seed=seed, epochs=epochs, weight_decay=weight_decay)
    return variant_name, seed, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--malicious-n", type=int, default=3000)
    parser.add_argument("--benign-n", type=int, default=3000)
    parser.add_argument("--benign-skip", type=int, default=2000)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--top-k-centrality", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="fetch/resolve seed, independent of the n-seeds eval loop")
    parser.add_argument("--epochs", type=int, default=200, help="200 matches the MANTIS paper's stated value")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=DEFAULT_CACHE,
        help="pre-built feature-table CSV to reuse (default: the bundled sample table)",
    )
    parser.add_argument(
        "--graph-cache",
        type=pathlib.Path,
        default=DEFAULT_GRAPH_CACHE,
        help="gzip-pickled hosting graph matching --cache (default: the bundled sample graph; "
        "must be the graph that --cache's feature table was built from, not an independently "
        "re-fetched one, or features/edges/labels can mismatch)",
    )
    parser.add_argument("--rebuild", action="store_true", help="ignore --cache/--graph-cache and rebuild from scratch")
    parser.add_argument("--n-processes", type=int, default=N_PROCESSES)
    args = parser.parse_args()

    if args.cache.exists() and args.graph_cache.exists() and not args.rebuild:
        print(f"Loading cached feature table: {args.cache}")
        df = pd.read_csv(args.cache)
        print(f"Loading cached hosting graph: {args.graph_cache}")
        with gzip.open(args.graph_cache, "rb") as f:
            g = pickle.load(f)
        labels = dict(zip(df["domain"], df["label"]))
    else:
        df, g, labels = _build_feature_table(args)

    domain_features = df.set_index("domain").to_dict("index")
    variants = {
        "no_tda": feature_columns(df, include_topo=False),
        "tda": feature_columns(df, include_topo=True),
    }

    tasks = []
    for variant_name, cols in variants.items():
        nodes, X, edge_index = build_node_features(g, cols, domain_features)
        for seed in range(args.n_seeds):
            tasks.append((variant_name, seed, nodes, X, edge_index, labels, args.epochs, args.weight_decay))

    print(f"Running {len(tasks)} GNN train/eval runs across {args.n_processes} process(es)...", flush=True)
    with multiprocessing.Pool(args.n_processes) as pool:
        results = pool.map(_run_one, tasks)

    by_variant: dict[str, dict[int, dict]] = {"no_tda": {}, "tda": {}}
    for variant_name, seed, metrics in results:
        by_variant[variant_name][seed] = metrics

    metrics = ["auc", "precision", "recall", "f1", "fpr"]
    print("\n--- paired comparison (tda - no_tda), paired t-test across seeds ---")
    for m in metrics:
        a = [by_variant["no_tda"][s][m] for s in range(args.n_seeds)]
        b = [by_variant["tda"][s][m] for s in range(args.n_seeds)]
        deltas = [y - x for x, y in zip(a, b)]
        mean_delta = sum(deltas) / len(deltas)
        t, p = stats.ttest_rel(b, a)
        n_improved = sum(1 for d in deltas if (d > 0 if m != "fpr" else d < 0))
        print(
            f"  {m}: no_tda={sum(a)/len(a):.3f} tda={sum(b)/len(b):.3f} "
            f"mean_delta={mean_delta:+.3f} t={t:.2f} p={p:.4f} "
            f"improved_in={n_improved}/{args.n_seeds}_seeds"
        )


if __name__ == "__main__":
    main()
