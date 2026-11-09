"""Paired multi-seed significance test: does adding TDA features to
lexical+hosting features change RF/XGBoost performance? Paired per seed
because `evaluate()`'s train_test_split depends only on the row index/labels
and the seed, not which feature columns are selected -- so the no-TDA and
+TDA runs at a given seed share the exact same train/test split.

Usage: python experiments/significance_classify.py --n-seeds 10
       python experiments/significance_classify.py --n-seeds 10 --cache path/to/features.csv
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd
from scipy import stats

from mantis.classify import evaluate, feature_columns
from mantis.graph import build_hosting_graph, graph_summary
from mantis.ingest import cymru_bulk_asn, fetch_phishing_db_domains, fetch_tranco_domains, resolve_many
from mantis_poc import _load_or_build_feature_table

DEFAULT_CACHE = pathlib.Path(__file__).parent / "sample_data" / "features_m3000_b3000_skip2000_seed0_r3_n3977_topk50.csv"


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

    return _load_or_build_feature_table(resolved_domains, labels, g, args.radius, _Args(), args.top_k_centrality)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--malicious-n", type=int, default=3000)
    parser.add_argument("--benign-n", type=int, default=3000)
    parser.add_argument("--benign-skip", type=int, default=2000)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--top-k-centrality", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="fetch/resolve seed, independent of the n-seeds eval loop")
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=DEFAULT_CACHE,
        help="pre-built feature-table CSV to reuse (default: the bundled sample table, "
        "skips the ~15min DNS resolution + TDA feature build entirely)",
    )
    parser.add_argument("--rebuild", action="store_true", help="ignore --cache and rebuild from scratch")
    args = parser.parse_args()

    if args.cache.exists() and not args.rebuild:
        print(f"Loading cached feature table: {args.cache}")
        df = pd.read_csv(args.cache)
    else:
        df = _build_feature_table(args)

    variants = {
        "no_tda": feature_columns(df, include_topo=False),
        "tda": feature_columns(df, include_topo=True),
    }

    by_variant = {"no_tda": {}, "tda": {}}
    for name, cols in variants.items():
        for seed in range(args.n_seeds):
            by_variant[name][seed] = evaluate(df, cols, seed=seed)

    models = next(iter(by_variant["no_tda"].values())).keys()
    metrics = ["auc", "precision", "recall", "f1", "fpr"]

    for model in models:
        print(f"\n=== {model} ===")
        print("--- paired comparison (tda - no_tda), paired t-test across seeds ---")
        for m in metrics:
            a = [by_variant["no_tda"][s][model][m] for s in range(args.n_seeds)]
            b = [by_variant["tda"][s][model][m] for s in range(args.n_seeds)]
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
