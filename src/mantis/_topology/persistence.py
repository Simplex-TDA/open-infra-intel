"""Layer 4: persistent homology descriptors for an AS graph window.

Distance-matrix + sentinel-capping approach for disconnected graphs adapted
from the sibling `sybil` TDA project (extract_topology_features.py): ripser
needs a finite, complete distance matrix, so unreachable pairs are capped at
a sentinel larger than any real finite distance. A complete finite distance
matrix always yields exactly one infinite H0 bar (everything merges by the
sentinel); true component count = 1 + (# bars that die at the sentinel).
Verify against a synthetic two-triangle graph before trusting this on new
data (see sybil/STATUS.md step 12 for the precedent).
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
from ripser import ripser


def distance_matrix(g: nx.Graph, weight: str | None) -> tuple[np.ndarray, float, list]:
    nodes = list(g.nodes())
    n = len(nodes)
    idx = {a: i for i, a in enumerate(nodes)}
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)

    if weight is None:
        lengths = dict(nx.all_pairs_shortest_path_length(g))
    else:
        lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight=weight))
    for u, targets in lengths.items():
        for v, d in targets.items():
            dist[idx[u], idx[v]] = d
            dist[idx[v], idx[u]] = d

    finite = dist[np.isfinite(dist)]
    sentinel = (finite.max() * 2.0) if finite.size and finite.max() > 0 else 1.0
    dist[np.isinf(dist)] = sentinel
    return dist, sentinel, nodes


def edge_inverse_weight_graph(g: nx.Graph, weight_attr: str = "weight") -> nx.Graph:
    """Distance = 1 / edge weight, so heavily-observed edges are "closer"."""
    h = nx.Graph()
    h.add_nodes_from(g.nodes())
    for u, v, data in g.edges(data=True):
        w = data.get(weight_attr, 1)
        h.add_edge(u, v, inv_weight=1.0 / w if w else 1.0)
    return h


def compute_diagrams(dist: np.ndarray, maxdim: int = 1) -> dict:
    if dist.shape[0] < 2:
        empty = np.empty((0, 2))
        return {"h0": empty, "h1": empty}
    result = ripser(dist, distance_matrix=True, maxdim=maxdim)
    return {"h0": result["dgms"][0], "h1": result["dgms"][1]}


def persistence_entropy(lifetimes: np.ndarray) -> float:
    total = float(np.sum(lifetimes))
    if lifetimes.size == 0 or total <= 0:
        return 0.0
    p = lifetimes / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def persistence_features(dgms: dict, sentinel: float, prefix: str) -> dict:
    h0, h1 = dgms["h0"], dgms["h1"]
    n = h0.shape[0]
    if n == 0:
        return {
            f"{prefix}_h0_components": 0,
            f"{prefix}_h0_total_persistence": 0.0,
            f"{prefix}_h1_count": 0,
            f"{prefix}_h1_total_persistence": 0.0,
            f"{prefix}_h1_max_persistence": 0.0,
            f"{prefix}_h1_entropy": 0.0,
            f"{prefix}_total_persistence": 0.0,
            f"{prefix}_persistence_entropy": 0.0,
            f"{prefix}_diagram_cardinality": 0,
        }

    finite_deaths = h0[np.isfinite(h0[:, 1]), 1]
    at_sentinel = finite_deaths >= sentinel * 0.999
    n_components = 1 + int(np.sum(at_sentinel))
    real_h0 = h0[np.isfinite(h0[:, 1])][~at_sentinel]
    h0_lifetimes = real_h0[:, 1] - real_h0[:, 0] if len(real_h0) else np.empty(0)
    h0_total = float(np.sum(h0_lifetimes)) if h0_lifetimes.size else 0.0

    h1_lifetimes = h1[:, 1] - h1[:, 0] if h1.shape[0] else np.empty(0)
    h1_total = float(np.sum(h1_lifetimes)) if h1_lifetimes.size else 0.0
    h1_max = float(np.max(h1_lifetimes)) if h1_lifetimes.size else 0.0

    all_lifetimes = np.concatenate([h0_lifetimes, h1_lifetimes])

    return {
        f"{prefix}_h0_components": n_components,
        f"{prefix}_h0_total_persistence": h0_total,
        f"{prefix}_h1_count": int(h1.shape[0]),
        f"{prefix}_h1_total_persistence": h1_total,
        f"{prefix}_h1_max_persistence": h1_max,
        f"{prefix}_h1_entropy": persistence_entropy(h1_lifetimes),
        f"{prefix}_total_persistence": h0_total + h1_total,
        f"{prefix}_persistence_entropy": persistence_entropy(all_lifetimes),
        f"{prefix}_diagram_cardinality": int(len(h0_lifetimes) + h1.shape[0]),
    }


def wasserstein_distance(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    """Wasserstein distance between two persistence diagrams (same dim),
    dropping non-finite deaths so infinite bars don't blow up the distance.
    """
    from persim import wasserstein

    def finite(d: np.ndarray) -> np.ndarray:
        if d.size == 0:
            return d
        return d[np.isfinite(d[:, 1])]

    a, b = finite(dgm_a), finite(dgm_b)
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(wasserstein(a, b))
