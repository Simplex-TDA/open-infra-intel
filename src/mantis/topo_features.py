"""Per-domain TDA topology features: this is the substitute for MANTIS's GNN
encoder. Each domain's local hosting neighborhood (graph.domain_ego_graph)
is reduced to a fixed-size feature vector via vendored Ollivier-Ricci
curvature + persistence-diagram code (originally from a sibling AS-routing
project's topology module, copied in verbatim so this package has no
external sibling-project dependency).
"""
from __future__ import annotations

import time

import networkx as nx

from mantis._topology.curvature import curvature_features
from mantis._topology.persistence import compute_diagrams, distance_matrix, persistence_features
from mantis.graph import domain_ego_graph


def domain_topo_features(
    domain: str,
    g: nx.Graph,
    radius: int = 2,
    max_edges: int | None = None,
    max_degree: int | None = None,
    top_k_centrality: int | None = None,
    prefix: str = "topo",
    timing: dict | None = None,
) -> dict:
    """`timing`, if given, accumulates cumulative seconds under "curvature"
    and "persistence" -- the two steps with documented, different cost
    drivers (see graph.domain_ego_graph) -- so a sampling strategy's wall-
    clock win can be attributed to the step it actually targets."""
    ego = domain_ego_graph(
        g, domain, radius=radius, max_edges=max_edges, max_degree=max_degree, top_k_centrality=top_k_centrality
    )

    t0 = time.perf_counter()
    curv = curvature_features(ego, prefix=prefix)
    t1 = time.perf_counter()
    dist, sentinel, _ = distance_matrix(ego, weight=None)
    dgms = compute_diagrams(dist)
    pers = persistence_features(dgms, sentinel, prefix=prefix)
    t2 = time.perf_counter()
    if timing is not None:
        timing["curvature"] = timing.get("curvature", 0.0) + (t1 - t0)
        timing["persistence"] = timing.get("persistence", 0.0) + (t2 - t1)
    size = {f"{prefix}_ego_n_nodes": ego.number_of_nodes(), f"{prefix}_ego_n_edges": ego.number_of_edges()}

    # Persistence totals/counts grow with neighborhood size almost by
    # construction (more points -> more bars), which can make "shape"
    # features look like disguised size features. Dividing by node count
    # isolates the size-independent part.
    n_nodes = max(size[f"{prefix}_ego_n_nodes"], 1)
    norm = {
        f"{prefix}_h0_total_persistence_per_node": pers[f"{prefix}_h0_total_persistence"] / n_nodes,
        f"{prefix}_h1_total_persistence_per_node": pers[f"{prefix}_h1_total_persistence"] / n_nodes,
        f"{prefix}_total_persistence_per_node": pers[f"{prefix}_total_persistence"] / n_nodes,
        f"{prefix}_h1_count_per_node": pers[f"{prefix}_h1_count"] / n_nodes,
    }
    return {**curv, **pers, **size, **norm}
