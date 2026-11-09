"""Ollivier-Ricci curvature per edge, adapted from the sibling `sybil`
project's precedent (extract_topology_features.py). Negative curvature
marks "bridge-like" edges (few common neighbors relative to degree --
structurally load-bearing, tree-like connectors); positive curvature marks
edges embedded in dense, redundantly-connected neighborhoods.

Cost is dominated by the optimal-transport problem at each edge, which
scales with the *product of endpoint degrees in the graph actually passed
in* -- not edge count. Measured directly: computing curvature on the real
~27,700-node/53,500-edge global AS graph (hub degree up to ~2,100) did not
finish in 10 minutes for a single window. Computing it on a ~750-node/
800-edge layer extracted from the same graph (hub degree capped at ~110
because most of a hub's edges get filtered out of the layer) took well
under a second. Conclusion: only run this on bounded layers
(graphs/layers.py), never on the full global graph.

Bypasses GraphRicciCurvature's public `OllivierRicci` class entirely and
calls its internal per-edge function directly in a plain loop. Not a style
choice: the public class's `_compute_ricci_curvature_edges` unconditionally
opens a `multiprocessing.get_context('fork').Pool(processes=proc)` --
`proc=1` still forks a worker, and its `pool.close(); pool.join()` runs
*before* the (lazily-evaluated) `imap_unordered` result is ever consumed.
Calling this ~100+ times over one experiment leaked dozens of orphaned
`multiprocessing.spawn`/`resource_tracker` processes that measurably
degraded performance for the whole machine, not just this script (found by
noticing the same computation getting progressively slower across windows
for no algorithmic reason, then finding ~60 idle Python processes with
hour-long uptimes). The module's internal single-edge function has no such
problem -- it just needs its module-level globals (including two
`lru_cache`s keyed by small integer node ids, which collide across
different graphs unless explicitly cleared) set up the way the class
itself would. Pinned to GraphRicciCurvature 0.5.4's internal layout; a
version bump could silently break this.
"""
from __future__ import annotations

import math

import networkit as nk
import networkx as nx
import numpy as np
from GraphRicciCurvature import OllivierRicci as _orc


def _edge_curvatures(g: nx.Graph, alpha: float) -> dict[frozenset, float]:
    """Shared setup/teardown for the module-global GraphRicciCurvature
    internals (see module docstring) -- returns raw per-edge values keyed
    by the edge itself, which curvature_features() aggregates and
    compute_edge_curvatures() (graphs/layers.py's curvature-tier partition)
    uses directly.
    """
    gc = nx.Graph()
    gc.add_edges_from(g.edges())
    for u, v in gc.edges():
        gc[u][v]["weight"] = 1.0

    nx2nk = {node: i for i, node in enumerate(gc.nodes())}

    _orc._Gk = nk.nxadapter.nx2nk(gc, weightAttr="weight")
    _orc._alpha = alpha
    _orc._weight = "weight"
    _orc._method = "OTDSinkhornMix"
    _orc._base = math.e
    _orc._exp_power = 2
    _orc._shortest_path = "all_pairs"
    _orc._nbr_topk = 3000
    # Both caches are keyed by small integer node ids that get reused across
    # unrelated graphs every time _Gk is reassigned -- stale hits would
    # silently return another window's/layer's curvature.
    _orc._get_single_node_neighbors_distributions.cache_clear()
    _orc._source_target_shortest_path.cache_clear()
    _orc._apsp = _orc._get_all_pairs_shortest_path()

    curvatures: dict[frozenset, float] = {}
    for u, v in gc.edges():
        result = _orc._compute_ricci_curvature_single_edge(nx2nk[u], nx2nk[v])
        curvatures[frozenset((u, v))] = next(iter(result.values()))

    _orc._get_single_node_neighbors_distributions.cache_clear()
    _orc._source_target_shortest_path.cache_clear()
    return curvatures


def curvature_features(g: nx.Graph, prefix: str, alpha: float = 0.5) -> dict:
    if g.number_of_edges() == 0:
        return {
            f"{prefix}_curv_mean": 0.0,
            f"{prefix}_curv_min": 0.0,
            f"{prefix}_curv_max": 0.0,
            f"{prefix}_curv_frac_negative": 0.0,
        }

    values = np.array(list(_edge_curvatures(g, alpha).values()))
    return {
        f"{prefix}_curv_mean": float(values.mean()),
        f"{prefix}_curv_min": float(values.min()),
        f"{prefix}_curv_max": float(values.max()),
        f"{prefix}_curv_frac_negative": float((values < 0).mean()),
    }


def compute_edge_curvatures(g: nx.Graph, alpha: float = 0.5) -> dict[frozenset, float]:
    """Per-edge Ollivier-Ricci curvature, for use as a partition criterion
    (bucket edges by curvature tier) rather than just a per-layer summary
    statistic. Same cost profile as curvature_features -- only call this on
    a bounded subgraph (graphs/layers.py's top_k_by_weight/top_k_by_centrality),
    never the full global graph.
    """
    if g.number_of_edges() == 0:
        return {}
    return _edge_curvatures(g, alpha)
