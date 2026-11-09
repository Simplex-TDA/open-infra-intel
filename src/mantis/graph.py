"""Hosting graph: domain--IP--subnet--ASN, per MANTIS Fig 5 (minus the
apex/subdomain split and domain-domain "similar" edges, which need data we
don't have -- see mantis package docstring).
"""
from __future__ import annotations

import networkx as nx

# Hyperscale CDN/cloud ASNs multiplex thousands of unrelated tenants behind
# a handful of edge IPs/announced prefixes. Letting their subnet/ASN nodes
# be shared graph hubs wires most of the dataset together on pure
# infrastructure-provider coincidence, not real shared hosting -- verified
# against live OpenPhish+Tranco data, where AS13335 (Cloudflare) alone
# touched 44% of resolved domains. Dedicated/boutique hosts are left alone:
# co-location there is still real signal.
CDN_CLOUD_ASNS = frozenset({
    "13335",  # Cloudflare
    "16509", "14618", "8987",  # Amazon AWS
    "15169", "396982",  # Google / Google Cloud
    "8075",  # Microsoft Azure
    "54113",  # Fastly
    "20940", "16625", "34164",  # Akamai
    "16276",  # OVH
    "14061",  # DigitalOcean
    "31898",  # Oracle Cloud
    "63949",  # Linode/Akamai Connected Cloud
})


def _top_k_by_weight(g: nx.Graph, k: int, ascending: bool, weight_attr: str = "weight") -> nx.Graph:
    edges = sorted(g.edges(data=True), key=lambda e: e[2].get(weight_attr, 1), reverse=not ascending)
    h = nx.Graph()
    for u, v, d in edges[:k]:
        h.add_edge(u, v, **d)
    return h


def build_hosting_graph(
    domain_ips: dict[str, list[str]],
    ip_info: dict[str, dict],
    labels: dict[str, int] | None = None,
) -> nx.Graph:
    g = nx.Graph()
    for domain, ips in domain_ips.items():
        g.add_node(domain, kind="domain", label=(labels or {}).get(domain))
        for ip in ips:
            g.add_node(ip, kind="ip")
            g.add_edge(domain, ip, kind="resolves")
            info = ip_info.get(ip)
            if not info:
                continue
            # Team Cymru returns the literal string "NA" (not missing/None)
            # for IPs it can't attribute to a routed prefix/ASN (private,
            # bogon, unallocated). Treat it as absent, else every such IP
            # collapses onto one shared "NA" hub node across domains -- for
            # asn == subnet == "NA" that's even a literal self-loop.
            subnet = info.get("subnet")
            if subnet == "NA":
                subnet = None
            if subnet:
                asn = info.get("asn")
                if asn == "NA":
                    asn = None
                is_cdn = asn in CDN_CLOUD_ASNS
                # scope per-domain so a shared hyperscale provider doesn't
                # become a hub linking unrelated domains together
                subnet_node = f"{subnet}::{domain}" if is_cdn else subnet
                g.add_node(subnet_node, kind="subnet")
                g.add_edge(ip, subnet_node, kind="within")
                if asn:
                    asn_node = f"{asn}::{domain}" if is_cdn else asn
                    g.add_node(asn_node, kind="asn")
                    g.add_edge(subnet_node, asn_node, kind="within")
    return g


def graph_summary(g: nx.Graph) -> dict:
    kinds = nx.get_node_attributes(g, "kind")
    counts: dict[str, int] = {}
    for kind in kinds.values():
        counts[kind] = counts.get(kind, 0) + 1
    return {"n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges(), "by_kind": counts}


def domain_ego_graph(
    g: nx.Graph,
    domain: str,
    radius: int = 2,
    max_edges: int | None = None,
    max_degree: int | None = None,
    top_k_centrality: int | None = None,
) -> nx.Graph:
    """Domain's local hosting neighborhood (domain-IP-subnet[-ASN] within
    `radius` hops), matching the paper's "Level 2" expansion -- this is the
    subgraph whose shape becomes that domain's TDA feature vector.

    Three independent, composable reduction knobs (apply in this order if
    more than one is given), each sized to directly bound a *measured* cost
    driver rather than just node/edge count in general -- curvature cost is
    dominated by the product of endpoint degrees per edge (topology/
    curvature.py), persistence cost by total node count (its distance
    matrix is dense). An earlier version only dropped edges while keeping
    every node (bounding curvature's driver but not persistence's) and
    measured *slower* than uncapped radius=3 at full scale -- see
    docs/archive/experiment_log.md.

      max_edges: keep the `max_edges` cheapest edges (lowest
        deg(u)*deg(v)) and the induced subgraph on just their endpoints --
        bounds edge count and, since untouched nodes are dropped, node
        count too.
      max_degree: drop any node whose degree here exceeds this (always
        keeps `domain` itself) -- removes a hub outright rather than
        trimming its edges, so it can't leave it in with inflated degree.
      top_k_centrality: keep only the `top_k_centrality` highest-PageRank
        nodes (always including `domain`) -- directly bounds node count.
        PageRank (power iteration, ~O(E) per round) rather than
        betweenness, which is too expensive to spend on a step whose whole
        point is cutting cost.
    """
    if domain not in g:
        return nx.Graph()
    ego = nx.ego_graph(g, domain, radius=radius)

    if max_edges is not None and ego.number_of_edges() > max_edges:
        for u, v, data in ego.edges(data=True):
            data["_deg_product"] = ego.degree(u) * ego.degree(v)
        ego = _top_k_by_weight(ego, max_edges, ascending=True, weight_attr="_deg_product")
        if domain not in ego:
            ego.add_node(domain)

    if max_degree is not None:
        keep = {domain} | {n for n in ego.nodes() if ego.degree(n) <= max_degree}
        ego = ego.subgraph(keep).copy()

    if top_k_centrality is not None and ego.number_of_nodes() > top_k_centrality:
        pr = nx.pagerank(ego)
        keep = {domain} | set(sorted(pr, key=pr.get, reverse=True)[:top_k_centrality])
        ego = ego.subgraph(keep).copy()

    return ego
