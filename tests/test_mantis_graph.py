from mantis.graph import build_hosting_graph, domain_ego_graph, graph_summary


def _sample_graph():
    domain_ips = {"evil.com": ["1.2.3.4"], "good.com": ["1.2.3.4"], "other.com": ["9.9.9.9"]}
    ip_info = {
        "1.2.3.4": {"asn": "111", "subnet": "1.2.3.0/24"},
        "9.9.9.9": {"asn": "222", "subnet": "9.9.9.0/24"},
    }
    return build_hosting_graph(domain_ips, ip_info, labels={"evil.com": 1, "good.com": 0, "other.com": 0})


def test_shared_ip_links_two_domains_through_hosting_graph():
    g = _sample_graph()
    assert nx_has_path(g, "evil.com", "good.com")
    assert not nx_has_path(g, "evil.com", "other.com")
    summary = graph_summary(g)
    assert summary["by_kind"] == {"domain": 3, "ip": 2, "subnet": 2, "asn": 2}


def test_domain_ego_graph_radius_limits_reach():
    g = _sample_graph()
    ego1 = domain_ego_graph(g, "evil.com", radius=1)
    assert set(ego1.nodes()) == {"evil.com", "1.2.3.4"}
    ego4 = domain_ego_graph(g, "evil.com", radius=4)
    assert "good.com" in ego4.nodes()


def nx_has_path(g, a, b):
    import networkx as nx
    return nx.has_path(g, a, b)


def _hub_graph():
    domain_ips = {f"d{i}.com": [f"1.2.3.{i}"] for i in range(20)}
    ip_info = {f"1.2.3.{i}": {"asn": "111", "subnet": "1.2.3.0/24"} for i in range(20)}
    return build_hosting_graph(domain_ips, ip_info)


def test_domain_ego_graph_max_edges_bounds_node_count_too():
    g = _hub_graph()
    full = domain_ego_graph(g, "d0.com", radius=3)
    assert full.number_of_nodes() > 10  # unbounded: pulls in every sibling domain/IP via the hub

    capped = domain_ego_graph(g, "d0.com", radius=3, max_edges=3)
    assert capped.number_of_edges() <= 3
    assert capped.number_of_nodes() < full.number_of_nodes()  # untouched nodes are dropped, not kept
    assert "d0.com" in capped


def test_domain_ego_graph_max_degree_removes_hub_node():
    g = _hub_graph()
    capped = domain_ego_graph(g, "d0.com", radius=3, max_degree=2)
    assert "1.2.3.0/24" not in capped.nodes()  # hub's degree (20) exceeds cap -> dropped outright
    assert "d0.com" in capped


def test_domain_ego_graph_top_k_centrality_bounds_node_count():
    g = _hub_graph()
    full = domain_ego_graph(g, "d0.com", radius=3)
    capped = domain_ego_graph(g, "d0.com", radius=3, top_k_centrality=5)
    assert capped.number_of_nodes() <= 5
    assert capped.number_of_nodes() < full.number_of_nodes()
    assert "d0.com" in capped


def test_cymru_na_sentinel_treated_as_absent_not_shared_hub():
    import networkx as nx

    domain_ips = {"a.com": ["10.0.0.1"], "b.com": ["10.0.0.2"]}
    ip_info = {
        "10.0.0.1": {"asn": "NA", "subnet": "NA"},
        "10.0.0.2": {"asn": "NA", "subnet": "NA"},
    }
    g = build_hosting_graph(domain_ips, ip_info)
    assert not list(nx.selfloop_edges(g))
    assert not nx.has_path(g, "a.com", "b.com")
