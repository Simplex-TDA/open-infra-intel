from mantis.graph import build_hosting_graph
from mantis.topo_features import domain_topo_features


def test_topo_features_on_shared_hosting_neighborhood():
    g = build_hosting_graph(
        {"evil.com": ["1.1.1.1", "2.2.2.2"], "good.com": ["1.1.1.1", "2.2.2.2"]},
        {
            "1.1.1.1": {"asn": "111", "subnet": "1.1.1.0/24"},
            "2.2.2.2": {"asn": "111", "subnet": "1.1.1.0/24"},
        },
    )
    f = domain_topo_features("evil.com", g, radius=2)
    assert f["topo_ego_n_nodes"] > 1
    assert isinstance(f["topo_curv_mean"], float)
    assert isinstance(f["topo_h0_total_persistence"], float)


def test_topo_features_default_for_isolated_domain():
    g = build_hosting_graph({"lonely.com": []}, {})
    f = domain_topo_features("lonely.com", g, radius=2)
    assert f["topo_ego_n_nodes"] == 1
    assert f["topo_curv_mean"] == 0.0
