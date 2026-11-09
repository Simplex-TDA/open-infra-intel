from mantis.classify import build_feature_table, evaluate, feature_columns
from mantis.graph import build_hosting_graph


def _synthetic_dataset():
    domain_ips, ip_info, labels = {}, {}, {}
    # malicious domains cluster on shared toxic IPs
    for i in range(10):
        d = f"evil{i}.tk"
        ip = f"6.6.6.{i % 3}"
        domain_ips[d] = [ip]
        ip_info[ip] = {"asn": "666", "subnet": "6.6.6.0/24"}
        labels[d] = 1
    # benign domains each on their own clean IP
    for i in range(10):
        d = f"good{i}.com"
        ip = f"7.7.{i}.1"
        domain_ips[d] = [ip]
        ip_info[ip] = {"asn": str(700 + i), "subnet": f"7.7.{i}.0/24"}
        labels[d] = 0
    g = build_hosting_graph(domain_ips, ip_info, labels=labels)
    return list(domain_ips), labels, g


def test_build_feature_table_and_evaluate_end_to_end():
    domains, labels, g = _synthetic_dataset()
    df = build_feature_table(domains, labels, g)
    assert len(df) == 20
    assert set(df["label"]) == {0, 1}

    cols_no_topo = feature_columns(df, include_topo=False)
    cols_with_topo = feature_columns(df, include_topo=True)
    assert all(c.startswith(("lex_", "host_")) for c in cols_no_topo)
    assert any(c.startswith("topo_") for c in cols_with_topo)

    results = evaluate(df, cols_with_topo, test_size=0.4, seed=0)
    assert set(results) == {"random_forest", "xgboost"}
    for metrics in results.values():
        for v in metrics.values():
            assert 0.0 <= v <= 1.0
