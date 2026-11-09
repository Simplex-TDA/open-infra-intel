from mantis.classify import feature_columns
from mantis.gnn import build_node_features, train_and_evaluate
from mantis.graph import build_hosting_graph


def _synthetic_dataset():
    domain_ips, ip_info, labels = {}, {}, {}
    for i in range(10):
        d = f"evil{i}.tk"
        ip = f"6.6.6.{i % 3}"
        domain_ips[d] = [ip]
        ip_info[ip] = {"asn": "666", "subnet": "6.6.6.0/24"}
        labels[d] = 1
    for i in range(10):
        d = f"good{i}.com"
        ip = f"7.7.{i}.1"
        domain_ips[d] = [ip]
        ip_info[ip] = {"asn": str(700 + i), "subnet": f"7.7.{i}.0/24"}
        labels[d] = 0
    g = build_hosting_graph(domain_ips, ip_info, labels=labels)
    return labels, g


def test_build_node_features_shapes():
    labels, g = _synthetic_dataset()
    domain_features = {d: {"lex_length": float(len(d))} for d in labels}
    nodes, X, edge_index = build_node_features(g, ["lex_length"], domain_features)
    assert X.shape == (len(nodes), 4 + 1 + 1)
    assert edge_index.shape[0] == 2
    assert edge_index.max().item() < len(nodes)
    domain_row = X[nodes.index("evil0.tk")]
    assert domain_row[-1] == len("evil0.tk")


def test_train_and_evaluate_end_to_end():
    labels, g = _synthetic_dataset()
    nodes, X, edge_index = build_node_features(g, [], {})
    metrics = train_and_evaluate(nodes, X, edge_index, labels, test_size=0.3, epochs=20)
    assert set(metrics) == {"auc", "precision", "recall", "f1", "fpr"}
    assert 0.0 <= metrics["f1"] <= 1.0

    metrics_wd = train_and_evaluate(nodes, X, edge_index, labels, test_size=0.3, epochs=20, weight_decay=1e-3)
    assert set(metrics_wd) == {"auc", "precision", "recall", "f1", "fpr"}
