from mantis.features import extract_features, lexical_features
from mantis.graph import build_hosting_graph


def test_lexical_features_flag_suspicious_tld_and_hyphen():
    f = lexical_features("paypal-verify.tk")
    assert f["lex_num_minus"] == 1
    assert f["lex_suspicious_tld"] == 1
    assert f["lex_length"] == len("paypal-verify.tk")


def test_hosting_features_reflect_shared_ip():
    g = build_hosting_graph(
        {"evil.com": ["1.2.3.4"], "good.com": ["1.2.3.4"]},
        {"1.2.3.4": {"asn": "111", "subnet": "1.2.3.0/24"}},
    )
    f = extract_features("evil.com", g)
    assert f["host_num_ips"] == 1
    assert f["host_max_domains_per_ip"] == 2
    assert f["host_has_asn"] == 1


def test_hosting_features_default_for_unresolved_domain():
    g = build_hosting_graph({}, {})
    f = extract_features("missing.com", g)
    assert f["host_num_ips"] == 0
