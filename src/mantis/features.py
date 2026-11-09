"""Per-domain lexical + hosting features, the computable subset of paper
Table 7. Dropped: num_subdomains/subdomain_len (we only have apex domains,
not a PDNS-derived subdomain history), and every query_count/duration field
(needs PDNS). These are real, if easily-manipulated (per paper sec 6.1),
signals -- the point of this PoC is whether adding TDA topology features
(topo_features.py) recovers what the paper's GNN gets from the same gap.
"""
from __future__ import annotations

import math

import networkx as nx

VOWELS = frozenset("aeiou")
SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "work", "support",
    "online", "site", "icu", "cn", "rest", "fit", "cam",
})


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def lexical_features(domain: str) -> dict:
    apex = domain.split(".")[0] if "." in domain else domain
    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
    letters = [c for c in apex if c.isalpha()]
    is_vowel = [c.lower() in VOWELS for c in letters]
    max_consec_consonant = 0
    run = 0
    for v in is_vowel:
        run = run + 1 if not v else 0
        max_consec_consonant = max(max_consec_consonant, run)

    return {
        "lex_length": len(domain),
        "lex_num_minus": apex.count("-"),
        "lex_num_digits": sum(c.isdigit() for c in apex),
        "lex_digit_ratio": sum(c.isdigit() for c in apex) / len(apex) if apex else 0.0,
        "lex_vowel_ratio": sum(is_vowel) / len(letters) if letters else 0.0,
        "lex_entropy": _shannon_entropy(apex),
        "lex_max_consec_consonant": max_consec_consonant,
        "lex_suspicious_tld": int(tld in SUSPICIOUS_TLDS),
    }


def hosting_features(domain: str, g: nx.Graph) -> dict:
    if domain not in g:
        return {"host_num_ips": 0, "host_max_domains_per_ip": 0, "host_subnet_domain_count": 0, "host_has_asn": 0}

    ips = [n for n in g.neighbors(domain) if g.nodes[n].get("kind") == "ip"]
    max_domains_per_ip = 0
    subnet_domain_count = 0
    has_asn = 0
    for ip in ips:
        domains_on_ip = sum(1 for n in g.neighbors(ip) if g.nodes[n].get("kind") == "domain")
        max_domains_per_ip = max(max_domains_per_ip, domains_on_ip)
        for subnet in (n for n in g.neighbors(ip) if g.nodes[n].get("kind") == "subnet"):
            for ip2 in (n for n in g.neighbors(subnet) if g.nodes[n].get("kind") == "ip"):
                subnet_domain_count += sum(1 for n in g.neighbors(ip2) if g.nodes[n].get("kind") == "domain")
            has_asn = int(any(g.nodes[n].get("kind") == "asn" for n in g.neighbors(subnet)))

    return {
        "host_num_ips": len(ips),
        "host_max_domains_per_ip": max_domains_per_ip,
        "host_subnet_domain_count": subnet_domain_count,
        "host_has_asn": has_asn,
    }


def extract_features(domain: str, g: nx.Graph) -> dict:
    return {**lexical_features(domain), **hosting_features(domain, g)}
