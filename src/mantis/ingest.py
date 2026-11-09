"""Ground truth + hosting resolution for the MANTIS-TDA PoC.

Malicious seeds: OpenPhish's free feed (no auth, ~hundreds of live phishing
URLs). Benign: a Tranco top-1M sample, skipping the very top ranks so the
benign set isn't pure global brands (paper sec 4.1 makes the same call, to
avoid a classifier that just learns "popular = benign").

No Farsight PDNS access, so hosting resolution is active-DNS only: a plain
`socket.gethostbyname_ex` per domain, then one bulk whois.cymru.com query
(port 43, no key, no rate limit encountered) to map every resolved IP to its
real announced BGP prefix and ASN in a single round trip.
"""
from __future__ import annotations

import io
import random
import socket
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests
import tldextract

OPENPHISH_URL = "https://openphish.com/feed.txt"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
# Free, no-auth, continuously-updated aggregate of OpenPhish/PhishTank/etc.
# Much larger than OpenPhish's live-only feed (~hundreds vs ~hundreds of
# thousands), which is what OpenPhish alone can't give us for scale.
PHISHING_DB_URL = (
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/"
    "master/phishing-domains-ACTIVE.txt"
)

# include_psl_private_domains=True: without it, tldextract's default treats
# PaaS suffixes like vercel.app/github.io as *not* suffixes (falls back to
# the ICANN .app/.io boundary), collapsing distinct phishing subdomains like
# browardcourtpdf-viewer.vercel.app down to the shared "vercel.app" apex.
_TLD_EXTRACT = tldextract.TLDExtract(include_psl_private_domains=True)


def _apex_domain(url_or_host: str) -> str | None:
    ext = _TLD_EXTRACT(url_or_host)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}"


def fetch_openphish_domains() -> list[str]:
    resp = requests.get(OPENPHISH_URL, timeout=30)
    resp.raise_for_status()
    domains = {_apex_domain(line.strip()) for line in resp.text.splitlines() if line.strip()}
    domains.discard(None)
    return sorted(domains)


def fetch_phishing_db_domains(n: int | None = None, seed: int = 0) -> list[str]:
    resp = requests.get(PHISHING_DB_URL, timeout=60)
    resp.raise_for_status()
    domains = {_apex_domain(line.strip()) for line in resp.text.splitlines() if line.strip()}
    domains.discard(None)
    domains = sorted(domains)
    if n is not None and n < len(domains):
        domains = random.Random(seed).sample(domains, n)
    return domains


def fetch_tranco_domains(n: int = 3000, skip: int = 2000) -> list[str]:
    resp = requests.get(TRANCO_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            lines = f.read().decode().splitlines()
    return [line.split(",", 1)[1].strip() for line in lines[skip:skip + n]]


def resolve_domain_ips(domain: str, timeout: float = 3.0) -> list[str]:
    socket.setdefaulttimeout(timeout)
    try:
        return sorted(set(socket.gethostbyname_ex(domain)[2]))
    except (socket.gaierror, socket.timeout, UnicodeError, OSError):
        return []


def resolve_many(domains: list[str], max_workers: int = 15) -> dict[str, list[str]]:
    """DNS resolution is I/O-bound (blocks on socket), so a thread pool gives
    a near-linear speedup over the ~0.5s/domain sequential rate -- needed
    once the malicious set scales past a few hundred domains.

    max_workers=15 (not a higher number): at full scale (~6000 mixed
    malicious+benign domains), 50 workers saturates something (local
    resolver/socket contention, not per-request timeout) and yield drops to
    ~0.36; 15 workers resolves the same set faster *and* at ~0.65 yield.
    Confirmed via isolated full-scale A/B, not a small-batch extrapolation."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(resolve_domain_ips, domains)
    return {d: ips for d, ips in zip(domains, results) if ips}


def parse_cymru_bulk_response(text: str) -> dict[str, dict]:
    """Pure parser, split out from cymru_bulk_asn() so it's testable without
    a network call."""
    out: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("Bulk mode"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 7:
            continue
        asn, ip, bgp_prefix, country, registry, alloc_date, as_name = parts[:7]
        out[ip] = {"asn": asn, "subnet": bgp_prefix, "country": country, "as_name": as_name}
    return out


def cymru_bulk_asn(ips: list[str], timeout: float = 15.0) -> dict[str, dict]:
    """One bulk query for every IP -> {ip: {asn, subnet, country, as_name}}."""
    if not ips:
        return {}
    query = "begin\nverbose\n" + "\n".join(ips) + "\nend\n"
    with socket.create_connection(("whois.cymru.com", 43), timeout=timeout) as s:
        s.sendall(query.encode())
        chunks = []
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return parse_cymru_bulk_response(b"".join(chunks).decode(errors="replace"))
