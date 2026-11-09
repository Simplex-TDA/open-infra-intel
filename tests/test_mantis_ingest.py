from mantis.ingest import _apex_domain, parse_cymru_bulk_response


def test_apex_domain_strips_scheme_path_and_subdomain():
    assert _apex_domain("www.example.co.uk") == "example.co.uk"
    assert _apex_domain("not a url") is None


def test_apex_domain_keeps_paas_subdomain_distinct():
    # vercel.app is a PSL "private" suffix (shared PaaS hosting): the
    # meaningful, distinct identifier is the label in front of it, not the
    # shared platform suffix itself.
    assert _apex_domain("https://login.paypal-verify.vercel.app/x") == "paypal-verify.vercel.app"
    assert _apex_domain("https://sub.evil.github.io/path") == "evil.github.io"


def test_parse_cymru_bulk_response():
    text = (
        "Bulk mode; whois.cymru.com [2026-11-06 21:56:43 +0000]\n"
        "15169   | 8.8.8.8          | 8.8.8.0/24          | US | arin     | 2023-12-28 | GOOGLE - Google LLC, US\n"
    )
    out = parse_cymru_bulk_response(text)
    assert out["8.8.8.8"] == {
        "asn": "15169",
        "subnet": "8.8.8.0/24",
        "country": "US",
        "as_name": "GOOGLE - Google LLC, US",
    }
