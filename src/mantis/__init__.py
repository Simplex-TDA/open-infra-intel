"""MANTIS-TDA: replicate MANTIS (IEEE S&P 2025) malicious-domain detection on
free, public data (Phishing.Database + Tranco + Team Cymru) since we have no
Farsight PDNS / VirusTotal feed access, and test whether topological
(Ollivier-Ricci curvature + persistent homology) features add anything on top
of the paper's own lexical+hosting features and GraphSAGE GNN encoder.

Consequence of no PDNS: the hosting graph is built from *active* DNS
resolution of our own seed domains only, not a passive-DNS crawl of
everything recently hosted on each IP. "Shared infrastructure" edges are
therefore limited to co-occurrences within our seed set -- a real but weaker
signal than the paper's. See ingest.py docstring.
"""
import os

# networkit (curvature) and torch (gnn) each init their own OpenMP runtime;
# running both multi-threaded in one process deadlocks on import order. Must
# be set before either is imported, so it lives here, at the top of the
# package __init__.
os.environ.setdefault("OMP_NUM_THREADS", "1")
