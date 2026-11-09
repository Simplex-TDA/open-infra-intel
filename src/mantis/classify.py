"""Build the feature table and run the paper's own ablation methodology
(Table 4: compare feature sets at a fixed 0.5 threshold) to answer: does
adding TDA topology features to lexical+hosting features close the gap that,
in the paper, only a GNN closes?
"""
from __future__ import annotations

import time

import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from mantis.features import extract_features
from mantis.topo_features import domain_topo_features

MODELS = {
    "random_forest": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed),
    "xgboost": lambda seed: XGBClassifier(eval_metric="logloss", random_state=seed),
}


def build_feature_table(
    domains: list[str],
    labels: dict[str, int],
    g: nx.Graph,
    radius: int = 2,
    max_edges: int | None = None,
    max_degree: int | None = None,
    top_k_centrality: int | None = None,
    progress_every: int | None = None,
) -> pd.DataFrame:
    rows = []
    timing: dict[str, float] = {}
    start = time.monotonic()
    for i, domain in enumerate(domains, 1):
        row = {"domain": domain, "label": labels[domain]}
        row.update(extract_features(domain, g))
        row.update(
            domain_topo_features(
                domain,
                g,
                radius=radius,
                max_edges=max_edges,
                max_degree=max_degree,
                top_k_centrality=top_k_centrality,
                timing=timing,
            )
        )
        rows.append(row)
        if progress_every and i % progress_every == 0:
            elapsed = time.monotonic() - start
            print(
                f"    [{i}/{len(domains)}] {elapsed:.1f}s elapsed "
                f"(curvature={timing.get('curvature', 0.0):.1f}s persistence={timing.get('persistence', 0.0):.1f}s)",
                flush=True,
            )
    return pd.DataFrame(rows)


# Pure neighborhood-size stand-ins: redundant with host_num_ips/
# host_subnet_domain_count, not "shape". Excluded by topo_mode="shape_only"
# to test whether the TDA lift survives on curvature/persistence alone.
_TOPO_SIZE_COLS = {"topo_ego_n_nodes", "topo_ego_n_edges", "topo_diagram_cardinality"}


def feature_columns(df: pd.DataFrame, include_topo: bool, topo_mode: str = "full") -> list[str]:
    prefixes = ("lex_", "host_") + (("topo_",) if include_topo else ())
    cols = [c for c in df.columns if c.startswith(prefixes)]
    if include_topo and topo_mode == "shape_only":
        cols = [c for c in cols if c not in _TOPO_SIZE_COLS]
    return cols


def _fpr(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def evaluate(df: pd.DataFrame, feature_cols: list[str], test_size: float = 0.3, seed: int = 0) -> dict:
    X = df[feature_cols].fillna(0.0)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    results = {}
    for name, make_model in MODELS.items():
        model = make_model(seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "auc": roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "fpr": _fpr(y_test, pred),
        }
    return results


def feature_importances(df: pd.DataFrame, feature_cols: list[str], seed: int = 0) -> dict[str, pd.Series]:
    """Fit each model on the full table and rank features by its native
    importance score, to see which topo_ features (if any) the models
    actually lean on."""
    X = df[feature_cols].fillna(0.0)
    y = df["label"]
    out = {}
    for name, make_model in MODELS.items():
        model = make_model(seed)
        model.fit(X, y)
        out[name] = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return out
