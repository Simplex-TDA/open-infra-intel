"""Plain-PyTorch GraphSAGE encoder + semi-supervised node classifier,
replicating the paper's own GNN (Sec 4.4/5.1): 3 layers, 256-dim embeddings,
a final layer that concatenates all layer embeddings (JK-style), trained
end-to-end on labeled domain nodes. Full-batch (no neighbor sampling) --
our hosting graphs are small enough that exact neighbor aggregation is
cheap and strictly more accurate than a sampled approximation; the paper
only needed sampling because its PDNS-scale graphs were far larger.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

KINDS = ("domain", "ip", "subnet", "asn")


def build_node_features(
    g: nx.Graph, domain_feature_cols: list[str], domain_features: dict[str, dict]
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Node feature matrix: one-hot kind + log-degree for every node, plus
    `domain_feature_cols` (e.g. lex_/host_/topo_ columns) for domain nodes
    only (zero elsewhere) -- the paper assigns its lexical/hosting/IP
    features the same way, per node type."""
    nodes = list(g.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    degrees = dict(g.degree())
    X = np.zeros((len(nodes), len(KINDS) + 1 + len(domain_feature_cols)), dtype=np.float32)
    for i, n in enumerate(nodes):
        kind = g.nodes[n].get("kind")
        if kind in KINDS:
            X[i, KINDS.index(kind)] = 1.0
        X[i, len(KINDS)] = np.log1p(degrees[n])
        if kind == "domain" and n in domain_features:
            row = domain_features[n]
            X[i, len(KINDS) + 1 :] = [row.get(c, 0.0) for c in domain_feature_cols]
    edges = [(index[u], index[v]) for u, v in g.edges()]
    edge_index = torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long).t()
    return nodes, torch.from_numpy(X), edge_index


class SAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        n = h.size(0)
        summed = torch.zeros(n, h.size(1), dtype=h.dtype).index_add_(0, dst, h[src])
        counts = (
            torch.zeros(n, dtype=h.dtype)
            .index_add_(0, dst, torch.ones(src.size(0), dtype=h.dtype))
            .clamp(min=1)
            .unsqueeze(1)
        )
        return torch.relu(self.self_lin(h) + self.neigh_lin(summed / counts))


class GraphSAGEClassifier(nn.Module):
    """3-layer GraphSAGE encoder (paper Sec 4.4), JK-concat final layer, linear head."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(SAGELayer(dims[i], dims[i + 1]) for i in range(n_layers))
        self.head = nn.Linear(hidden_dim * n_layers, 2)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        outs = []
        for layer in self.layers:
            h = layer(h, edge_index)
            outs.append(h)
        return self.head(torch.cat(outs, dim=1))


def _fpr(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def train_and_evaluate(
    nodes: list[str],
    X: torch.Tensor,
    edge_index: torch.Tensor,
    labels: dict[str, int],
    test_size: float = 0.3,
    seed: int = 0,
    epochs: int = 200,
    lr: float = 0.01,
    hidden_dim: int = 256,
    weight_decay: float = 0.0,
) -> dict:
    """Transductive semi-supervised node classification: one forward pass over
    the whole graph per epoch, loss only on the training domain nodes --
    matching mantis.classify.evaluate()'s return shape for direct comparison."""
    torch.manual_seed(seed)
    domain_idx = [i for i, n in enumerate(nodes) if n in labels]
    y_all = np.array([labels[nodes[i]] for i in domain_idx])
    train_pos, test_pos = train_test_split(
        np.arange(len(domain_idx)), test_size=test_size, random_state=seed, stratify=y_all
    )
    train_idx = torch.tensor([domain_idx[p] for p in train_pos])
    test_idx = torch.tensor([domain_idx[p] for p in test_pos])
    y_train = torch.tensor(y_all[train_pos], dtype=torch.long)
    y_test = y_all[test_pos]

    model = GraphSAGEClassifier(X.size(1), hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X, edge_index)[train_idx], y_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(X, edge_index)[test_idx], dim=1)[:, 1].numpy()
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "fpr": _fpr(y_test, pred),
    }
