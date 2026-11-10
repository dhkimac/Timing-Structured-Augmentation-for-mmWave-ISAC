"""Window encoder + attention pooling over the windows of a recording (multiple-instance learning).

Encoder: 3x3 convolutions with batch norm and leaky ReLU, 16/32/64/128 channels at stride 2,
128 at stride 1, then a 1x1 convolution to 64 channels: (1, 64, 64) -> 64 x 4 x 4 = 1024 features.
Pooling: gated attention (Ilse et al., ICML 2018) with hidden size 32, then dropout 0.2 and a
linear classifier. 323,157 parameters.

Ablations (``pooling``): "mean" averages the window embeddings, "vote" classifies every window
and averages the window probabilities; neither has attention parameters.
"""
from __future__ import annotations

import torch
from torch import nn


def block(cin: int, cout: int, stride: int = 2, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.LeakyReLU())


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1, self.block2 = block(1, 16), block(16, 32)
        self.block3, self.block4 = block(32, 64), block(64, 128)
        self.block5 = block(128, 128, stride=1)
        self.out_proj = block(128, 64, stride=1, k=1)

    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return self.out_proj(self.block5(x)).flatten(1)


class GatedAttention(nn.Module):
    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.attention_V = nn.Linear(dim, hidden)
        self.attention_U = nn.Linear(dim, hidden)
        self.attention_w = nn.Linear(hidden, 1)

    def forward(self, h, bag, pos, n_bags):
        s = self.attention_w(torch.tanh(self.attention_V(h)) * torch.sigmoid(self.attention_U(h)))
        grid = torch.full((n_bags, int(pos.max()) + 1), float("-inf"), dtype=s.dtype, device=s.device)
        a = torch.softmax(grid.index_put((bag, pos), s.squeeze(-1)), dim=1)
        hp = h.new_zeros(n_bags, a.shape[1], h.shape[1]).index_put((bag, pos), h)
        return torch.bmm(a.unsqueeze(1), hp).squeeze(1), a[bag, pos]


def bag_mean(v, bag, pos, n_bags):
    """Mean over each recording's windows of a per-window vector (N, D) -> (n_bags, D)."""
    k = int(pos.max()) + 1
    g = v.new_zeros(n_bags, k, v.shape[1]).index_put((bag, pos), v)
    n = torch.zeros(n_bags, k, dtype=v.dtype, device=v.device).index_put(
        (bag, pos), torch.ones(pos.shape[0], dtype=v.dtype, device=v.device)).sum(1)
    return g.sum(1) / n.unsqueeze(1)


class Model(nn.Module):
    def __init__(self, num_classes: int = 4, pooling: str = "attention", dropout: float = 0.2):
        super().__init__()
        self.pooling = pooling
        self.encoder = Encoder()
        self.attention = GatedAttention(1024) if pooling == "attention" else None
        self.classifier = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout)

    def embed(self, h, bag, pos, n_bags):
        """Recording embedding (used for batch-norm re-estimation and the bag output)."""
        if self.pooling == "attention":
            return self.attention(h, bag, pos, n_bags)[0]
        return bag_mean(h, bag, pos, n_bags)

    def window_logits(self, x):
        return self.classifier(self.dropout(self.encoder(x)))

    def forward(self, x, bag, pos, n_bags):
        """Recording logits; for "vote" the mean window probabilities."""
        h = self.encoder(x)
        if self.pooling == "vote":
            return bag_mean(torch.softmax(self.classifier(self.dropout(h)), 1), bag, pos, n_bags)
        return self.classifier(self.dropout(self.embed(h, bag, pos, n_bags)))

    def loss(self, b):
        """Cross-entropy per recording; for "vote" per window, each recording weighted equally."""
        if self.pooling != "vote":
            return nn.functional.cross_entropy(self(b["x"], b["bag"], b["pos"], b["n_bags"]),
                                               b["labels"])
        ce = nn.functional.cross_entropy(self.window_logits(b["x"]), b["labels"][b["bag"]],
                                         reduction="none")
        n = torch.bincount(b["bag"].cpu(), minlength=b["n_bags"]).to(ce.dtype).to(ce.device)
        return (ce * (1.0 / n[b["bag"]])).sum() / float(b["n_bags"])
