#!/usr/bin/env python3
"""
Phase 2 — Projection Head + Auxiliary Heads.

Transforms frozen MolDeBERTa embeddings (general chemical space) into a
potency-graded pharmacological space for OSCC drug discovery.

Architecture:
  ProjectionHead:    768 → 512 → 256 → 128 (L2-normalised onto unit hypersphere)
  PotencyRegressor:  128 → 1 (predicted pChEMBL, reads from z)
  ContrastiveModel:  Wraps both heads; single forward returns (z, pchembl_pred)

The 128-d bottleneck is the shared representation consumed by all downstream
modules (generative RL, drug repurposing, lead optimization, validation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """3-layer MLP projection from frozen MolDeBERTa space to contrastive space.

    Transforms 768-d general chemical embeddings into 128-d pharmacological
    embeddings where distance = potency similarity. Three nonlinear layers
    (not one) are needed to *reorganize* the geometry — a single linear layer
    can only rotate/scale, not remap which molecules are neighbors.

    Design choices:
      - LayerNorm (not BatchNorm): batch stats would be noisy due to
        variable curriculum composition (70% easy negs in warmup vs 45%
        hard negs in sharpen).
      - GELU (not ReLU): smoother activation avoids dead neurons in the
        6× bottleneck (768→128); losing neurons here is costly.
      - Dropout(0.1): light regularization — ~34K training compounds vs
        559K parameters (~16 params/compound).
      - L2 normalization: places all embeddings on unit hypersphere so
        cosine similarity = dot product; temperature scaling works uniformly.
    """

    def __init__(self, input_dim=768, hidden1=512, hidden2=256, output_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        """Returns L2-normalised projection."""
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class PotencyRegressor(nn.Module):
    """Auxiliary head: predicts pChEMBL from projected features.

    Intentionally minimal (single linear layer, no activation) — this is a
    *regularizer*, not a predictor. By forcing pChEMBL to be linearly
    decodable from z, we constrain the ProjectionHead to arrange compounds
    with a smooth, monotonic potency gradient in the 128-d space.

    A deeper regressor could untangle a messy projection through its own
    capacity, removing the regularization pressure entirely.

    Reads from z (not raw 768-d) so Huber loss gradients flow back through
    z into the ProjectionHead — the sequential architecture is the mechanism
    that couples the potency objective to the contrastive geometry.
    """

    def __init__(self, input_dim=128):
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(self, z):
        return self.head(z).squeeze(-1)


class ContrastiveModel(nn.Module):
    """Full model: projection head + potency regressor.

    Single forward pass returns (z, pchembl_pred) for computing all three
    losses (contrastive, potency, ranking). The PotencyRegressor reads from
    z (not raw embeddings) so auxiliary gradients shape the projection.

    The trained checkpoint (best.pt) is the shared artifact consumed by 10+
    downstream modules: generative RL, CRAVE, drug repurposing, lead
    optimization, category annotation, and validation pipelines.
    """

    def __init__(self, input_dim=768, proj_dim=128, dropout=0.1):
        super().__init__()
        self.projection = ProjectionHead(
            input_dim=input_dim, output_dim=proj_dim, dropout=dropout
        )
        self.potency_head = PotencyRegressor(input_dim=proj_dim)

    def forward(self, embeddings):
        z = self.projection(embeddings)
        pchembl_pred = self.potency_head(z)
        return z, pchembl_pred

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
