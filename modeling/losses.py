#!/usr/bin/env python3
"""
Phase 2 — Loss functions for potency-graded contrastive learning.

OrdinalSupConLoss: Supervised contrastive loss with potency-proximity weights.
PotencyLoss: Huber loss on pchembl predictions.
TierMarginRankingLoss: Tier-aware margin ranking loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalSupConLoss(nn.Module):
    """
    Ordinal Supervised Contrastive Loss.

    Extends SupCon (Khosla et al. NeurIPS 2020) with ordinal potency weights.
    For each anchor i, positives in P(i) get weight w_ip based on potency proximity,
    and the denominator includes all non-anchor samples.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, labels, potency_classes, pair_weights, neg_weights=None):
        """
        Args:
            z: (B, D) L2-normalized projections
            labels: (B,) 1 for positive, 0 for negative
            potency_classes: (B,) potency class (1,2,3) for positives, 0 for negatives
            pair_weights: (B, B) pairwise weight matrix (w_ip from ordinal policy)
            neg_weights: (B,) per-sample weight for tier-based weighting (optional)
        Returns:
            scalar loss
        """
        device = z.device
        B = z.shape[0]

        # Pairwise cosine similarity (z is already L2-normalised)
        sim = torch.mm(z, z.t()) / self.temperature  # (B, B)

        # Mask: positive pairs (both are positives, excluding self)
        pos_mask = labels.unsqueeze(0) * labels.unsqueeze(1)  # (B, B)
        self_mask = torch.eye(B, device=device).bool()
        pos_mask = pos_mask.float() * (~self_mask).float()

        # Apply ordinal weights to positive pairs
        weighted_pos_mask = pos_mask * pair_weights

        # All-pairs mask (exclude self)
        all_mask = (~self_mask).float()

        # Apply negative weights if provided
        if neg_weights is not None:
            neg_mask = (1 - labels.float()).unsqueeze(0).expand(B, B) * all_mask
            weight_matrix = torch.ones(B, B, device=device)
            weight_matrix = weight_matrix * (1 - neg_mask) + neg_mask * neg_weights.unsqueeze(0)
        else:
            weight_matrix = torch.ones(B, B, device=device)

        # Log-sum-exp for denominator (numerical stability)
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max

        # Weighted denominator
        exp_logits = torch.exp(logits) * all_mask * weight_matrix
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Weighted mean of log probabilities over positive pairs
        weighted_log_prob = weighted_pos_mask * log_prob
        num_positives = weighted_pos_mask.sum(dim=1)

        # Only compute loss for anchors that have at least one positive
        valid = num_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(weighted_log_prob.sum(dim=1)[valid] / num_positives[valid]).mean()
        return loss


class PotencyLoss(nn.Module):
    """Huber loss on predicted pchembl values."""

    def __init__(self, delta=1.0):
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta)

    def forward(self, pchembl_pred, pchembl_true, mask):
        """
        Args:
            pchembl_pred: (B,) predicted pchembl
            pchembl_true: (B,) true pchembl_filled
            mask: (B,) bool, True where pchembl_true is available
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pchembl_pred.device, requires_grad=True)
        return self.loss_fn(pchembl_pred[mask], pchembl_true[mask])


class TierMarginRankingLoss(nn.Module):
    """
    Margin ranking loss with tier-dependent margins.

    For triplets (anchor, positive, negative), ensures:
        sim(anchor, positive) > sim(anchor, negative) + margin
        where margin depends on negative tier hardness.
    """

    TIER_MARGINS = {
        'T3A_inactive': 0.1,  # Gold/hard: tight margin
        'T3C': 0.1,           # Hard: tight margin
        'T1': 0.2,            # Hard: clear separation
        'T2': 0.2,            # Medium: clear separation
        'T3B': 0.3,           # Soft: easy separation
        'T4': 0.3,            # Easy: easy separation
    }
    DEFAULT_MARGIN = 0.2

    def forward(self, z_anchor, z_positive, z_negative, neg_tiers):
        """
        Args:
            z_anchor: (N, D) anchor projections
            z_positive: (N, D) positive projections
            z_negative: (N, D) negative projections
            neg_tiers: list of N tier strings
        """
        if len(z_anchor) == 0:
            return torch.tensor(0.0, device=z_anchor.device, requires_grad=True)

        pos_sim = (z_anchor * z_positive).sum(dim=1)
        neg_sim = (z_anchor * z_negative).sum(dim=1)

        margins = torch.tensor(
            [self.TIER_MARGINS.get(t, self.DEFAULT_MARGIN) for t in neg_tiers],
            device=z_anchor.device, dtype=z_anchor.dtype
        )

        loss = F.relu(margins + neg_sim - pos_sim).mean()
        return loss
