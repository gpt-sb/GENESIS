#!/usr/bin/env python3
"""
Phase 2 — Contrastive dataset and tier-weighted sampler.

Implements:
- ContrastiveDataset: loads precomputed embeddings, builds batches with
  potency-graded positive pairs and tier-weighted negative sampling.
- CurriculumSampler: adjusts hard/easy negative fraction by epoch.
- pair_weight_matrix: computes ordinal potency-proximity weights.
"""
import numpy as np
import pandas as pd
import pickle
import json
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict


# --- Tier sampling weights per curriculum phase ---
CURRICULUM = {
    'warmup': {  # epochs 1-5
        'T4': 0.35, 'T3B': 0.35, 'T1': 0.10, 'T2': 0.10,
        'T3A_inactive': 0.05, 'T3C': 0.05,
    },
    'middle': {  # epochs 6-15
        'T4': 0.15, 'T3B': 0.15, 'T1': 0.25, 'T2': 0.20,
        'T3A_inactive': 0.15, 'T3C': 0.10,
    },
    'sharpen': {  # epochs 16-25
        'T4': 0.10, 'T3B': 0.10, 'T1': 0.20, 'T2': 0.15,
        'T3A_inactive': 0.25, 'T3C': 0.20,
    },
    'finetune': {  # epochs 25+
        'T4': 0.05, 'T3B': 0.10, 'T1': 0.15, 'T2': 0.15,
        'T3A_inactive': 0.30, 'T3C': 0.25,
    },
    'uniform': {  # ablation: equal weight across all tiers
        'T4': 1/6, 'T3B': 1/6, 'T1': 1/6, 'T2': 1/6,
        'T3A_inactive': 1/6, 'T3C': 1/6,
    },
}

# Loss weights per tier
TIER_LOSS_WEIGHTS = {
    'T1': 1.0, 'T2': 0.8, 'T3A_inactive': 1.2,
    'T3B': 0.6, 'T3C': 1.0, 'T4': 0.4,
}

# Ordinal pair weight matrix: w[anchor_potency][positive_potency]
POTENCY_PAIR_WEIGHTS = {
    (3, 3): 1.0,  # very_potent <-> very_potent
    (3, 2): 0.7,  # very_potent <-> potent
    (3, 1): 0.4,  # very_potent <-> active
    (2, 3): 0.7,
    (2, 2): 1.0,  # potent <-> potent
    (2, 1): 0.7,  # potent <-> active
    (1, 3): 0.4,
    (1, 2): 0.7,
    (1, 1): 1.0,  # active <-> active
}

CROSS_TARGET_WEIGHT = 0.5


def get_curriculum_phase(epoch):
    if epoch <= 5:
        return 'warmup'
    elif epoch <= 15:
        return 'middle'
    elif epoch <= 25:
        return 'sharpen'
    else:
        return 'finetune'


def compute_pair_weights(potency_classes, target_names, device='cpu'):
    """
    Compute (B, B) pair weight matrix for ordinal contrastive loss.
    Weights based on potency proximity + same-target bonus.
    """
    B = len(potency_classes)
    weights = torch.zeros(B, B, device=device)

    for i in range(B):
        if potency_classes[i] == 0:  # negative
            continue
        for j in range(B):
            if i == j or potency_classes[j] == 0:
                continue
            pi, pj = int(potency_classes[i]), int(potency_classes[j])
            base_w = POTENCY_PAIR_WEIGHTS.get((pi, pj), 0.3)

            # Same-target bonus
            ti = target_names[i] if target_names[i] else ''
            tj = target_names[j] if target_names[j] else ''
            if ti and tj and ti == tj:
                weights[i, j] = base_w
            else:
                weights[i, j] = base_w * CROSS_TARGET_WEIGHT

    return weights


class ContrastiveDataset(Dataset):
    """
    Dataset that yields (embedding, label, potency_class, pchembl_filled, neg_tier, target_name)
    for contrastive training.
    """

    def __init__(self, split_ids, compound_index, emb_mmap, emb_dim=768):
        self.emb_mmap = emb_mmap
        self.emb_dim = emb_dim
        self.compound_index = compound_index

        # Build ID lists
        self.pos_ids = [cid for cid in split_ids['positive_ids'] if cid in compound_index]
        self.neg_ids = [cid for cid in split_ids['negative_ids'] if cid in compound_index]

        # Group negatives by tier for curriculum sampling
        self.neg_by_tier = defaultdict(list)
        for cid in self.neg_ids:
            tier = compound_index[cid]['neg_tier']
            self.neg_by_tier[tier].append(cid)

    def __len__(self):
        return len(self.pos_ids)

    def get_embedding(self, chembl_id):
        info = self.compound_index[chembl_id]
        emb = self.emb_mmap[info['emb_idx']].astype(np.float32)
        return emb

    def get_positive(self, idx):
        cid = self.pos_ids[idx]
        info = self.compound_index[cid]
        emb = self.get_embedding(cid)
        return {
            'embedding': emb,
            'label': 1,
            'potency_class': info['potency_class_num'] or 0,
            'pchembl_filled': info['pchembl_filled'] if info['pchembl_filled'] is not None else float('nan'),
            'neg_tier': '',
            'target_name': info.get('target_name', '') or '',
            'chembl_id': cid,
        }

    def __getitem__(self, idx):
        return self.get_positive(idx)

    def sample_negatives(self, n, curriculum_phase='warmup', rng=None):
        """Sample n negatives according to curriculum tier distribution."""
        if rng is None:
            rng = np.random.default_rng()

        tier_probs = CURRICULUM[curriculum_phase]
        samples = []

        for tier, prob in tier_probs.items():
            n_tier = max(1, int(n * prob))
            pool = self.neg_by_tier.get(tier, [])
            if not pool:
                continue
            chosen = rng.choice(pool, size=min(n_tier, len(pool)), replace=len(pool) < n_tier)
            for cid in chosen:
                info = self.compound_index[cid]
                emb = self.get_embedding(cid)
                samples.append({
                    'embedding': emb,
                    'label': 0,
                    'potency_class': 0,
                    'pchembl_filled': float('nan'),
                    'neg_tier': tier,
                    'target_name': info.get('target_name', '') or '',
                    'chembl_id': cid,
                })

        # Trim or pad to exactly n
        rng.shuffle(samples)
        return samples[:n]


def collate_contrastive_batch(pos_samples, neg_samples, device='cpu'):
    """
    Collate positive and negative samples into a batch.
    Returns dict with tensors + metadata lists.
    """
    all_samples = pos_samples + neg_samples

    embeddings = torch.tensor(
        np.stack([s['embedding'] for s in all_samples]),
        dtype=torch.float32, device=device
    )
    labels = torch.tensor(
        [s['label'] for s in all_samples],
        dtype=torch.float32, device=device
    )
    potency_classes = torch.tensor(
        [s['potency_class'] for s in all_samples],
        dtype=torch.long, device=device
    )
    pchembl_values = torch.tensor(
        [s['pchembl_filled'] for s in all_samples],
        dtype=torch.float32, device=device
    )
    pchembl_mask = ~torch.isnan(pchembl_values)
    pchembl_values = torch.nan_to_num(pchembl_values, nan=0.0)

    neg_tiers = [s['neg_tier'] for s in all_samples]
    target_names = [s['target_name'] for s in all_samples]
    chembl_ids = [s['chembl_id'] for s in all_samples]

    # Compute pair weights
    pair_weights = compute_pair_weights(
        potency_classes.tolist(), target_names, device=device
    )

    # Compute per-sample loss weights for negatives
    neg_weights = torch.ones(len(all_samples), device=device)
    for i, tier in enumerate(neg_tiers):
        if tier:
            neg_weights[i] = TIER_LOSS_WEIGHTS.get(tier, 1.0)

    return {
        'embeddings': embeddings,
        'labels': labels,
        'potency_classes': potency_classes,
        'pchembl_values': pchembl_values,
        'pchembl_mask': pchembl_mask,
        'pair_weights': pair_weights,
        'neg_weights': neg_weights,
        'neg_tiers': neg_tiers,
        'target_names': target_names,
        'chembl_ids': chembl_ids,
    }
