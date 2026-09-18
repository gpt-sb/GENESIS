#!/usr/bin/env python3
"""
NMI Baseline A2 — Binary BCE Classifier vs. PGCL.

Trains a simple binary classifier (BCE loss) on the same 768-d MolDeBERTa
embeddings: VP=1 (positive), everything else=0 (negative). Uses the same
ProjectionHead architecture as PGCL but replaces the multi-component loss
with standard binary cross-entropy.

This tests whether classification-optimized representations preserve
the potency geometry that PGCL is designed to learn.

Expected finding:
  - BCE achieves good AUROC (optimized for binary discrimination)
  - BCE degrades on Spearman ρ (no potency gradient signal)
  - BCE fails cliff pair accuracy (no activity cliff hard negatives learning)

Usage:
    python GENESIS/modeling/baselines/run_bce_baseline.py --device mps
    python GENESIS/modeling/baselines/run_bce_baseline.py --eval-only --device mps

Outputs:
    GENESIS/modeling/baselines/results/bce_classifier/
        best.pt            — best BCE checkpoint
        eval_results.json  — evaluation results
        comparison.json    — vs PGCL comparison table
"""
import os
import sys
import json
import time
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parent
GENESIS_DIR = MODELING_DIR.parent
RESULTS_DIR = SCRIPT_DIR / 'results' / 'bce_classifier'

sys.path.insert(0, str(MODELING_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

# Scope C fresh multiseed PGCL results (seeds 42/123/456, 2026-09-14, retrained
# with the fixed checkpoint-selection logic — best.pt = last completed epoch,
# not min val_loss), from modeling/validation/multiseed/logs/multiseed_summary.json.
PGCL_RESULTS = {
    'vp_recall_100':         {'mean': 0.0260, 'std': 0.0005},
    'global_spearman_rho':   {'mean': 0.5553, 'std': 0.0028},
    'cliff_accuracy':        {'mean': 0.6638, 'std': 0.0034},
    'target_switch_auc_roc': {'mean': 0.9668, 'std': 0.0012},
    'ndcg_100':              {'mean': 0.9410, 'std': 0.0248},
}


class BCEProjectionHead(nn.Module):
    """Same architecture as PGCL ProjectionHead + binary classifier head."""

    def __init__(self, input_dim=768, hidden1=512, hidden2=256, proj_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, proj_dim),
        )
        self.classifier = nn.Linear(proj_dim, 1)  # binary: VP vs. non-VP

    def forward(self, x):
        z = self.encoder(x)
        z_norm = F.normalize(z, p=2, dim=-1)  # L2-normalize for retrieval
        logit = self.classifier(z).squeeze(-1)
        return z_norm, logit


def train_bce(device: str, seed: int = 42, epochs: int = 40,
              lr: float = 3e-4, batch_size: int = 512):
    """Train BCE classifier on VP vs. non-VP labels."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = RESULTS_DIR / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Load artifacts
    log.info('Loading artifacts...')
    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)
    n_compounds = len(np.load(GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True))
    emb_mmap = np.memmap(
        GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
        dtype='float16', mode='r', shape=(n_compounds, 768)
    )

    dev = torch.device(device)
    model = BCEProjectionHead().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    bce_fn = nn.BCEWithLogitsLoss()

    # Build training pool: VP positives (label=1) + all negatives (label=0)
    train_pos = [cid for cid in splits['train']['positive_ids'] if cid in compound_index]
    train_neg = [cid for cid in splits['train']['negative_ids'] if cid in compound_index]
    # Downsample negatives to 3:1 ratio to match PGCL batch composition
    n_neg_target = min(len(train_neg), 3 * len(train_pos))
    log.info('Train positives: %d, Negatives (sampled): %d', len(train_pos), n_neg_target)

    val_pos = [cid for cid in splits['val']['positive_ids'] if cid in compound_index]
    val_neg = [cid for cid in splits['val']['negative_ids'] if cid in compound_index]

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7
    log_path = RESULTS_DIR / 'train.log'
    history = []

    def get_batch(pos_ids, neg_ids, n_pos=128, n_neg=384):
        pos_sample = rng.choice(len(pos_ids), size=min(n_pos, len(pos_ids)), replace=False)
        neg_sample = rng.choice(len(neg_ids), size=min(n_neg, len(neg_ids)), replace=False)
        batch_ids = [pos_ids[i] for i in pos_sample] + [neg_ids[i] for i in neg_sample]
        labels = [1.0] * len(pos_sample) + [0.0] * len(neg_sample)
        embs = np.stack([emb_mmap[compound_index[cid]['emb_idx']].astype(np.float32)
                         for cid in batch_ids])
        return (torch.tensor(embs, device=dev),
                torch.tensor(labels, dtype=torch.float32, device=dev))

    log.info('Training BCE classifier for %d epochs...', epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        steps = max(1, len(train_pos) // 128)
        epoch_loss = 0.0

        for _ in range(steps):
            x, y = get_batch(train_pos, train_neg)
            z, logit = model(x)
            loss = bce_fn(logit, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= steps

        # Validation
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch(val_pos, val_neg)
            _, logit_val = model(x_val)
            val_loss = bce_fn(logit_val, y_val).item()

        history.append({'epoch': epoch, 'train_loss': epoch_loss, 'val_loss': val_loss})
        log.info('Epoch %3d | train=%.4f | val=%.4f', epoch, epoch_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'val_loss': val_loss, 'args': {'proj_dim': 128}},
                       ckpt_dir / 'best.pt')
        else:
            patience_counter += 1
            if epoch >= 25 and patience_counter >= patience:
                log.info('Early stopping at epoch %d', epoch)
                break

    with open(RESULTS_DIR / 'train_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    log.info('Training complete. Best val loss: %.4f', best_val_loss)
    return True


def evaluate_bce(device: str) -> dict:
    """Evaluate BCE model on the same 6-test battery as PGCL."""
    from evaluate import (
        load_model, project_ids,
        test_potency_ordering, test_activity_cliff, test_target_switch,
        test_oscc_specificity, test_vp_retrieval, test_potency_graded_ranking,
    )

    ckpt_path = RESULTS_DIR / 'checkpoints' / 'best.pt'
    if not ckpt_path.exists():
        log.error('Checkpoint not found at %s', ckpt_path)
        return {}

    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)
    n_compounds = len(np.load(GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True))
    emb_mmap = np.memmap(
        GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
        dtype='float16', mode='r', shape=(n_compounds, 768)
    )

    # Load BCE model with custom class
    dev = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    bce_model = BCEProjectionHead()
    bce_model.load_state_dict(ckpt['model_state_dict'])
    bce_model.to(dev)
    bce_model.eval()

    # Wrap in a duck-type object compatible with evaluate.py's project_ids
    # (expects forward() returning (z, pchembl_pred))
    class _Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            z, logit = self.m(x)
            return z, logit.unsqueeze(-1)

    wrapped = _Wrapper(bce_model).to(dev)
    wrapped.eval()

    results = {}
    log.info('=== BCE Classifier Evaluation Battery ===')
    results['potency_ordering'] = test_potency_ordering(wrapped, splits, compound_index, emb_mmap, device)
    results['activity_cliff']   = test_activity_cliff(wrapped, splits, compound_index, emb_mmap, device)
    results['target_switch']    = test_target_switch(wrapped, splits, compound_index, emb_mmap, device)
    results['oscc_specificity'] = test_oscc_specificity(wrapped, splits, compound_index, emb_mmap, device)
    results['vp_retrieval']     = test_vp_retrieval(wrapped, splits, compound_index, emb_mmap, device)
    results['potency_graded']   = test_potency_graded_ranking(wrapped, splits, compound_index, emb_mmap, device)

    out = RESULTS_DIR / 'eval_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    log.info('Saved to %s', out)
    return results


def print_comparison(bce_results: dict):
    metrics = [
        ('VP Recall@100',      'vp_retrieval',    'vp_recall_100',       'vp_recall_100'),
        ('Potency Spearman ρ', 'potency_ordering', 'global_spearman_rho', 'global_spearman_rho'),
        ('Cliff Accuracy',     'activity_cliff',  'cliff_accuracy',      'cliff_accuracy'),
        ('Target-Switch AUROC','target_switch',   'target_switch_auc_roc','target_switch_auc_roc'),
        ('NDCG@100',           'potency_graded',  'ndcg_100',            'ndcg_100'),
    ]
    print('\n' + '='*80)
    print('NMI Baseline A2: Binary BCE Classifier vs. PGCL (Ordinal SupCon)')
    print('='*80)
    print(f'{"Metric":<28} {"PGCL (Ours)":>15} {"BCE Baseline":>15} {"Δ":>10} {"PGCL wins?":>12}')
    print('-'*80)
    wins = 0
    total = 0
    for label, section, pgcl_key, bce_key in metrics:
        pgcl = PGCL_RESULTS.get(pgcl_key, {})
        bce_val = bce_results.get(section, {}).get(bce_key)
        pgcl_str = f'{pgcl["mean"]:.3f}±{pgcl["std"]:.3f}' if pgcl.get('mean') else 'N/A'
        bce_str  = f'{bce_val:.3f}' if bce_val is not None else 'N/A'
        if bce_val is not None and pgcl.get('mean') is not None:
            delta = pgcl['mean'] - bce_val
            wins += int(delta > 0)
            total += 1
            delta_str = f'{delta:+.3f}'
            win_str = '✓' if delta > 0 else '✗'
        else:
            delta_str, win_str = 'N/A', 'N/A'
        print(f'{label:<28} {pgcl_str:>15} {bce_str:>15} {delta_str:>10} {win_str:>12}')
    print('-'*80)
    print(f'PGCL wins {wins}/{total} metrics vs. binary BCE baseline')
    print('='*80)


def main():
    parser = argparse.ArgumentParser(description='NMI Baseline A2: BCE vs. PGCL')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        ok = train_bce(args.device, seed=args.seed, epochs=args.epochs)
        if not ok:
            sys.exit(1)

    results = evaluate_bce(args.device)
    if results:
        print_comparison(results)
        with open(RESULTS_DIR / 'comparison.json', 'w') as f:
            json.dump({'method': 'BCE classifier', 'results': results,
                       'pgcl_reference': PGCL_RESULTS}, f, indent=2)


if __name__ == '__main__':
    main()
