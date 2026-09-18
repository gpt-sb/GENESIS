#!/usr/bin/env python3
"""
Phase 2 — Training loop with curriculum, evaluation, and checkpointing.

Trains the ProjectionHead + auxiliary heads using:
- Ordinal Supervised Contrastive Loss (primary)
- Potency Huber Loss (auxiliary)
- Tier-aware Margin Ranking Loss (auxiliary)

Loss weight schedule:
  Epochs 1-10:  lambda1=0.1, lambda2=0.0  (contrastive structure)
  Epochs 11-20: lambda1=0.3, lambda2=0.2  (add potency + ranking)
  Epochs 20+:   lambda1=0.5, lambda2=0.3  (full objective)
"""
import os
import sys
import json
import time
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure GENESIS/modeling is on the path for local imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import ContrastiveModel
from losses import OrdinalSupConLoss, PotencyLoss, TierMarginRankingLoss
from dataset import (
    ContrastiveDataset, collate_contrastive_batch,
    get_curriculum_phase, TIER_LOSS_WEIGHTS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

# Paths — all relative to GENESIS/
MODELING_DIR = SCRIPT_DIR
GENESIS_DIR = SCRIPT_DIR.parent
INDEX_FILE = MODELING_DIR / 'compound_index.pkl'
SPLITS_FILE = MODELING_DIR / 'splits.json'
EMB_MMAP = GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap'
EMB_IDS = GENESIS_DIR / 'embeddings' / 'compound_ids.npy'
CHECKPOINT_DIR = MODELING_DIR / 'checkpoints'
LOG_DIR = MODELING_DIR / 'logs'


def get_loss_weights(epoch):
    """Stepped loss weight schedule (lambda1, lambda2) for multi-task curriculum.

    Returns (lambda1, lambda2) where:
      total_loss = L_contrastive + lambda1 * L_potency + lambda2 * L_ranking

    Rationale: let OrdinalSupConLoss establish a coherent 128-d geometry on the
    hypersphere first, then gradually phase in auxiliary objectives so they
    refine the space without disrupting the contrastive structure.

    Schedule:
      Epochs  1-10: (0.1, 0.0) — contrastive geometry only; ranking OFF
      Epochs 11-20: (0.3, 0.2) — potency regression + ranking introduced
      Epochs   21+: (0.5, 0.3) — full multi-task objective
    """
    if epoch <= 10:
        return 0.1, 0.0      # contrastive structure only; ranking disabled
    elif epoch <= 20:
        return 0.3, 0.2      # add potency regression + tier margin ranking
    else:
        return 0.5, 0.3      # full objective — all three losses active


def evaluate(model, dataset, device, n_batches=20, batch_pos=64, batch_neg=192):
    """Quick validation evaluation using fixed-seed batches for reproducibility.

    Uses 'sharpen' curriculum (hard negatives) regardless of training phase so
    validation difficulty is consistent across epochs. Reports contrastive and
    potency losses with a fixed lambda1=0.3 weighting (no ranking loss).

    Args:
        n_batches: number of mini-batches to average over (20 × 256 = 5120 samples).
        batch_pos/batch_neg: 64 positives + 192 negatives = 256 per batch.
    """
    model.eval()
    rng = np.random.default_rng(42)  # fixed seed → identical val batches every epoch
    total_loss = 0.0
    total_con = 0.0
    total_pot = 0.0
    n = 0

    con_loss_fn = OrdinalSupConLoss(temperature=0.07)
    pot_loss_fn = PotencyLoss()

    with torch.no_grad():
        for _ in range(n_batches):
            pos_indices = rng.choice(len(dataset), size=min(batch_pos, len(dataset)), replace=False)
            pos_samples = [dataset.get_positive(i) for i in pos_indices]
            # Always use 'sharpen' for validation — tests against hard negatives
            neg_samples = dataset.sample_negatives(batch_neg, 'sharpen', rng)

            batch = collate_contrastive_batch(pos_samples, neg_samples, device)
            z, pchembl_pred = model(batch['embeddings'])

            l_con = con_loss_fn(
                z, batch['labels'], batch['potency_classes'],
                batch['pair_weights'], batch['neg_weights']
            )
            l_pot = pot_loss_fn(pchembl_pred, batch['pchembl_values'], batch['pchembl_mask'])

            total_con += l_con.item()
            total_pot += l_pot.item()
            # Fixed weighting for val loss (independent of epoch schedule)
            total_loss += (l_con + 0.3 * l_pot).item()
            n += 1

    model.train()  # restore training mode (dropout, batchnorm)
    return {
        'val_loss': total_loss / max(n, 1),
        'val_contrastive': total_con / max(n, 1),
        'val_potency': total_pot / max(n, 1),
    }


def train(args):
    device = torch.device(args.device)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log.info('Device: %s', device)
    log.info('Seed: %d', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- Load pre-computed artifacts from Phase 1 ---
    # compound_index: maps ChEMBL compound IDs → memmap row indices, potency
    #   class, pChEMBL values, target names, and negative tier labels.
    log.info('Loading compound index...')
    with open(INDEX_FILE, 'rb') as f:
        compound_index = pickle.load(f)

    # splits: scaffold-aware train/val/test split with PLOS/approved-drug holdouts
    log.info('Loading splits...')
    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    # Memory-mapped 768-d MolDeBERTa embeddings (read-only, float16).
    # MolDeBERTa is frozen — we only train the projection head on top.
    log.info('Loading embeddings memmap...')
    n_compounds = len(np.load(EMB_IDS, allow_pickle=True))
    emb_mmap = np.memmap(EMB_MMAP, dtype='float16', mode='r', shape=(n_compounds, 768))

    # ContrastiveDataset wraps compound_index + memmap; exposes get_positive()
    # for potency-graded actives and sample_negatives() for curriculum-aware
    # negative sampling from tier pools (T1–T4).
    train_dataset = ContrastiveDataset(splits['train'], compound_index, emb_mmap)
    val_dataset = ContrastiveDataset(splits['val'], compound_index, emb_mmap)
    log.info('Train: %d positives, %d negatives', len(train_dataset.pos_ids), len(train_dataset.neg_ids))
    log.info('Val: %d positives, %d negatives', len(val_dataset.pos_ids), len(val_dataset.neg_ids))

    # ContrastiveModel: ProjectionHead (768→512→256→128, L2-normed) +
    #   PotencyRegressor (128→1 pChEMBL prediction). Forward returns (z, pred).
    model = ContrastiveModel(
        input_dim=768, proj_dim=args.proj_dim, dropout=args.dropout
    ).to(device)
    log.info('Model parameters: %d', model.param_count())

    # AdamW with cosine annealing: LR decays from args.lr to 1e-6 over all epochs.
    # Weight decay provides L2 regularization on projection head weights.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Three loss functions (combined as: L_con + λ1*L_pot + λ2*L_rank)
    # - OrdinalSupConLoss: extended SupCon (Khosla 2020) with ordinal potency-
    #   proximity pair weights; pulls same-class compounds together on hypersphere.
    # - PotencyLoss: Huber loss on pChEMBL predictions (masked for NaN negatives);
    #   ensures the projection retains potency-predictive information.
    # - TierMarginRankingLoss: triplet margin loss with tier-dependent margins
    #   (easy negs: 0.3, hard negs: 0.1); enforces separation ordering.
    con_loss_fn = OrdinalSupConLoss(temperature=args.temperature)
    pot_loss_fn = PotencyLoss()
    rank_loss_fn = TierMarginRankingLoss()

    # --- Training loop ---
    # No PyTorch DataLoader — batches are constructed manually because the
    # positive/negative ratio and negative tier distribution must be controlled
    # per-epoch by the curriculum schedule.
    # best_val_loss is used ONLY as the early-stopping patience baseline; it
    # intentionally resets at each curriculum phase transition (see below).
    # It no longer gates checkpoint selection — see the checkpoint-saving
    # comment further down for why (composite val_loss is a poor proxy for
    # downstream model quality in this codebase).
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    prev_phase = None  # tracks curriculum phase transitions

    # Default: 128 positives + 384 negatives = 512 per batch (3:1 neg:pos ratio)
    n_pos_per_batch = args.batch_pos
    n_neg_per_batch = args.batch_size - n_pos_per_batch
    steps_per_epoch = max(1, len(train_dataset) // n_pos_per_batch)

    log.info('Batch: %d pos + %d neg = %d total', n_pos_per_batch, n_neg_per_batch, args.batch_size)
    log.info('Steps per epoch: %d', steps_per_epoch)
    log.info('Starting training for %d epochs...', args.epochs)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        # Curriculum phase determines negative tier sampling distribution:
        #   warmup  (1-5):   70% easy (T4+T3B) — stable early learning
        #   middle  (6-15):  balanced across tiers
        #   sharpen (16-25): 45% hard (T3A+T3C) — refine decision boundary
        #   finetune (25+):  55% hard — maximum emphasis on near-boundary cases
        phase = get_curriculum_phase(epoch)
        if getattr(args, 'uniform_sampling', False):
            phase = 'uniform'  # ablation: bypass curriculum entirely
        lambda1, lambda2 = get_loss_weights(epoch)

        # Phase transition reset: when curriculum shifts to harder negatives,
        # loss temporarily spikes. Without this reset, the model would early-stop
        # right at the transition. Resetting patience + best_val_loss gives the
        # model room to adapt to the new negative distribution.
        if prev_phase is not None and phase != prev_phase:
            log.info('  Phase transition: %s -> %s — resetting patience', prev_phase, phase)
            patience_counter = 0
            best_val_loss = float('inf')  # new baseline for the new regime
        prev_phase = phase

        # Ablation overrides — force specific losses to zero for ablation studies
        if getattr(args, 'no_potency', False):
            lambda1 = 0.0
        if getattr(args, 'no_ranking', False):
            lambda2 = 0.0

        epoch_losses = {'contrastive': 0, 'potency': 0, 'ranking': 0, 'total': 0}
        n_steps = 0

        # Shuffle positive indices
        pos_indices = rng.permutation(len(train_dataset))

        for step in range(steps_per_epoch):
            # Positives: sequential scan through shuffled indices (each compound
            # seen once per epoch). Each positive has a potency_class (1-3) and
            # known pChEMBL value for the regression head.
            start = step * n_pos_per_batch
            end = min(start + n_pos_per_batch, len(train_dataset))
            batch_pos_idx = pos_indices[start:end]
            pos_samples = [train_dataset.get_positive(i) for i in batch_pos_idx]

            # Negatives: freshly sampled each step from tier pools according to
            # curriculum distribution. pChEMBL is NaN (masked in potency loss).
            neg_samples = train_dataset.sample_negatives(n_neg_per_batch, phase, rng)

            if not pos_samples or not neg_samples:
                continue

            # Collate into tensors: embeddings [B,768], labels [B], potency_classes [B],
            # pchembl_values [B], pchembl_mask [B], pair_weights [B,B] (ordinal +
            # same-target bonus), neg_weights [B] (tier-based), neg_tiers [B] (strings).
            batch = collate_contrastive_batch(pos_samples, neg_samples, device)

            # Forward: project 768-d → L2-normed 128-d (z) + pChEMBL scalar
            z, pchembl_pred = model(batch['embeddings'])

            # --- Loss 1 (primary): Ordinal Supervised Contrastive Loss ---
            # Pairwise cosine_sim / temperature on the L2-normed z vectors.
            # pair_weights modulate attraction strength by ordinal potency proximity
            # (e.g., class 3↔3 = 1.0, class 3↔2 = 0.7, cross-target = ×0.5).
            # neg_weights apply tier-based per-sample weighting.
            # Baseline (--flat-supcon): override to uniform 1.0 for all pos-pos pairs.
            effective_pair_weights = batch['pair_weights']
            if getattr(args, 'flat_supcon', False):
                pos_mask_2d = (batch['labels'].unsqueeze(0) * batch['labels'].unsqueeze(1)).float()
                eye = torch.eye(pos_mask_2d.shape[0], device=device)
                effective_pair_weights = pos_mask_2d * (1 - eye)
            l_con = con_loss_fn(
                z, batch['labels'], batch['potency_classes'],
                effective_pair_weights, batch['neg_weights']
            )

            # --- Loss 2 (auxiliary): Potency Huber Regression ---
            # Huber loss (delta=1.0) on pChEMBL predictions vs ground truth.
            # pchembl_mask zeros out negatives (NaN pChEMBL) so only positives
            # with known potency values contribute to this loss.
            l_pot = pot_loss_fn(pchembl_pred, batch['pchembl_values'], batch['pchembl_mask'])

            # --- Loss 3 (auxiliary): Tier-aware Margin Ranking Loss ---
            # Only computed when lambda2 > 0 (epochs 11+) to save compute.
            # Constructs random (anchor, different-positive, negative) triplets and
            # enforces: sim(anchor, pos) > sim(anchor, neg) + margin.
            # Margins are tier-dependent: easy negs (T3B/T4) → 0.3 (must be well
            # separated), hard negs (T3A/T3C) → 0.1 (allowed to be closer).
            l_rank = torch.tensor(0.0, device=device)
            if lambda2 > 0:
                pos_mask = batch['labels'] == 1
                neg_mask = batch['labels'] == 0
                n_pos = pos_mask.sum().item()
                n_neg = neg_mask.sum().item()
                # Need ≥ 2 positives (anchor + different positive) and ≥ 1 negative
                if n_pos >= 2 and n_neg >= 1:
                    z_pos = z[pos_mask]
                    z_neg = z[neg_mask]
                    n_triplets = min(n_pos, n_neg)
                    # Random triplet construction: two distinct positives + one negative
                    perm_pos = torch.randperm(n_pos, device=device)[:n_triplets]
                    perm_pos2 = torch.randperm(n_pos, device=device)[:n_triplets]
                    # Ensure anchor ≠ positive (different compounds in the triplet)
                    same = perm_pos == perm_pos2
                    perm_pos2[same] = (perm_pos2[same] + 1) % n_pos
                    perm_neg = torch.randperm(n_neg, device=device)[:n_triplets]

                    # Pass tier labels so the loss function can select margin per negative
                    neg_indices = torch.where(neg_mask)[0]
                    triplet_neg_tiers = [batch['neg_tiers'][neg_indices[k].item()] for k in perm_neg]

                    l_rank = rank_loss_fn(
                        z_pos[perm_pos], z_pos[perm_pos2],
                        z_neg[perm_neg], triplet_neg_tiers
                    )

            # Combined multi-task loss: L_con + λ1·L_pot + λ2·L_rank
            # Gradients from all three losses flow back through the shared
            # ProjectionHead — contrastive shapes geometry, potency preserves
            # pChEMBL information, ranking enforces separation margins.
            loss = l_con + lambda1 * l_pot + lambda2 * l_rank

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents loss spikes from hard-negative batches
            # in late curriculum phases from destabilizing training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses['contrastive'] += l_con.item()
            epoch_losses['potency'] += l_pot.item()
            epoch_losses['ranking'] += l_rank.item()
            epoch_losses['total'] += loss.item()
            n_steps += 1

        # Cosine annealing LR step (decays from args.lr → 1e-6 over all epochs)
        scheduler.step()

        # Average epoch losses across all steps
        for k in epoch_losses:
            epoch_losses[k] /= max(n_steps, 1)

        # Validation on held-out set (always uses 'sharpen' curriculum for
        # consistent difficulty measurement across epochs)
        val_metrics = evaluate(model, val_dataset, device)
        elapsed = time.time() - epoch_start

        log.info(
            'Epoch %d/%d [%s] (%.0fs) | '
            'train: total=%.4f con=%.4f pot=%.4f rank=%.4f | '
            'val: loss=%.4f con=%.4f pot=%.4f | '
            'l1=%.1f l2=%.1f lr=%.2e',
            epoch, args.epochs, phase, elapsed,
            epoch_losses['total'], epoch_losses['contrastive'],
            epoch_losses['potency'], epoch_losses['ranking'],
            val_metrics['val_loss'], val_metrics['val_contrastive'],
            val_metrics['val_potency'],
            lambda1, lambda2, scheduler.get_last_lr()[0],
        )

        # Record history
        record = {
            'epoch': epoch, 'phase': phase, 'lambda1': lambda1, 'lambda2': lambda2,
            **{f'train_{k}': v for k, v in epoch_losses.items()},
            **val_metrics,
            'lr': scheduler.get_last_lr()[0], 'elapsed_s': elapsed,
        }
        history.append(record)

        # Per-epoch checkpoint (includes optimizer/scheduler state for resumption)
        ckpt_path = CHECKPOINT_DIR / f'epoch_{epoch:03d}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_metrics['val_loss'],
            'args': vars(args),
        }, ckpt_path)

        # Checkpoint selection: best.pt is ALWAYS the most recently completed
        # epoch (unconditional overwrite), not the epoch with lowest val_loss.
        #
        # EMPIRICAL JUSTIFICATION (2026-09-13 epoch scan, see modeling/logs/
        # epoch_scan_results*.json): val_loss (contrastive + fixed-weight potency
        # loss, never includes the ranking loss) is a POOR proxy for downstream
        # quality here. Scanning epochs 10-40 on the actual scientific metrics
        # (Spearman ρ, cliff-pair accuracy, target-switch AUROC, NDCG@100) showed
        # these metrics improve monotonically through the 'sharpen' phase even as
        # composite val_loss gets worse (contrastive val loss rises under harder
        # negatives faster than potency val loss falls). The epoch with the true
        # minimum val_loss (~epoch 10) scored WORSE on every downstream metric
        # than later epochs. The best composite score across the full 40-epoch
        # scan was epoch 25 — which is exactly where the default min_epochs=25
        # floor naturally concludes training. So: just keep the last completed
        # epoch. best.pt is the checkpoint consumed by all downstream modules
        # (generative RL, repurposing, lead-opt, validation).
        best_path = CHECKPOINT_DIR / 'best.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_metrics['val_loss'],
            'args': vars(args),
        }, best_path)

        # Early stopping with phase-aware patience (still keyed off val_loss as
        # a stopping SIGNAL only — it decides WHEN to stop, not WHICH checkpoint
        # to keep; the checkpoint saved above is always the current/last epoch
        # regardless of this counter). Patience resets at phase transitions so
        # the model gets a fresh window to adapt to harder curriculum negatives.
        # min_epochs (default 25) ensures the model trains through at least the
        # warmup + middle + sharpen curriculum before early stopping can trigger.
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if epoch >= args.min_epochs and patience_counter >= args.patience:
                log.info('Early stopping at epoch %d (patience=%d)', epoch, args.patience)
                break

    # Save training history
    hist_path = LOG_DIR / f'history_seed{args.seed}.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    log.info('Training complete. History -> %s', hist_path)
    log.info('Final epoch: %d, val_loss=%.4f (best.pt = last completed epoch, not min val_loss)',
              history[-1]['epoch'], history[-1]['val_loss'])

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train contrastive projection head')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--batch-pos', type=int, default=128,
                        help='Number of positives per batch')
    parser.add_argument('--proj-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--min-epochs', type=int, default=25,
                        help='Minimum epochs before early stopping activates')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--no-ranking', action='store_true',
                        help='Ablation: disable ranking loss (lambda2=0 throughout)')
    parser.add_argument('--no-potency', action='store_true',
                        help='Ablation: disable potency regression branch')
    parser.add_argument('--uniform-sampling', action='store_true',
                        help='Ablation: uniform negative sampling (no curriculum)')
    parser.add_argument('--flat-supcon', action='store_true',
                        help='Baseline: flat SupCon (all pair weights=1.0, no ordinal weighting)')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
