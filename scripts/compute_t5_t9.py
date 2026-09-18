#!/usr/bin/env python3
"""
Compute T6–T9 metrics for the improved model across 3 seeds.
These metrics were present in the original evaluation but omitted from the
redesigned 6-test battery. Fills the S2 table gaps.

T6: Dose-response ρ — already available from existing evaluation (per_target_mean_rho)
T7: Cross-pathway AUC — VP centroid from one target family ranks related families
T8: Hardness-stratified pair accuracy — VP-vs-A, VP-vs-P, P-vs-A ordering
T9: Silhouette score — unsupervised clustering quality in projected space
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import silhouette_score, roc_auc_score

MODELING_DIR = Path(__file__).resolve().parent.parent / 'modeling'
GENESIS_DIR = MODELING_DIR.parent
sys.path.insert(0, str(MODELING_DIR))

from model import ContrastiveModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

VALIDATION_DIR = MODELING_DIR / 'validation' / 'multiseed'


def load_model(checkpoint_path, device='mps'):
    import torch
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    proj_dim = ckpt.get('args', {}).get('proj_dim', 128)
    model = ContrastiveModel(input_dim=768, proj_dim=proj_dim, dropout=0.1)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def project_ids(model, ids, compound_index, emb_mmap, device):
    import torch
    embs = []
    for cid in ids:
        info = compound_index[cid]
        embs.append(emb_mmap[info['emb_idx']].astype(np.float32))
    emb_tensor = torch.tensor(np.stack(embs), dtype=torch.float32, device=device)
    with torch.no_grad():
        z, _ = model(emb_tensor)
    return z.cpu().numpy()


def test_cross_pathway_auc(model, splits, compound_index, emb_mmap, device):
    """T7: For each OSCC target family, compute VP centroid and measure
    AUC for ranking held-out compounds from RELATED target families."""
    log.info('=== T7: Cross-Pathway AUC ===')

    # Group test positives by target
    target_groups = defaultdict(list)
    for cid in splits['test']['positive_ids']:
        if cid not in compound_index:
            continue
        target = compound_index[cid].get('target_name', '')
        if target:
            target_groups[target].append(cid)

    # Also group training positives for centroid
    train_target_groups = defaultdict(list)
    for cid in splits['train']['positive_ids']:
        if cid not in compound_index:
            continue
        target = compound_index[cid].get('target_name', '')
        pcn = compound_index[cid].get('potency_class_num', 0)
        if target and pcn == 3:
            train_target_groups[target].append(cid)

    test_neg_ids = [cid for cid in splits['test']['negative_ids'] if cid in compound_index]

    aucs = []
    for target, train_vp_ids in sorted(train_target_groups.items(), key=lambda x: -len(x[1])):
        if len(train_vp_ids) < 5:
            continue

        # Centroid from this target's training VP compounds
        z_centroid_set = project_ids(model, train_vp_ids, compound_index, emb_mmap, device)
        centroid = z_centroid_set.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        # Test: other targets' positives (cross-pathway) vs negatives
        other_pos_ids = []
        for t2, ids2 in target_groups.items():
            if t2 != target:
                other_pos_ids.extend(ids2)

        if len(other_pos_ids) < 10:
            continue

        # Score: cosine similarity to centroid
        eval_ids = other_pos_ids + test_neg_ids[:len(other_pos_ids)]
        labels = [1] * len(other_pos_ids) + [0] * min(len(test_neg_ids), len(other_pos_ids))

        z_eval = project_ids(model, eval_ids, compound_index, emb_mmap, device)
        scores = z_eval @ centroid

        try:
            auc = roc_auc_score(labels, scores)
            aucs.append(auc)
            log.info('  %s: AUC=%.3f (n_other_pos=%d)', target, auc, len(other_pos_ids))
        except ValueError:
            pass

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    log.info('  Cross-pathway mean AUC: %.4f (across %d targets)', mean_auc, len(aucs))
    return {'cross_pathway_auc': mean_auc, 'cross_pathway_n_targets': len(aucs)}


def test_hardness_stratified_pairs(model, splits, compound_index, emb_mmap, device):
    """T8: Pair ordering accuracy stratified by difficulty.
    - Easy: VP vs Active (large potency gap)
    - Moderate: VP vs Potent (small gap at top)
    - Hard: Potent vs Active (small gap at bottom)
    """
    log.info('=== T8: Hardness-Stratified Pair Accuracy ===')

    test_pos_ids = [cid for cid in splits['test']['positive_ids'] if cid in compound_index]
    by_class = defaultdict(list)
    for cid in test_pos_ids:
        pcn = compound_index[cid].get('potency_class_num', 0)
        by_class[pcn].append(cid)

    results = {}
    rng = np.random.default_rng(42)

    # VP centroid from training
    vp_train_ids = [cid for cid in splits['train']['positive_ids']
                    if cid in compound_index and compound_index[cid].get('potency_class_num') == 3]
    z_vp_train = project_ids(model, vp_train_ids, compound_index, emb_mmap, device)
    centroid = z_vp_train.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    pairs = [
        ('easy_vp_vs_a', 3, 1, 'VP vs Active'),
        ('hard_vp_vs_p', 3, 2, 'VP vs Potent'),
        ('hard_p_vs_a', 2, 1, 'Potent vs Active'),
    ]

    for key, hi_class, lo_class, label in pairs:
        hi_ids = by_class.get(hi_class, [])
        lo_ids = by_class.get(lo_class, [])
        if not hi_ids or not lo_ids:
            results[key] = None
            continue

        # Sample pairs (up to 5000)
        n_pairs = min(5000, len(hi_ids) * len(lo_ids))
        correct = 0
        total = 0

        z_hi = project_ids(model, hi_ids, compound_index, emb_mmap, device)
        z_lo = project_ids(model, lo_ids, compound_index, emb_mmap, device)

        sim_hi = z_hi @ centroid
        sim_lo = z_lo @ centroid

        # All pairs (capped)
        for _ in range(n_pairs):
            i = rng.integers(len(hi_ids))
            j = rng.integers(len(lo_ids))
            if sim_hi[i] > sim_lo[j]:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        results[key] = float(acc)
        log.info('  %s: %.1f%% (%d/%d pairs)', label, 100 * acc, correct, total)

    return results


def test_silhouette(model, splits, compound_index, emb_mmap, device):
    """T9: Silhouette score — clustering quality in projected space.
    Uses potency class as cluster label."""
    log.info('=== T9: Silhouette Score ===')

    test_pos_ids = [cid for cid in splits['test']['positive_ids'] if cid in compound_index]
    test_neg_ids = [cid for cid in splits['test']['negative_ids'] if cid in compound_index]

    # Sample negatives to balance (max 2000)
    rng = np.random.default_rng(42)
    if len(test_neg_ids) > 2000:
        test_neg_ids = list(rng.choice(test_neg_ids, 2000, replace=False))

    all_ids = test_pos_ids + test_neg_ids
    labels = []
    for cid in test_pos_ids:
        labels.append(compound_index[cid].get('potency_class_num', 0))
    for _ in test_neg_ids:
        labels.append(0)
    labels = np.array(labels)

    z_all = project_ids(model, all_ids, compound_index, emb_mmap, device)

    # Weighted silhouette (by class)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {'silhouette_weighted': 0.0}

    sil = silhouette_score(z_all, labels, metric='cosine')
    log.info('  Silhouette (weighted, cosine): %.4f', sil)
    return {'silhouette_weighted': float(sil)}


def main():
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)
    n_compounds = len(np.load(GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True))
    emb_mmap = np.memmap(
        GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
        dtype='float16', mode='r', shape=(n_compounds, 768)
    )

    seeds = [42, 123, 456]
    all_results = {}

    for seed in seeds:
        ckpt_path = VALIDATION_DIR / 'checkpoints' / f'best_seed{seed}.pt'
        if not ckpt_path.exists():
            log.warning('Checkpoint not found: %s', ckpt_path)
            continue

        log.info('\n=== Seed %d ===', seed)
        model = load_model(str(ckpt_path), device)

        seed_results = {}

        # T6: already available — pull from existing evaluation
        eval_path = VALIDATION_DIR / 'logs' / f'evaluation_seed{seed}.json'
        if eval_path.exists():
            with open(eval_path) as f:
                existing = json.load(f)
            seed_results['dose_response_rho_global'] = existing.get('global_spearman_rho')
            seed_results['dose_response_rho_per_target'] = existing.get('per_target_mean_rho')
            log.info('T6: global ρ=%.4f, per-target ρ=%.4f',
                     seed_results['dose_response_rho_global'],
                     seed_results['dose_response_rho_per_target'])

        # T7: Cross-pathway AUC
        t7 = test_cross_pathway_auc(model, splits, compound_index, emb_mmap, device)
        seed_results.update(t7)

        # T8: Hardness-stratified pairs
        t8 = test_hardness_stratified_pairs(model, splits, compound_index, emb_mmap, device)
        seed_results.update(t8)

        # T9: Silhouette
        t9 = test_silhouette(model, splits, compound_index, emb_mmap, device)
        seed_results.update(t9)

        all_results[str(seed)] = seed_results

    # Aggregate
    print("\n" + "=" * 70)
    print("T5-T9 MULTI-SEED RESULTS")
    print("=" * 70)

    metrics_to_report = [
        ('dose_response_rho_global', 'T6 Dose-response ρ (global)'),
        ('dose_response_rho_per_target', 'T6 Dose-response ρ (per-target)'),
        ('cross_pathway_auc', 'T7 Cross-pathway AUC'),
        ('easy_vp_vs_a', 'T8 Easy pairs (VP vs A)'),
        ('hard_vp_vs_p', 'T8 Hard pairs (VP vs P)'),
        ('hard_p_vs_a', 'T8 Hard pairs (P vs A)'),
        ('silhouette_weighted', 'T9 Silhouette (weighted)'),
    ]

    summary = {}
    for key, label in metrics_to_report:
        vals = [all_results[s][key] for s in all_results if key in all_results[s] and all_results[s][key] is not None]
        if vals:
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            summary[key] = {'mean': mean_v, 'std': std_v, 'values': vals}
            print(f"  {label:<35s} {mean_v:.4f} ± {std_v:.4f}")
        else:
            print(f"  {label:<35s} N/A")

    out_path = VALIDATION_DIR / 'logs' / 't5_t9_results.json'
    with open(out_path, 'w') as f:
        json.dump({'per_seed': all_results, 'summary': summary}, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == '__main__':
    main()
