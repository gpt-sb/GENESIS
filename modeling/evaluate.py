#!/usr/bin/env python3
"""
Phase 2 — Scientific Evaluation Battery.

6 core tests:
1. Potency ordering (Spearman ρ)
2. Activity cliff challenge (T3A pair ordering)
3. Target-switch challenge (T3C classification AUC)
4. OSCC specificity (Cohen's d — T2 vs positives)
5. Repurposing retrieval (PLOS Recall@K, enrichment factor)
6. Approved drug ranking (AUC-ROC)
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import ContrastiveModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

MODELING_DIR = SCRIPT_DIR
GENESIS_DIR = SCRIPT_DIR.parent
CHECKPOINT_DIR = MODELING_DIR / 'checkpoints'
LOG_DIR = MODELING_DIR / 'logs'
os.makedirs(LOG_DIR, exist_ok=True)


def load_model(checkpoint_path, device='mps', proj_dim=None, dropout=0.1):
    """Load model from checkpoint. Infers proj_dim from saved args."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if proj_dim is None:
        saved_args = ckpt.get('args', {})
        proj_dim = saved_args.get('proj_dim', 128)
    model = ContrastiveModel(input_dim=768, proj_dim=proj_dim, dropout=dropout)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt


def project_ids(model, compound_ids, compound_index, emb_mmap, device, batch_size=4096):
    """Project a list of compound IDs through the model. Returns (N, D) numpy array."""
    projections = []
    for start in range(0, len(compound_ids), batch_size):
        batch_ids = compound_ids[start:start + batch_size]
        embs = []
        for cid in batch_ids:
            info = compound_index[cid]
            embs.append(emb_mmap[info['emb_idx']].astype(np.float32))
        embs = torch.tensor(np.stack(embs), device=device)
        with torch.no_grad():
            z, _ = model(embs)
        projections.append(z.cpu().numpy())
    return np.concatenate(projections, axis=0)


# ---------------------------------------------------------------------------
# Test 1: Potency Ordering
# ---------------------------------------------------------------------------
def test_potency_ordering(model, splits, compound_index, emb_mmap, device):
    """Spearman ρ between projected-space cosine-to-VP-centroid and potency_class_num."""
    log.info('=== Test 1: Potency Ordering ===')
    test_pos_ids = [cid for cid in splits['test']['positive_ids'] if cid in compound_index]

    z = project_ids(model, test_pos_ids, compound_index, emb_mmap, device)
    potencies = np.array([compound_index[cid]['potency_class_num'] for cid in test_pos_ids])

    # Per-target groups
    target_groups = defaultdict(list)
    for i, cid in enumerate(test_pos_ids):
        target = compound_index[cid].get('target_name', '')
        if target:
            target_groups[target].append(i)

    # VP centroid
    vp_mask = potencies == 3
    if vp_mask.sum() > 0:
        centroid = z[vp_mask].mean(axis=0)
    else:
        centroid = z.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    # Global Spearman
    dists = 1.0 - z @ centroid
    rho_global, p_global = stats.spearmanr(-dists, potencies)
    log.info('Global Spearman ρ: %.4f (p=%.2e)', rho_global, p_global)

    # Per-target Spearman
    rhos = []
    for target, indices in target_groups.items():
        if len(indices) < 5:
            continue
        idx = np.array(indices)
        r, _ = stats.spearmanr(-dists[idx], potencies[idx])
        if not np.isnan(r):
            rhos.append(r)
    if rhos:
        log.info('Per-target Spearman ρ: mean=%.4f, median=%.4f, n_targets=%d',
                 np.mean(rhos), np.median(rhos), len(rhos))

    return {'global_spearman_rho': float(rho_global), 'global_spearman_p': float(p_global),
            'per_target_mean_rho': float(np.mean(rhos)) if rhos else None,
            'n_targets_tested': len(rhos)}


# ---------------------------------------------------------------------------
# Test 2: Activity Cliff Challenge
# ---------------------------------------------------------------------------
def test_activity_cliff(model, splits, compound_index, emb_mmap, device):
    """T3A held-out activity cliff pair ordering.
    For each structurally similar active/inactive pair, test whether the model
    assigns higher similarity-to-VP-centroid to the active compound.
    """
    log.info('=== Test 2: Activity Cliff Challenge ===')
    pairs_file = GENESIS_DIR / 'negatives' / 'tier3a_activity_cliff_pairs.tsv'
    df = pd.read_csv(pairs_file, sep='\t', low_memory=False)

    active_ids_col = df['active_chembl_id'].tolist()
    inactive_ids_col = df['inactive_chembl_id'].tolist()
    all_pair_ids = list(set(active_ids_col + inactive_ids_col))
    valid_ids = [cid for cid in all_pair_ids if cid in compound_index]

    if not valid_ids:
        log.info('No valid cliff pair compounds in index')
        return {'cliff_accuracy': None}

    z_all = project_ids(model, valid_ids, compound_index, emb_mmap, device)
    id_to_z = dict(zip(valid_ids, z_all))

    # VP centroid from train
    vp_ids = [cid for cid in splits['train']['positive_ids']
              if cid in compound_index and compound_index[cid].get('potency_class_num') == 3]
    z_vp = project_ids(model, vp_ids, compound_index, emb_mmap, device)
    centroid = z_vp.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    correct = 0
    total = 0
    for _, row in df.iterrows():
        active_id = row['active_chembl_id']
        inactive_id = row['inactive_chembl_id']
        if active_id not in id_to_z or inactive_id not in id_to_z:
            continue
        sim_active = np.dot(id_to_z[active_id], centroid)
        sim_inactive = np.dot(id_to_z[inactive_id], centroid)
        if sim_active > sim_inactive:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    log.info('Activity cliff pair accuracy: %d/%d = %.1f%%', correct, total, 100 * accuracy)
    return {'cliff_accuracy': float(accuracy), 'cliff_pairs_tested': total}


# ---------------------------------------------------------------------------
# Test 3: Target-Switch Challenge
# ---------------------------------------------------------------------------
def test_target_switch(model, splits, compound_index, emb_mmap, device):
    """T3C target-switch challenge — binary classification AUC."""
    log.info('=== Test 3: Target-Switch Challenge ===')
    t3c_ids = [cid for cid in splits['test']['negative_ids']
               if cid in compound_index and compound_index[cid].get('neg_tier') == 'T3C']
    test_pos_ids = [cid for cid in splits['test']['positive_ids'] if cid in compound_index]

    if not t3c_ids or not test_pos_ids:
        log.info('Insufficient T3C or positive test data')
        return {'target_switch_auc': None}

    z_pos = project_ids(model, test_pos_ids, compound_index, emb_mmap, device)
    z_t3c = project_ids(model, t3c_ids, compound_index, emb_mmap, device)

    centroid = z_pos.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    pos_scores = z_pos @ centroid
    t3c_scores = z_t3c @ centroid

    from sklearn.metrics import roc_auc_score, average_precision_score
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(t3c_scores))])
    y_scores = np.concatenate([pos_scores, t3c_scores])
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)

    log.info('Target-switch AUC-ROC: %.4f, AUC-PR: %.4f', auc_roc, auc_pr)
    log.info('  Positives: %d, T3C: %d', len(pos_scores), len(t3c_scores))
    return {'target_switch_auc_roc': float(auc_roc), 'target_switch_auc_pr': float(auc_pr)}


# ---------------------------------------------------------------------------
# Test 4: OSCC Specificity
# ---------------------------------------------------------------------------
def test_oscc_specificity(model, splits, compound_index, emb_mmap, device):
    """OSCC actives vs other-cancer actives (T2) — Cohen's d separation."""
    log.info('=== Test 4: OSCC Specificity ===')
    test_pos_ids = [cid for cid in splits['test']['positive_ids'] if cid in compound_index]
    t2_ids = [cid for cid in splits['test']['negative_ids']
              if cid in compound_index and compound_index[cid].get('neg_tier') == 'T2']

    if not t2_ids or not test_pos_ids:
        return {'specificity_p_value': None}

    z_pos = project_ids(model, test_pos_ids, compound_index, emb_mmap, device)
    z_t2 = project_ids(model, t2_ids, compound_index, emb_mmap, device)

    centroid = z_pos.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    sim_pos = z_pos @ centroid
    sim_t2 = z_t2 @ centroid

    u_stat, p_value = stats.mannwhitneyu(sim_pos, sim_t2, alternative='greater')
    pooled_std = np.sqrt((sim_pos.std()**2 + sim_t2.std()**2) / 2)
    cohens_d = (sim_pos.mean() - sim_t2.mean()) / (pooled_std + 1e-12)

    log.info("Mann-Whitney U p=%.2e, Cohen's d=%.4f", p_value, cohens_d)
    log.info('  OSCC mean sim: %.4f, T2 mean sim: %.4f', sim_pos.mean(), sim_t2.mean())
    return {'specificity_p_value': float(p_value), 'cohens_d': float(cohens_d)}


# ---------------------------------------------------------------------------
# Test 5: Very-Potent Retrieval (ChEMBL-verified)
# ---------------------------------------------------------------------------
def test_vp_retrieval(model, splits, compound_index, emb_mmap, device):
    """Hide test-set very_potent compounds (pChEMBL >= 7, experimental) in
    the full negative pool and measure retrieval performance.
    ~3,600 VP compounds vs ~142K negatives → high statistical power.
    """
    log.info('=== Test 5: Very-Potent Retrieval (ChEMBL-verified) ===')
    # VP test-set compounds as targets
    vp_test_ids = [cid for cid in splits['test']['positive_ids']
                   if cid in compound_index
                   and compound_index[cid].get('potency_class_num') == 3]
    test_neg_ids = [cid for cid in splits['test']['negative_ids'] if cid in compound_index]

    if not vp_test_ids:
        return {'vp_retrieval_recall_100': None}

    # Pool = all test negatives + VP test positives (hidden needles)
    pool_ids = list(set(test_neg_ids + vp_test_ids))
    pool_size = len(pool_ids)
    log.info('  Pool: %d total (negs: %d, VP targets: %d)', pool_size, len(test_neg_ids), len(vp_test_ids))

    z_pool = project_ids(model, pool_ids, compound_index, emb_mmap, device)

    # Query: VP centroid from TRAINING set (no leakage)
    vp_train_ids = [cid for cid in splits['train']['positive_ids']
                    if cid in compound_index and compound_index[cid].get('potency_class_num') == 3]
    z_vp_train = project_ids(model, vp_train_ids, compound_index, emb_mmap, device)
    centroid = z_vp_train.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    sims = z_pool @ centroid
    ranked = np.argsort(-sims)

    vp_set = set(vp_test_ids)
    results = {'vp_targets': len(vp_set), 'pool_size': pool_size}

    for k in [100, 500, 1000, 5000, 10000]:
        if k > pool_size:
            continue
        top_k_ids = set(pool_ids[i] for i in ranked[:k])
        found = top_k_ids & vp_set
        recall = len(found) / max(len(vp_set), 1)
        expected = k * len(vp_set) / pool_size
        ef = len(found) / max(expected, 1e-12)
        log.info('  Recall@%d: %d/%d = %.1f%% (EF=%.1fx)', k, len(found), len(vp_set), 100 * recall, ef)
        results[f'vp_recall_{k}'] = float(recall)
        results[f'vp_ef_{k}'] = float(ef)

    # Percentile ranks of VP compounds
    pool_id_list = pool_ids
    vp_ranks = []
    for vid in vp_test_ids:
        idx = pool_id_list.index(vid)
        rank = int(np.where(ranked == idx)[0][0]) + 1
        vp_ranks.append(rank)
    median_pct = 100 * (1 - np.median(vp_ranks) / pool_size)
    mean_pct = 100 * (1 - np.mean(vp_ranks) / pool_size)
    log.info('  VP median percentile: %.1f%%, mean: %.1f%%', median_pct, mean_pct)
    results['vp_median_percentile'] = float(median_pct)
    results['vp_mean_percentile'] = float(mean_pct)
    return results


# ---------------------------------------------------------------------------
# Test 6: Potency-Graded Ranking
# ---------------------------------------------------------------------------
def _dcg(relevances, k):
    """Discounted cumulative gain at rank k."""
    rel = np.array(relevances[:k], dtype=np.float64)
    positions = np.arange(1, len(rel) + 1)
    return np.sum(rel / np.log2(positions + 1))


def test_potency_graded_ranking(model, splits, compound_index, emb_mmap, device):
    """Rank all test compounds (3 potency classes + negatives) by VP-centroid
    similarity. Measures whether ranking follows the potency gradient.

    Metrics:
    - Per-class mean percentile (VP > Potent > Active expected)
    - NDCG@K with graded relevance (VP=3, Potent=2, Active=1, Neg=0)
    - Per-target-family Spearman ρ
    - Dose-response monotonicity (binned similarity vs potency)
    """
    log.info('=== Test 6: Potency-Graded Ranking ===')
    # Gather ALL test compounds: positives + negatives
    test_pos_ids = [cid for cid in splits['test']['positive_ids'] if cid in compound_index]
    test_neg_ids = [cid for cid in splits['test']['negative_ids'] if cid in compound_index]
    all_test_ids = test_pos_ids + test_neg_ids
    n_total = len(all_test_ids)
    log.info('  Pool: %d pos + %d neg = %d total', len(test_pos_ids), len(test_neg_ids), n_total)

    # Relevance scores: VP=3, Potent=2, Active=1, Neg=0
    relevances = []
    potency_labels = []  # numeric for Spearman
    for cid in all_test_ids:
        info = compound_index[cid]
        pcn = info.get('potency_class_num', 0)
        if cid in set(test_neg_ids):
            relevances.append(0)
            potency_labels.append(0)
        else:
            relevances.append(pcn)  # 1, 2, or 3
            potency_labels.append(pcn)
    relevances = np.array(relevances)
    potency_labels = np.array(potency_labels)

    # Project all
    z_all = project_ids(model, all_test_ids, compound_index, emb_mmap, device)

    # VP centroid from training set
    vp_train_ids = [cid for cid in splits['train']['positive_ids']
                    if cid in compound_index and compound_index[cid].get('potency_class_num') == 3]
    z_vp = project_ids(model, vp_train_ids, compound_index, emb_mmap, device)
    centroid = z_vp.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    sims = z_all @ centroid
    ranked = np.argsort(-sims)

    results = {}

    # --- Per-class mean percentile ---
    for cls, label in [(3, 'VP'), (2, 'Potent'), (1, 'Active'), (0, 'Neg')]:
        mask = relevances == cls
        if mask.sum() == 0:
            continue
        cls_ranks = []
        for idx in np.where(mask)[0]:
            rank = int(np.where(ranked == idx)[0][0]) + 1
            cls_ranks.append(rank)
        pct = 100 * (1 - np.mean(cls_ranks) / n_total)
        results[f'mean_pctl_{label}'] = float(pct)
        log.info('  %s mean percentile: %.1f%% (n=%d)', label, pct, len(cls_ranks))

    # --- NDCG@K ---
    ranked_relevances = relevances[ranked]
    ideal_relevances = np.sort(relevances)[::-1]
    for k in [100, 500, 1000, 5000]:
        if k > n_total:
            continue
        dcg = _dcg(ranked_relevances, k)
        idcg = _dcg(ideal_relevances, k)
        ndcg = dcg / max(idcg, 1e-12)
        results[f'ndcg_{k}'] = float(ndcg)
        log.info('  NDCG@%d: %.4f', k, ndcg)

    # --- Per-target-family Spearman ρ ---
    target_groups = defaultdict(list)
    for i, cid in enumerate(test_pos_ids):  # only positives have target info
        family = compound_index[cid].get('target_family', '')
        if family:
            target_groups[family].append(i)

    family_rhos = []
    for family, indices in sorted(target_groups.items(), key=lambda x: -len(x[1])):
        if len(indices) < 10:
            continue
        idx = np.array(indices)
        fam_sims = sims[idx]
        fam_potency = potency_labels[idx]
        r, p = stats.spearmanr(fam_sims, fam_potency)
        if not np.isnan(r):
            family_rhos.append((family, float(r), float(p), len(indices)))
            log.info('  %s: ρ=%.4f (p=%.2e, n=%d)', family, r, p, len(indices))

    if family_rhos:
        mean_fam_rho = np.mean([r for _, r, _, _ in family_rhos])
        results['per_family_mean_rho'] = float(mean_fam_rho)
        results['per_family_n'] = len(family_rhos)
        results['per_family_detail'] = [
            {'family': f, 'rho': r, 'p': p, 'n': n} for f, r, p, n in family_rhos
        ]
        log.info('  Mean per-family ρ: %.4f (across %d families)', mean_fam_rho, len(family_rhos))

    # --- Dose-response monotonicity ---
    # Bin positives by potency class, compute mean similarity per bin
    for cls, label in [(1, 'Active'), (2, 'Potent'), (3, 'VP')]:
        mask = relevances == cls
        if mask.sum() > 0:
            results[f'mean_sim_{label}'] = float(sims[mask].mean())
    # Check monotonicity: Active < Potent < VP
    sim_active = results.get('mean_sim_Active', 0)
    sim_potent = results.get('mean_sim_Potent', 0)
    sim_vp = results.get('mean_sim_VP', 0)
    monotonic = sim_active < sim_potent < sim_vp
    results['dose_response_monotonic'] = monotonic
    log.info('  Dose-response: Active=%.4f < Potent=%.4f < VP=%.4f → %s',
             sim_active, sim_potent, sim_vp, 'PASS' if monotonic else 'FAIL')

    return results


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
def run_full_evaluation(checkpoint_path, device='mps'):
    """Run all 6 evaluation tests and save results."""
    log.info('Loading model from %s', checkpoint_path)
    model, ckpt = load_model(checkpoint_path, device)
    log.info('Model args: %s', ckpt.get('args', {}))

    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)

    n_compounds = len(np.load(GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True))
    emb_mmap = np.memmap(
        GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
        dtype='float16', mode='r', shape=(n_compounds, 768)
    )

    results = {}
    results.update(test_potency_ordering(model, splits, compound_index, emb_mmap, device))
    results.update(test_activity_cliff(model, splits, compound_index, emb_mmap, device))
    results.update(test_target_switch(model, splits, compound_index, emb_mmap, device))
    results.update(test_oscc_specificity(model, splits, compound_index, emb_mmap, device))
    results.update(test_vp_retrieval(model, splits, compound_index, emb_mmap, device))
    results.update(test_potency_graded_ranking(model, splits, compound_index, emb_mmap, device))

    log.info('========== EVALUATION SUMMARY ==========')
    for k, v in results.items():
        log.info('  %s: %s', k, v)

    out_path = LOG_DIR / 'evaluation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info('Results -> %s', out_path)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run evaluation battery')
    parser.add_argument('--checkpoint', default=str(CHECKPOINT_DIR / 'best.pt'))
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    run_full_evaluation(args.checkpoint, args.device)
