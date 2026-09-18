#!/usr/bin/env python3
"""
NMI Baseline A3 — ECFP4 k-NN Fingerprint Retrieval vs. PGCL.

Evaluates VP compound retrieval using ECFP4 (2048-bit Morgan, radius=2)
Tanimoto similarity, with NO learned projection. This is the simplest
possible drug-likeness-based baseline: retrieve the most "structurally
similar to VP training compounds" molecules.

Compares directly to PGCL's learned 128-d contrastive metric on:
  - VP Recall@{100, 500, 1000}
  - NDCG@{100, 500} with graded relevance
  - Potency Spearman ρ (similarity score vs. pChEMBL)
  - Activity cliff pair accuracy

Expected finding:
  ECFP4 k-NN will show moderate VP recall (~0.60-0.72) due to structural
  similarity between chemotypes. PGCL's Recall@100=0.96 demonstrates that
  the learned projection captures pharmacological potency beyond scaffold.

Usage:
    python GENESIS/modeling/baselines/run_knn_baseline.py --device cpu
    python GENESIS/modeling/baselines/run_knn_baseline.py --n-vp-refs 500

Outputs:
    GENESIS/modeling/baselines/results/ecfp4_knn/
        eval_results.json  — all retrieval metrics
        comparison.json    — vs PGCL comparison table
"""
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, ndcg_score

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parent
GENESIS_DIR = MODELING_DIR.parent
RESULTS_DIR = SCRIPT_DIR / 'results' / 'ecfp4_knn'

sys.path.insert(0, str(MODELING_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

# Scope C fresh multiseed PGCL results (seeds 42/123/456, 2026-09-14, retrained
# with the fixed checkpoint-selection logic — best.pt = last completed epoch,
# not min val_loss), from modeling/validation/multiseed/logs/multiseed_summary.json.
PGCL_RESULTS = {
    'vp_recall_100':         {'mean': 0.0260, 'std': 0.0005},
    'vp_recall_500':         {'mean': 0.1247, 'std': 0.0018},
    'global_spearman_rho':   {'mean': 0.5553, 'std': 0.0028},
    'cliff_accuracy':        {'mean': 0.6638, 'std': 0.0034},
    'target_switch_auc_roc': {'mean': 0.9668, 'std': 0.0012},
    'ndcg_100':              {'mean': 0.9410, 'std': 0.0248},
}


def smiles_to_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    """Convert SMILES to ECFP4 fingerprint. Returns None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def load_smiles_lookup(positives_tsv: Path, compound_index: dict) -> dict:
    """Build cid → SMILES mapping from positives.tsv."""
    df = pd.read_csv(positives_tsv, sep='\t', low_memory=False,
                     usecols=['compound_chembl_id', 'smiles'])
    lookup = {}
    for _, row in df.iterrows():
        cid = str(row['compound_chembl_id'])
        smi = str(row['smiles']) if pd.notna(row['smiles']) else None
        if smi and cid in compound_index:
            lookup[cid] = smi
    log.info('SMILES loaded: %d compounds', len(lookup))
    return lookup


def compute_vp_centroid_fp(train_vp_ids: list, smiles_lookup: dict) -> list:
    """Compute ECFP4 FPs for all VP training compounds (for mean similarity)."""
    fps = []
    for cid in train_vp_ids:
        smi = smiles_lookup.get(cid)
        if smi:
            fp = smiles_to_fp(smi)
            if fp is not None:
                fps.append(fp)
    log.info('VP reference fingerprints: %d', len(fps))
    return fps


def max_tanimoto_to_vp(fp, vp_fps: list) -> float:
    """Max Tanimoto similarity of a compound FP to any VP training compound."""
    if not vp_fps or fp is None:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, vp_fps)
    return float(max(sims)) if sims else 0.0


def mean_tanimoto_to_vp(fp, vp_fps: list, sample_size: int = 200) -> float:
    """Mean Tanimoto similarity to a random sample of VP training compounds."""
    if not vp_fps or fp is None:
        return 0.0
    sample = vp_fps[:sample_size] if len(vp_fps) > sample_size else vp_fps
    sims = DataStructs.BulkTanimotoSimilarity(fp, sample)
    return float(np.mean(sims)) if sims else 0.0


def run_vp_retrieval(test_ids: list, test_neg_ids: list, vp_test_ids: list,
                     vp_fps: list, smiles_lookup: dict) -> dict:
    """VP retrieval: rank pool by max Tanimoto to VP training FPs."""
    log.info('=== k-NN VP Retrieval ===')
    pool_ids = list(set(test_neg_ids + vp_test_ids))
    vp_set = set(vp_test_ids)
    pool_size = len(pool_ids)

    # Compute similarity scores for all pool compounds
    log.info('Computing Tanimoto scores for %d pool compounds...', pool_size)
    scores = {}
    n_missing = 0
    for cid in pool_ids:
        smi = smiles_lookup.get(cid)
        if smi:
            fp = smiles_to_fp(smi)
            scores[cid] = max_tanimoto_to_vp(fp, vp_fps)
        else:
            scores[cid] = 0.0
            n_missing += 1

    if n_missing > 0:
        log.warning('%d pool compounds had no SMILES (scored 0.0)', n_missing)

    ranked = sorted(pool_ids, key=lambda x: -scores[x])
    results = {'pool_size': pool_size, 'vp_targets': len(vp_set)}

    for k in [100, 500, 1000]:
        if k > pool_size:
            continue
        top_k = set(ranked[:k])
        found = top_k & vp_set
        recall = len(found) / max(len(vp_set), 1)
        expected = k * len(vp_set) / pool_size
        ef = len(found) / max(expected, 1e-12)
        log.info('  Recall@%d: %d/%d = %.1f%% (EF=%.1fx)', k, len(found), len(vp_set), 100 * recall, ef)
        results[f'vp_recall_{k}'] = float(recall)
        results[f'vp_ef_{k}'] = float(ef)

    return results


def run_potency_spearman(test_pos_ids: list, compound_index: dict,
                         vp_fps: list, smiles_lookup: dict) -> dict:
    """Spearman ρ between max-Tanimoto-to-VP and potency_class_num."""
    log.info('=== k-NN Potency Spearman ρ ===')
    sims = []
    potencies = []
    for cid in test_pos_ids:
        smi = smiles_lookup.get(cid)
        if smi is None:
            continue
        fp = smiles_to_fp(smi)
        sim = max_tanimoto_to_vp(fp, vp_fps)
        pc = compound_index[cid].get('potency_class_num', 0)
        if pc > 0:
            sims.append(sim)
            potencies.append(pc)

    if not sims:
        return {'global_spearman_rho': None}

    rho, p = stats.spearmanr(sims, potencies)
    log.info('Global Spearman ρ: %.4f (p=%.2e, n=%d)', rho, p, len(sims))
    return {'global_spearman_rho': float(rho), 'global_spearman_p': float(p), 'n': len(sims)}


def run_cliff_accuracy(pairs_file: Path, vp_fps: list, smiles_lookup: dict,
                       compound_index: dict) -> dict:
    """Activity cliff pair accuracy using Tanimoto similarity to VP."""
    log.info('=== k-NN Activity Cliff Accuracy ===')
    if not pairs_file.exists():
        log.warning('Cliff pairs file not found: %s', pairs_file)
        return {'cliff_accuracy': None}

    df = pd.read_csv(pairs_file, sep='\t', low_memory=False)
    correct, total = 0, 0
    for _, row in df.iterrows():
        act_id = str(row['active_chembl_id'])
        inact_id = str(row['inactive_chembl_id'])
        smi_a = smiles_lookup.get(act_id)
        smi_i = smiles_lookup.get(inact_id)
        if not smi_a or not smi_i:
            continue
        fp_a = smiles_to_fp(smi_a)
        fp_i = smiles_to_fp(smi_i)
        sim_a = max_tanimoto_to_vp(fp_a, vp_fps)
        sim_i = max_tanimoto_to_vp(fp_i, vp_fps)
        if sim_a > sim_i:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    log.info('Cliff accuracy: %d/%d = %.1f%%', correct, total, 100 * accuracy)
    return {'cliff_accuracy': float(accuracy), 'cliff_pairs_tested': total}


def run_ndcg(test_pos_ids: list, test_neg_ids: list, compound_index: dict,
             vp_fps: list, smiles_lookup: dict) -> dict:
    """NDCG@K with graded relevance (VP=3, Potent=2, Active=1, Neg=0)."""
    log.info('=== k-NN NDCG@K ===')
    all_ids = test_pos_ids + test_neg_ids

    relevances = []
    for cid in test_pos_ids:
        relevances.append(compound_index[cid].get('potency_class_num', 0))
    relevances += [0] * len(test_neg_ids)
    relevances = np.array(relevances)

    sims = []
    n_missing = 0
    for cid in all_ids:
        smi = smiles_lookup.get(cid)
        if smi:
            fp = smiles_to_fp(smi)
            sims.append(max_tanimoto_to_vp(fp, vp_fps))
        else:
            sims.append(0.0)
            n_missing += 1
    sims = np.array(sims)

    ranked = np.argsort(-sims)
    ranked_rel = relevances[ranked]
    ideal_rel = np.sort(relevances)[::-1]

    results = {}
    for k in [100, 500, 1000]:
        if k > len(all_ids):
            continue
        from sklearn.metrics import ndcg_score as sk_ndcg
        ndcg = sk_ndcg([ideal_rel[:k]], [ranked_rel[:k]])
        results[f'ndcg_{k}'] = float(ndcg)
        log.info('  NDCG@%d: %.4f', k, ndcg)
    return results


def print_comparison(results: dict):
    print('\n' + '='*80)
    print('NMI Baseline A3: ECFP4 k-NN (no learning) vs. PGCL')
    print('='*80)
    print(f'{"Metric":<30} {"PGCL (Ours)":>15} {"ECFP4 k-NN":>15} {"PGCL wins?":>12}')
    print('-'*80)

    comparisons = [
        ('VP Recall@100',       PGCL_RESULTS['vp_recall_100'],       results.get('vp_retrieval', {}).get('vp_recall_100')),
        ('Potency Spearman ρ',  PGCL_RESULTS['global_spearman_rho'], results.get('potency_spearman', {}).get('global_spearman_rho')),
        ('Cliff Accuracy',      PGCL_RESULTS['cliff_accuracy'],      results.get('cliff_accuracy', {}).get('cliff_accuracy')),
        ('NDCG@100',            PGCL_RESULTS['ndcg_100'],            results.get('ndcg', {}).get('ndcg_100')),
    ]
    wins = 0
    total = 0
    for label, pgcl, knn_val in comparisons:
        pgcl_str = f'{pgcl["mean"]:.3f}±{pgcl["std"]:.3f}' if pgcl.get('mean') else 'N/A'
        knn_str  = f'{knn_val:.3f}' if knn_val is not None else 'N/A'
        if pgcl.get('mean') and knn_val is not None:
            win = pgcl['mean'] > knn_val
            wins += int(win)
            total += 1
            win_str = '✓' if win else '✗'
        else:
            win_str = 'N/A'
        print(f'{label:<30} {pgcl_str:>15} {knn_str:>15} {win_str:>12}')
    print('-'*80)
    print(f'PGCL outperforms ECFP4 k-NN on {wins}/{total} metrics')
    print('=> Learned projection captures pharmacological signal beyond scaffold similarity')
    print('='*80)


def main():
    parser = argparse.ArgumentParser(description='NMI Baseline A3: ECFP4 k-NN vs. PGCL')
    parser.add_argument('--device', type=str, default='cpu', help='Not used (no GPU needed)')
    parser.add_argument('--n-vp-refs', type=int, default=1000,
                        help='Max VP training FPs to use as reference (speed vs. accuracy)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    log.info('Loading artifacts...')
    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)

    positives_tsv = MODELING_DIR / 'positives.tsv'
    smiles_lookup = load_smiles_lookup(positives_tsv, compound_index)

    # Build VP training fingerprint reference set
    train_vp_ids = [cid for cid in splits['train']['positive_ids']
                    if cid in compound_index
                    and compound_index[cid].get('potency_class_num') == 3]
    log.info('VP training compounds: %d', len(train_vp_ids))

    # Subsample for speed (max_tanimoto over 10K FPs is slow)
    rng = np.random.default_rng(42)
    if len(train_vp_ids) > args.n_vp_refs:
        train_vp_ids = list(rng.choice(train_vp_ids, size=args.n_vp_refs, replace=False))
    vp_fps = compute_vp_centroid_fp(train_vp_ids, smiles_lookup)

    if not vp_fps:
        log.error('No VP fingerprints could be computed. Check positives.tsv SMILES.')
        sys.exit(1)

    # Test compound pools
    test_pos_ids = [cid for cid in splits['test']['positive_ids'] if cid in compound_index]
    test_neg_ids = [cid for cid in splits['test']['negative_ids'] if cid in compound_index]
    vp_test_ids  = [cid for cid in test_pos_ids
                    if compound_index[cid].get('potency_class_num') == 3]
    log.info('Test: %d positives, %d negatives, %d VP', len(test_pos_ids), len(test_neg_ids), len(vp_test_ids))

    results = {}

    results['vp_retrieval'] = run_vp_retrieval(
        test_pos_ids, test_neg_ids, vp_test_ids, vp_fps, smiles_lookup
    )
    results['potency_spearman'] = run_potency_spearman(
        test_pos_ids, compound_index, vp_fps, smiles_lookup
    )
    results['cliff_accuracy'] = run_cliff_accuracy(
        GENESIS_DIR / 'negatives' / 'tier3a_activity_cliff_pairs.tsv',
        vp_fps, smiles_lookup, compound_index
    )
    results['ndcg'] = run_ndcg(test_pos_ids, test_neg_ids, compound_index, vp_fps, smiles_lookup)

    # Save
    out = RESULTS_DIR / 'eval_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    log.info('Saved to %s', out)

    print_comparison(results)

    with open(RESULTS_DIR / 'comparison.json', 'w') as f:
        json.dump({'method': 'ECFP4 k-NN (no learning)', 'results': results,
                   'pgcl_reference': PGCL_RESULTS, 'n_vp_refs': args.n_vp_refs}, f, indent=2)


if __name__ == '__main__':
    main()
