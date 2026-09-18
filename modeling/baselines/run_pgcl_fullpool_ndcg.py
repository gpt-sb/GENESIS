#!/usr/bin/env python3
"""
Re-run PGCL graded-relevance NDCG on the FULL 178,151-compound baseline pool.

Rationale: Table 1's PGCL NDCG@100 (0.810) came from test_potency_graded_ranking
on the restricted 146,330-compound test pool (test positives + test negatives
that resolve in compound_index). The baselines (Flat SupCon, BCE, ECFP4 k-NN)
were instead evaluated on the full library of 178,151 compounds
(174,491 test negatives + 3,660 VP test positives). This script re-evaluates
PGCL on the identical full pool so Table 1 is like-for-like.

Protocol mirrors evaluate.test_potency_graded_ranking exactly:
  - Graded relevance: VP=3, Potent=2, Active=1, Neg=0
  - Rank by cosine similarity to the training-set VP centroid (no leakage)
  - NDCG@{100,500,1000,5000} via the same _dcg implementation
Pool definition matches the baseline vp_retrieval pool: test negatives + VP test
positives (union). Non-VP test positives are appended with their graded relevance
so the ranking contains the full potency gradient exactly as in
test_potency_graded_ranking.

Outputs:
  modeling/baselines/results/pgcl_fullpool_ndcg.json  (per-seed + mean/std)
"""
import sys
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import torch
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent          # modeling/baselines
MODELING_DIR = SCRIPT_DIR.parent                      # modeling/
GENESIS_DIR = MODELING_DIR.parent                     # GENESIS/
OUT_PATH = SCRIPT_DIR / 'results' / 'pgcl_fullpool_ndcg.json'

sys.path.insert(0, str(MODELING_DIR))
from evaluate import load_model, project_ids, _dcg    # reuse exact implementation

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

SEEDS = [42, 123, 456]
CKPT = {
    42:  MODELING_DIR / 'validation' / 'multiseed' / 'checkpoints' / 'best_seed42.pt',
    123: MODELING_DIR / 'validation' / 'multiseed' / 'checkpoints' / 'best_seed123.pt',
    456: MODELING_DIR / 'validation' / 'multiseed' / 'checkpoints' / 'best_seed456.pt',
}


def full_pool_ndcg(model, splits, compound_index, emb_mmap, device):
    """Graded-relevance NDCG on the full 178K baseline pool."""
    test_pos_ids = [c for c in splits['test']['positive_ids'] if c in compound_index]
    test_neg_ids = [c for c in splits['test']['negative_ids'] if c in compound_index]
    neg_set = set(test_neg_ids)

    # Pool = all test negatives + all test positives (positives carry graded relevance)
    pool_ids = list(test_neg_ids) + [c for c in test_pos_ids if c not in neg_set]
    relevances = np.array(
        [0] * len(test_neg_ids) +
        [compound_index[c].get('potency_class_num', 0) for c in test_pos_ids if c not in neg_set]
    )
    n_total = len(pool_ids)
    log.info('  Full pool: %d (neg %d + pos %d)', n_total, len(test_neg_ids),
             len([c for c in test_pos_ids if c not in neg_set]))

    z_all = project_ids(model, pool_ids, compound_index, emb_mmap, device)

    # Training-set VP centroid (no test leakage)
    vp_train_ids = [c for c in splits['train']['positive_ids']
                    if c in compound_index and compound_index[c].get('potency_class_num') == 3]
    z_vp = project_ids(model, vp_train_ids, compound_index, emb_mmap, device)
    centroid = z_vp.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    sims = z_all @ centroid
    ranked = np.argsort(-sims)
    ranked_rel = relevances[ranked]
    ideal_rel = np.sort(relevances)[::-1]

    res = {'pool_size': int(n_total),
           'n_pos': int((relevances > 0).sum()), 'n_vp': int((relevances == 3).sum())}
    for k in [100, 500, 1000, 5000]:
        if k > n_total:
            continue
        res[f'ndcg_{k}'] = float(_dcg(ranked_rel, k) / max(_dcg(ideal_rel, k), 1e-12))
        log.info('  NDCG@%d: %.4f', k, res[f'ndcg_{k}'])

    rho, p = stats.spearmanr(sims, relevances)
    res['spearman_rho'] = float(rho)
    res['spearman_p'] = float(p)
    log.info('  Full-pool Spearman rho: %.4f (p=%.2e)', rho, p)
    return res


def main():
    device = 'mps' if torch.backends.mps.is_available() else \
             ('cuda' if torch.cuda.is_available() else 'cpu')
    log.info('Device: %s', device)

    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)
    n_compounds = len(np.load(GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True))
    emb_mmap = np.memmap(GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
                         dtype='float16', mode='r', shape=(n_compounds, 768))

    per_seed = {}
    for seed in SEEDS:
        log.info('=== Seed %d (%s) ===', seed, CKPT[seed].name)
        model, _ = load_model(str(CKPT[seed]), device=device)
        per_seed[seed] = full_pool_ndcg(model, splits, compound_index, emb_mmap, device)
        del model
        if device == 'mps':
            torch.mps.empty_cache()

    # Aggregate
    agg = {'per_seed': {str(s): per_seed[s] for s in SEEDS}}
    for key in ['ndcg_100', 'ndcg_500', 'ndcg_1000', 'ndcg_5000', 'spearman_rho']:
        vals = [per_seed[s][key] for s in SEEDS if key in per_seed[s]]
        agg[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=0)),
                    'values': vals, 'n_seeds': len(vals)}
        log.info('Full-pool %s: %.4f ± %.4f  %s', key, agg[key]['mean'], agg[key]['std'], vals)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(agg, f, indent=2)
    log.info('Saved %s', OUT_PATH)

    # Compare against the restricted-pool reference
    print('\n' + '=' * 70)
    print('PGCL NDCG — restricted pool (146,330) vs FULL pool (178,151)')
    print('=' * 70)
    restricted = {'ndcg_100': 0.810, 'ndcg_500': None, 'ndcg_1000': None}
    for key in ['ndcg_100', 'ndcg_500', 'ndcg_1000', 'ndcg_5000']:
        r = restricted.get(key)
        m = agg[key]['mean']
        rs = f'{r:.3f}' if r is not None else '  n/a'
        print(f'  {key}: restricted {rs}  ->  full-pool {m:.4f} ± {agg[key]["std"]:.4f}')
    print('  Baselines (full pool): Flat SupCon NDCG@100 = 0.932, BCE = 0.925, ECFP4 = 1.000')
    print('=' * 70)


if __name__ == '__main__':
    main()
