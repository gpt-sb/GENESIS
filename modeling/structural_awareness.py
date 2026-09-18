#!/usr/bin/env python3
"""
Structural Awareness Analysis — Phase 2 Post-Hoc Diagnostic

Question: After contrastive training optimizes for potency-graded biological
proximity, does the 128-d projection still retain structural information?

Tests:
  A. Intra-class structural correlation
     For pairs within the same potency class, Spearman(cos_projected, tanimoto).
  B. Projection vs. MolDeBERTa structural retention
     Compare structural correlation in 768-d (pre-projection) vs. 128-d (post).
  C. Same-target same-class structural ordering
     Within compounds sharing both target AND potency class, does projected
     similarity correlate with Tanimoto?
  D. Scaffold clustering preservation
     Do same-scaffold compounds remain closer than cross-scaffold compounds
     within the same potency class?
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import ContrastiveModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

MODELING_DIR = SCRIPT_DIR
GENESIS_DIR = SCRIPT_DIR.parent
CHECKPOINT = MODELING_DIR / 'checkpoints' / 'best.pt'
POS_FILE = MODELING_DIR / 'positives.tsv'
SCAFFOLD_FILE = MODELING_DIR / 'scaffold_groups.pkl'
RESULTS_DIR = GENESIS_DIR / 'results'

# Number of random pairs to sample per class (for tractable Tanimoto computation)
N_PAIRS = 50_000
# Random seed
SEED = 42


def load_model_and_data(device='mps'):
    """Load trained model, embeddings, splits, and SMILES data."""
    # Model
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    proj_dim = ckpt.get('args', {}).get('proj_dim', 128)
    model = ContrastiveModel(input_dim=768, proj_dim=proj_dim, dropout=0.1)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Compound index
    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)

    # Splits
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)

    # Embeddings
    emb_ids = np.load(GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True)
    emb_mmap = np.memmap(
        GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
        dtype='float16', mode='r', shape=(len(emb_ids), 768)
    )

    # SMILES from positives.tsv
    pos_df = pd.read_csv(POS_FILE, sep='\t', low_memory=False)
    smiles_map = dict(zip(pos_df['compound_chembl_id'], pos_df['smiles']))

    # Scaffold groups
    scaffolds = None
    if SCAFFOLD_FILE.exists():
        with open(SCAFFOLD_FILE, 'rb') as f:
            raw = pickle.load(f)
        # Handle nested structure: {'scaffold_to_compounds': {scaffold: [cids]}, ...}
        if 'scaffold_to_compounds' in raw:
            scaffolds = raw['scaffold_to_compounds']
        else:
            scaffolds = raw

    return model, compound_index, splits, emb_mmap, smiles_map, scaffolds, device


def compute_fingerprints(smiles_list):
    """Compute Morgan fingerprints (radius=2, 2048 bits). Returns list of (idx, fp) tuples."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    for i, smi in enumerate(smiles_list):
        if pd.isna(smi) or not smi:
            fps.append(None)
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
        else:
            fps.append(None)
    return fps


def project_batch(model, emb_mmap, compound_index, cids, device, batch_size=4096):
    """Project compound IDs through model. Returns (projected_128d, raw_768d) arrays."""
    projections = []
    raw_embeddings = []
    for start in range(0, len(cids), batch_size):
        batch_ids = cids[start:start + batch_size]
        embs = []
        for cid in batch_ids:
            info = compound_index[cid]
            embs.append(emb_mmap[info['emb_idx']].astype(np.float32))
        embs_np = np.stack(embs)
        raw_embeddings.append(embs_np)
        embs_t = torch.tensor(embs_np, device=device)
        with torch.no_grad():
            z, _ = model(embs_t)
        projections.append(z.cpu().numpy())
    return np.concatenate(projections, axis=0), np.concatenate(raw_embeddings, axis=0)


def sample_pairs(n_compounds, n_pairs, rng):
    """Sample random pairs of indices."""
    if n_compounds < 2:
        return np.array([]), np.array([])
    max_pairs = n_compounds * (n_compounds - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    pairs_i = []
    pairs_j = []
    seen = set()
    while len(pairs_i) < n_pairs:
        batch = rng.integers(0, n_compounds, size=(n_pairs * 2, 2))
        for a, b in batch:
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            pairs_i.append(key[0])
            pairs_j.append(key[1])
            if len(pairs_i) >= n_pairs:
                break
    return np.array(pairs_i), np.array(pairs_j)


def compute_tanimoto_batch(fps, idx_i, idx_j):
    """Compute Tanimoto similarities for given index pairs."""
    from rdkit import DataStructs
    sims = []
    valid_mask = []
    for a, b in zip(idx_i, idx_j):
        if fps[a] is not None and fps[b] is not None:
            sims.append(DataStructs.TanimotoSimilarity(fps[a], fps[b]))
            valid_mask.append(True)
        else:
            sims.append(np.nan)
            valid_mask.append(False)
    return np.array(sims), np.array(valid_mask)


def cosine_batch(vecs, idx_i, idx_j):
    """Compute cosine similarities for given index pairs from a matrix of vectors."""
    v_i = vecs[idx_i]
    v_j = vecs[idx_j]
    # L2 normalize
    v_i_norm = v_i / (np.linalg.norm(v_i, axis=1, keepdims=True) + 1e-12)
    v_j_norm = v_j / (np.linalg.norm(v_j, axis=1, keepdims=True) + 1e-12)
    return np.sum(v_i_norm * v_j_norm, axis=1)


def test_a_intra_class_correlation(model, compound_index, splits, emb_mmap,
                                    smiles_map, device, rng):
    """Test A: Within same potency class, ρ(cos_projected, tanimoto)."""
    log.info('=== Test A: Intra-Class Structural Correlation ===')
    results = {}

    test_pos_ids = [cid for cid in splits['test']['positive_ids']
                    if cid in compound_index and cid in smiles_map]

    # Group by potency class
    class_groups = defaultdict(list)
    for cid in test_pos_ids:
        pcn = compound_index[cid].get('potency_class_num', 0)
        if pcn > 0:
            class_groups[pcn].append(cid)

    for cls in [1, 2, 3]:
        label = {1: 'Active', 2: 'Potent', 3: 'VP'}[cls]
        cids = class_groups[cls]
        if len(cids) < 10:
            log.info('  %s: too few compounds (%d)', label, len(cids))
            continue

        log.info('  %s: %d compounds, sampling %d pairs...', label, len(cids), N_PAIRS)

        # Project
        z_proj, z_raw = project_batch(model, emb_mmap, compound_index, cids, device)

        # Fingerprints
        smiles = [smiles_map.get(cid, '') for cid in cids]
        fps = compute_fingerprints(smiles)

        # Sample pairs
        idx_i, idx_j = sample_pairs(len(cids), N_PAIRS, rng)
        if len(idx_i) == 0:
            continue

        # Cosine in projected space
        cos_proj = cosine_batch(z_proj, idx_i, idx_j)

        # Cosine in MolDeBERTa space
        cos_raw = cosine_batch(z_raw, idx_i, idx_j)

        # Tanimoto
        tani, valid = compute_tanimoto_batch(fps, idx_i, idx_j)

        # Filter to valid pairs (both have fingerprints)
        mask = valid & ~np.isnan(tani)
        cos_proj_v = cos_proj[mask]
        cos_raw_v = cos_raw[mask]
        tani_v = tani[mask]

        if len(tani_v) < 10:
            log.info('  %s: too few valid pairs (%d)', label, len(tani_v))
            continue

        # Spearman: projected vs Tanimoto
        rho_proj, p_proj = stats.spearmanr(cos_proj_v, tani_v)
        # Spearman: MolDeBERTa vs Tanimoto
        rho_raw, p_raw = stats.spearmanr(cos_raw_v, tani_v)
        # Retention ratio
        retention = rho_proj / rho_raw if rho_raw > 0 else float('nan')

        log.info('  %s: ρ(projected, tanimoto) = %.4f (p=%.2e)', label, rho_proj, p_proj)
        log.info('  %s: ρ(MolDeBERTa, tanimoto) = %.4f (p=%.2e)', label, rho_raw, p_raw)
        log.info('  %s: Retention ratio = %.1f%%', label, 100 * retention)
        log.info('  %s: Valid pairs = %d', label, len(tani_v))

        results[label] = {
            'n_compounds': len(cids),
            'n_valid_pairs': int(len(tani_v)),
            'rho_projected_tanimoto': float(rho_proj),
            'p_projected': float(p_proj),
            'rho_moldeberta_tanimoto': float(rho_raw),
            'p_moldeberta': float(p_raw),
            'retention_pct': float(100 * retention),
            'tanimoto_mean': float(tani_v.mean()),
            'tanimoto_std': float(tani_v.std()),
            'cos_projected_mean': float(cos_proj_v.mean()),
            'cos_projected_std': float(cos_proj_v.std()),
            'cos_moldeberta_mean': float(cos_raw_v.mean()),
            'cos_moldeberta_std': float(cos_raw_v.std()),
        }

    return results


def test_b_cross_space_comparison(model, compound_index, splits, emb_mmap,
                                   smiles_map, device, rng):
    """Test B: Pairwise distances in projected vs MolDeBERTa vs Tanimoto (all test positives)."""
    log.info('=== Test B: Cross-Space Structural Retention ===')

    test_pos_ids = [cid for cid in splits['test']['positive_ids']
                    if cid in compound_index and cid in smiles_map]
    log.info('  Test positives with SMILES: %d', len(test_pos_ids))

    z_proj, z_raw = project_batch(model, emb_mmap, compound_index, test_pos_ids, device)
    smiles = [smiles_map.get(cid, '') for cid in test_pos_ids]
    fps = compute_fingerprints(smiles)

    idx_i, idx_j = sample_pairs(len(test_pos_ids), N_PAIRS, rng)
    cos_proj = cosine_batch(z_proj, idx_i, idx_j)
    cos_raw = cosine_batch(z_raw, idx_i, idx_j)
    tani, valid = compute_tanimoto_batch(fps, idx_i, idx_j)

    mask = valid & ~np.isnan(tani)
    cos_proj_v = cos_proj[mask]
    cos_raw_v = cos_raw[mask]
    tani_v = tani[mask]

    # Cross correlations
    rho_proj_tani, p1 = stats.spearmanr(cos_proj_v, tani_v)
    rho_raw_tani, p2 = stats.spearmanr(cos_raw_v, tani_v)
    rho_proj_raw, p3 = stats.spearmanr(cos_proj_v, cos_raw_v)

    log.info('  ρ(projected, tanimoto) = %.4f', rho_proj_tani)
    log.info('  ρ(MolDeBERTa, tanimoto) = %.4f', rho_raw_tani)
    log.info('  ρ(projected, MolDeBERTa) = %.4f', rho_proj_raw)
    log.info('  Structural retention: %.1f%%', 100 * rho_proj_tani / rho_raw_tani)

    # Binned analysis: group pairs by Tanimoto range, check if projected cos follows
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
    bin_results = []
    for lo, hi in bins:
        bmask = (tani_v >= lo) & (tani_v < hi)
        if bmask.sum() < 10:
            continue
        mean_proj = cos_proj_v[bmask].mean()
        mean_raw = cos_raw_v[bmask].mean()
        mean_tani = tani_v[bmask].mean()
        bin_results.append({
            'tanimoto_range': f'{lo:.1f}-{hi:.1f}',
            'n_pairs': int(bmask.sum()),
            'mean_cos_projected': float(mean_proj),
            'mean_cos_moldeberta': float(mean_raw),
            'mean_tanimoto': float(mean_tani),
        })
        log.info('  Tani [%.1f–%.1f]: n=%d, cos_proj=%.4f, cos_raw=%.4f',
                 lo, hi, bmask.sum(), mean_proj, mean_raw)

    return {
        'n_valid_pairs': int(mask.sum()),
        'rho_projected_tanimoto': float(rho_proj_tani),
        'rho_moldeberta_tanimoto': float(rho_raw_tani),
        'rho_projected_moldeberta': float(rho_proj_raw),
        'retention_pct': float(100 * rho_proj_tani / rho_raw_tani),
        'binned_analysis': bin_results,
    }


def test_c_same_target_same_class(model, compound_index, splits, emb_mmap,
                                   smiles_map, device, rng):
    """Test C: Within same target + same potency class, ρ(cos_proj, tanimoto)."""
    log.info('=== Test C: Same-Target Same-Class Structural Ordering ===')

    test_pos_ids = [cid for cid in splits['test']['positive_ids']
                    if cid in compound_index and cid in smiles_map]

    # Group by (target, potency_class)
    groups = defaultdict(list)
    for cid in test_pos_ids:
        info = compound_index[cid]
        target = info.get('target_name', '') or ''
        pcn = info.get('potency_class_num', 0)
        if target and pcn > 0:
            groups[(target, pcn)].append(cid)

    # Filter to groups with enough compounds
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 10}
    log.info('  Groups with >= 10 compounds: %d', len(valid_groups))

    group_results = []
    all_rhos = []

    for (target, cls), cids in sorted(valid_groups.items(), key=lambda x: -len(x[1])):
        z_proj, z_raw = project_batch(model, emb_mmap, compound_index, cids, device)
        smiles = [smiles_map.get(cid, '') for cid in cids]
        fps = compute_fingerprints(smiles)

        n_sample = min(N_PAIRS // len(valid_groups), len(cids) * (len(cids) - 1) // 2)
        n_sample = max(n_sample, 100)
        idx_i, idx_j = sample_pairs(len(cids), n_sample, rng)
        if len(idx_i) < 5:
            continue

        cos_proj = cosine_batch(z_proj, idx_i, idx_j)
        tani, valid = compute_tanimoto_batch(fps, idx_i, idx_j)

        mask = valid & ~np.isnan(tani)
        if mask.sum() < 5:
            continue

        rho, p = stats.spearmanr(cos_proj[mask], tani[mask])
        if np.isnan(rho):
            continue

        label = {1: 'Active', 2: 'Potent', 3: 'VP'}[cls]
        all_rhos.append(rho)
        group_results.append({
            'target': target,
            'potency_class': label,
            'n_compounds': len(cids),
            'n_valid_pairs': int(mask.sum()),
            'rho': float(rho),
            'p': float(p),
        })

    if all_rhos:
        mean_rho = np.mean(all_rhos)
        median_rho = np.median(all_rhos)
        log.info('  Mean intra-group ρ: %.4f (median: %.4f, n_groups: %d)',
                 mean_rho, median_rho, len(all_rhos))
        # Show top 10
        for g in sorted(group_results, key=lambda x: -x['rho'])[:10]:
            log.info('    %s [%s] n=%d: ρ=%.4f', g['target'], g['potency_class'],
                     g['n_compounds'], g['rho'])

    return {
        'n_groups_tested': len(group_results),
        'mean_rho': float(np.mean(all_rhos)) if all_rhos else None,
        'median_rho': float(np.median(all_rhos)) if all_rhos else None,
        'std_rho': float(np.std(all_rhos)) if all_rhos else None,
        'group_detail': group_results,
    }


def test_d_scaffold_clustering(model, compound_index, splits, emb_mmap, scaffolds,
                                device, rng):
    """Test D: Within same potency class, are same-scaffold pairs closer
    in projected space than cross-scaffold pairs?"""
    log.info('=== Test D: Scaffold Clustering Preservation ===')

    if scaffolds is None:
        log.info('  No scaffold data available')
        return {'error': 'no scaffold data'}

    test_pos_ids = [cid for cid in splits['test']['positive_ids']
                    if cid in compound_index]

    # Build reverse map: compound → scaffold group
    cid_to_scaffold = {}
    for scaffold, members in scaffolds.items():
        for cid in members:
            cid_to_scaffold[cid] = scaffold

    results = {}
    for cls in [1, 2, 3]:
        label = {1: 'Active', 2: 'Potent', 3: 'VP'}[cls]
        cids = [cid for cid in test_pos_ids
                if compound_index[cid].get('potency_class_num') == cls
                and cid in cid_to_scaffold]

        if len(cids) < 20:
            continue

        z_proj, _ = project_batch(model, emb_mmap, compound_index, cids, device)

        # Compute pairwise cosine for same-scaffold and cross-scaffold pairs
        same_scaffold_sims = []
        cross_scaffold_sims = []

        # Scaffold groups within this class
        scaf_groups = defaultdict(list)
        for i, cid in enumerate(cids):
            scaf_groups[cid_to_scaffold[cid]].append(i)

        # Same-scaffold pairs
        for scaf, indices in scaf_groups.items():
            if len(indices) < 2:
                continue
            for a_idx in range(len(indices)):
                for b_idx in range(a_idx + 1, min(a_idx + 20, len(indices))):
                    i, j = indices[a_idx], indices[b_idx]
                    sim = float(np.dot(z_proj[i], z_proj[j]))
                    same_scaffold_sims.append(sim)

        # Cross-scaffold pairs (sample)
        n_cross = min(len(same_scaffold_sims) * 3, 50000)
        if n_cross > 0 and len(cids) >= 2:
            cross_i, cross_j = sample_pairs(len(cids), n_cross, rng)
            for a, b in zip(cross_i, cross_j):
                if cid_to_scaffold[cids[a]] != cid_to_scaffold[cids[b]]:
                    sim = float(np.dot(z_proj[a], z_proj[b]))
                    cross_scaffold_sims.append(sim)

        if same_scaffold_sims and cross_scaffold_sims:
            same_mean = np.mean(same_scaffold_sims)
            cross_mean = np.mean(cross_scaffold_sims)
            gap = same_mean - cross_mean
            # Effect size
            pooled_std = np.sqrt(
                (np.std(same_scaffold_sims)**2 + np.std(cross_scaffold_sims)**2) / 2
            )
            d = gap / (pooled_std + 1e-12)
            u_stat, p_val = stats.mannwhitneyu(
                same_scaffold_sims, cross_scaffold_sims, alternative='greater'
            )

            log.info('  %s: same-scaffold cos=%.4f (n=%d), cross-scaffold cos=%.4f (n=%d)',
                     label, same_mean, len(same_scaffold_sims),
                     cross_mean, len(cross_scaffold_sims))
            log.info('  %s: gap=%.4f, Cohen d=%.4f, p=%.2e', label, gap, d, p_val)

            results[label] = {
                'n_compounds': len(cids),
                'n_same_scaffold_pairs': len(same_scaffold_sims),
                'n_cross_scaffold_pairs': len(cross_scaffold_sims),
                'same_scaffold_cos_mean': float(same_mean),
                'same_scaffold_cos_std': float(np.std(same_scaffold_sims)),
                'cross_scaffold_cos_mean': float(cross_mean),
                'cross_scaffold_cos_std': float(np.std(cross_scaffold_sims)),
                'gap': float(gap),
                'cohens_d': float(d),
                'p_value': float(p_val),
            }

    return results


def main():
    rng = np.random.default_rng(SEED)
    model, compound_index, splits, emb_mmap, smiles_map, scaffolds, device = \
        load_model_and_data()

    all_results = {}

    all_results['test_a_intra_class'] = test_a_intra_class_correlation(
        model, compound_index, splits, emb_mmap, smiles_map, device, rng
    )
    all_results['test_b_cross_space'] = test_b_cross_space_comparison(
        model, compound_index, splits, emb_mmap, smiles_map, device, rng
    )
    all_results['test_c_same_target_class'] = test_c_same_target_same_class(
        model, compound_index, splits, emb_mmap, smiles_map, device, rng
    )
    all_results['test_d_scaffold_clustering'] = test_d_scaffold_clustering(
        model, compound_index, splits, emb_mmap, scaffolds, device, rng
    )

    # Summary
    log.info('\n' + '=' * 60)
    log.info('STRUCTURAL AWARENESS SUMMARY')
    log.info('=' * 60)

    # Test A summary
    if 'test_a_intra_class' in all_results:
        for cls, data in all_results['test_a_intra_class'].items():
            log.info('  [A] %s: ρ(proj,tani)=%.4f, ρ(raw,tani)=%.4f, retention=%.1f%%',
                     cls, data['rho_projected_tanimoto'], data['rho_moldeberta_tanimoto'],
                     data['retention_pct'])

    # Test B summary
    b = all_results.get('test_b_cross_space', {})
    if b:
        log.info('  [B] Overall: ρ(proj,tani)=%.4f, ρ(raw,tani)=%.4f, retention=%.1f%%',
                 b['rho_projected_tanimoto'], b['rho_moldeberta_tanimoto'],
                 b['retention_pct'])

    # Test C summary
    c = all_results.get('test_c_same_target_class', {})
    if c.get('mean_rho') is not None:
        log.info('  [C] Same-target same-class: mean ρ=%.4f (median=%.4f, n=%d groups)',
                 c['mean_rho'], c['median_rho'], c['n_groups_tested'])

    # Test D summary
    if 'test_d_scaffold_clustering' in all_results:
        for cls, data in all_results['test_d_scaffold_clustering'].items():
            if isinstance(data, dict) and 'gap' in data:
                log.info('  [D] %s: same-scaf=%.4f, cross-scaf=%.4f, gap=%.4f, d=%.4f',
                         cls, data['same_scaffold_cos_mean'],
                         data['cross_scaffold_cos_mean'], data['gap'], data['cohens_d'])

    # Save
    out_path = MODELING_DIR / 'logs' / 'structural_awareness.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info('\nResults → %s', out_path)


if __name__ == '__main__':
    main()
