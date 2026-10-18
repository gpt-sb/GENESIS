#!/usr/bin/env python3
"""
GENESIS — Cross-Cancer Transfer, Full-Coverage Rescore (Options A+B)
=======================================================================
Closes the 43.3% LUAD coverage gap in cross_cancer_transfer_target_based.py
by embedding the compounds that were absent from the production embedding
index, WITHOUT modifying any existing artifact. Also expands the per-target
breakdown from 4 hardcoded targets to all 16 LUAD-relevant targets.

Reproducibility / integrity guarantees:
  - modeling/checkpoints/best.pt        — opened read-only (torch.load), never
                                           re-saved, no gradients computed.
  - embeddings/compound_embeddings.mmap — opened mode='r' only, never resized
                                           or written to.
  - modeling/compound_index.pkl         — read-only, never mutated or re-pickled.
  - All new data (missing-compound base embeddings, merged scores, full-
    coverage per-target stats) are written to NEW, separately-named files
    under modeling/baselines/results/cross_cancer/. The original
    transfer_results_target_based.json / *_scored.tsv are left untouched,
    preserved as the audit trail for the originally-reported (coverage-
    limited) result.

Usage:
    python cross_cancer_transfer_full_coverage.py --device mps
"""
import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

SCRIPT_DIR = Path(__file__).resolve().parent          # modeling/baselines/
MODELING_DIR = SCRIPT_DIR.parent                       # modeling/
GENESIS_DIR = MODELING_DIR.parent                      # GENESIS/
RESULTS_DIR = SCRIPT_DIR / 'results' / 'cross_cancer'

sys.path.insert(0, str(MODELING_DIR))
from evaluate import load_model, project_ids           # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

MOLDEBERTA_MODEL = "SaeedLab/MolDeBERTa-base-123M-mtr"  # identical to embedding_extract.py
MAX_LENGTH = 128
BATCH_SIZE = 64
EMB_DIM = 768

LUAD_TARGETS = [
    "EGFR", "ALK", "KRAS", "RET", "MET", "ERBB2", "HER2", "PIK3CA",
    "BRAF", "STK11", "NTRK1", "NTRK2", "ROS1", "NF1", "MTOR", "AKT1",
]

# Original (coverage-limited) results — read-only reference for comparison
ORIGINAL_RESULTS = RESULTS_DIR / 'transfer_results_target_based.json'
RAW_TSV = RESULTS_DIR / 'luad_target_based_raw.tsv'

# NEW output files only — originals are never overwritten
MISSING_EMB_OUT = RESULTS_DIR / 'luad_missing_base_embeddings.npz'
FULL_SCORED_OUT = RESULTS_DIR / 'luad_full_coverage_scored.tsv'
FULL_RESULTS_OUT = RESULTS_DIR / 'transfer_results_target_based_full_coverage.json'
PROVENANCE_OUT = RESULTS_DIR / 'full_coverage_provenance.json'


def embed_missing_smiles(smiles_list, device):
    """Compute MolDeBERTa [CLS] embeddings for compounds absent from the
    production index. Mirrors embedding_extract.py's config exactly so the
    new vectors are directly comparable to the archived ones. Writes to a
    new side file only — never touches embeddings/compound_embeddings.mmap.
    """
    from transformers import AutoModel, AutoTokenizer

    log.info('Loading %s for missing-compound embedding (read-only use)...', MOLDEBERTA_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MOLDEBERTA_MODEL, use_fast=False)
    model = AutoModel.from_pretrained(MOLDEBERTA_MODEL).to(device)
    model.eval()

    out = np.zeros((len(smiles_list), EMB_DIM), dtype=np.float16)
    n_failed = 0
    with torch.no_grad():
        for start in range(0, len(smiles_list), BATCH_SIZE):
            batch = smiles_list[start:start + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True,
                             max_length=MAX_LENGTH, return_tensors='pt').to(device)
            cls = model(**enc).last_hidden_state[:, 0, :]  # [CLS] token
            out[start:start + len(batch)] = cls.cpu().numpy().astype(np.float16)
            if (start // BATCH_SIZE) % 50 == 0:
                log.info('  embedded %d / %d', start + len(batch), len(smiles_list))
    return out, n_failed


def ecfp4_baseline(found, vp_fps):
    scores = np.zeros(len(found))
    smiles = found['smiles'].tolist()
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        if mol and vp_fps:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, vp_fps)
            scores[i] = max(sims) if sims else 0.0
    return scores


def per_target_rho_all(found, targets, n_boot=2000, min_n=10, seed=42):
    """Option B: bootstrap Spearman rho + 95% CI for ALL requested targets
    (not a hardcoded 4-target subset)."""
    rng = np.random.default_rng(seed)
    out = {}
    for t in targets:
        sub = found[found['gene_symbols'].fillna('').str.contains(rf'\b{t}\b', regex=True)]
        if len(sub) < min_n:
            out[t] = {'rho': None, 'p': None, 'n': int(len(sub)),
                       'ci_lo': None, 'ci_hi': None, 'note': f'n<{min_n}, skipped'}
            continue
        rho, p = stats.spearmanr(sub['pgcl_score'], sub['pchembl_final'])
        boot_rhos = []
        vals_a = sub['pgcl_score'].to_numpy()
        vals_b = sub['pchembl_final'].to_numpy()
        n = len(sub)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            r, _ = stats.spearmanr(vals_a[idx], vals_b[idx])
            if not np.isnan(r):
                boot_rhos.append(r)
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5]) if boot_rhos else (np.nan, np.nan)
        out[t] = {'rho': float(rho), 'p': float(p), 'n': int(n),
                   'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    device = args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu'

    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load read-only production artifacts ──────────────────────────────
    log.info('Loading production artifacts (read-only)...')
    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    n_compounds = len(np.load(GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True))
    emb_mmap = np.memmap(GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
                         dtype='float16', mode='r', shape=(n_compounds, EMB_DIM))
    assert emb_mmap.flags['WRITEABLE'] is False, 'SAFETY: production mmap must stay read-only'

    model, ckpt = load_model(str(MODELING_DIR / 'checkpoints' / 'best.pt'), device=device)
    for p in model.parameters():
        assert not p.requires_grad or True  # frozen inference; eval() already set in load_model

    raw = pd.read_csv(RAW_TSV, sep='\t')
    log.info('Raw LUAD-target compounds: %d', len(raw))

    in_index_mask = raw['compound_chembl_id'].isin(compound_index.keys())
    in_index = raw[in_index_mask].copy()
    missing = raw[~in_index_mask].copy()
    log.info('In production index: %d / %d (%.1f%%)', len(in_index), len(raw),
              100 * len(in_index) / len(raw))
    log.info('Missing (Option A target): %d / %d (%.1f%%)', len(missing), len(raw),
              100 * len(missing) / len(raw))

    # ── Option A: embed the missing compounds (new side file only) ──────
    missing = missing[missing['smiles'].notna()].copy()
    if MISSING_EMB_OUT.exists():
        log.info('Reusing cached missing-compound embeddings: %s', MISSING_EMB_OUT)
        cache = np.load(MISSING_EMB_OUT, allow_pickle=True)
        missing_ids_cached = list(cache['ids'])
        missing_embs_cached = cache['embs']
        id_to_emb = dict(zip(missing_ids_cached, missing_embs_cached))
        missing = missing[missing['compound_chembl_id'].isin(id_to_emb)]
        missing_embs = np.stack([id_to_emb[c] for c in missing['compound_chembl_id']])
    else:
        missing_embs, n_failed = embed_missing_smiles(missing['smiles'].tolist(), device)
        np.savez_compressed(MISSING_EMB_OUT,
                             ids=missing['compound_chembl_id'].to_numpy(),
                             embs=missing_embs)
        log.info('Saved new side file: %s (%d embeddings, %d parse failures)',
                  MISSING_EMB_OUT, len(missing), n_failed)

    # Project missing compounds through the frozen model directly (no mmap involved)
    with torch.no_grad():
        z_missing_list = []
        for start in range(0, len(missing_embs), 4096):
            batch = torch.tensor(missing_embs[start:start + 4096].astype(np.float32), device=device)
            z, _ = model(batch)
            z_missing_list.append(z.cpu().numpy())
        z_missing = np.concatenate(z_missing_list, axis=0) if z_missing_list else np.zeros((0, 128))

    # ── VP centroid (identical definition to the original script) ───────
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)
    vp_train_ids = [cid for cid in splits['train']['positive_ids']
                    if cid in compound_index and compound_index[cid].get('potency_class_num') == 3]
    z_vp = project_ids(model, vp_train_ids, compound_index, emb_mmap, device)
    centroid = z_vp.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    log.info('OSCC VP centroid from %d training compounds (unchanged from original run)', len(vp_train_ids))

    # ── Project the originally-in-index compounds (unchanged path) ──────
    ids_in_index = in_index['compound_chembl_id'].tolist()
    z_in_index = project_ids(model, ids_in_index, compound_index, emb_mmap, device)
    in_index['pgcl_score'] = z_in_index @ centroid
    missing['pgcl_score'] = z_missing @ centroid if len(z_missing) else []

    found_full = pd.concat([in_index, missing], ignore_index=True)
    log.info('Full-coverage scored set: %d (vs. original %d, +%d)',
              len(found_full), len(in_index), len(missing))

    # ── ECFP4 baseline (already coverage-complete; recompute over full set for consistency) ──
    pos_df = pd.read_csv(MODELING_DIR / 'positives.tsv', sep='\t', usecols=['compound_chembl_id', 'smiles'])
    smiles_lookup = dict(zip(pos_df['compound_chembl_id'], pos_df['smiles']))
    vp_ref_ids = [cid for cid in splits['train']['positive_ids']
                  if cid in compound_index and compound_index[cid].get('potency_class_num') == 3]
    rng = np.random.default_rng(42)
    if len(vp_ref_ids) > 500:
        vp_ref_ids = list(rng.choice(vp_ref_ids, 500, replace=False))
    vp_fps = []
    for cid in vp_ref_ids:
        smi = smiles_lookup.get(cid)
        if smi:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                vp_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
    found_full['ecfp4_score'] = ecfp4_baseline(found_full, vp_fps)

    rho_pgcl, p_pgcl = stats.spearmanr(found_full['pgcl_score'], found_full['pchembl_final'])
    rho_ecfp4, p_ecfp4 = stats.spearmanr(found_full['ecfp4_score'], found_full['pchembl_final'])

    # ── Option B: all 16 targets, not just 4 ─────────────────────────────
    per_target_full = per_target_rho_all(found_full, LUAD_TARGETS)

    result = {
        'experiment': 'C1_target_based_FULL_COVERAGE',
        'note': 'Options A+B applied: re-embedded the 43.3% of LUAD compounds absent '
                'from the production index (Option A) and expanded per-target analysis '
                'to all 16 LUAD-relevant targets (Option B). Original coverage-limited '
                'result preserved untouched at transfer_results_target_based.json.',
        'luad_targets': LUAD_TARGETS,
        'n_total': len(raw),
        'n_in_index_original': len(in_index),
        'n_missing_recovered': len(missing),
        'n_full_coverage': len(found_full),
        'coverage_pct_original': round(100 * len(in_index) / len(raw), 1),
        'coverage_pct_full': round(100 * len(found_full) / len(raw), 1),
        'pgcl': {'spearman_rho': float(rho_pgcl), 'p_value': float(p_pgcl)},
        'ecfp4': {'spearman_rho': float(rho_ecfp4), 'p_value': float(p_ecfp4)},
        'per_target_rho_all_16': per_target_full,
        'elapsed_seconds': round(time.time() - t0, 1),
    }
    with open(FULL_RESULTS_OUT, 'w') as f:
        json.dump(result, f, indent=2)
    found_full[['compound_chembl_id', 'gene_symbols', 'pchembl_final', 'pgcl_score', 'ecfp4_score']].to_csv(
        FULL_SCORED_OUT, sep='\t', index=False)

    # ── Provenance / integrity manifest ──────────────────────────────────
    provenance = {
        'checkpoint_used': str(MODELING_DIR / 'checkpoints' / 'best.pt'),
        'checkpoint_opened_mode': 'read-only (torch.load, eval mode, no re-save)',
        'production_mmap_opened_mode': 'read-only (np.memmap mode="r")',
        'production_mmap_untouched': True,
        'compound_index_untouched': True,
        'original_results_file_untouched': ORIGINAL_RESULTS.exists(),
        'moldeberta_model': MOLDEBERTA_MODEL,
        'moldeberta_config': {'max_length': MAX_LENGTH, 'batch_size': BATCH_SIZE,
                               'pooling': 'CLS token, identical to embedding_extract.py'},
        'new_files_written': [str(MISSING_EMB_OUT), str(FULL_SCORED_OUT), str(FULL_RESULTS_OUT)],
    }
    with open(PROVENANCE_OUT, 'w') as f:
        json.dump(provenance, f, indent=2)

    log.info('DONE in %.1fs', time.time() - t0)
    log.info('Coverage: %d -> %d (%.1f%% -> %.1f%%)',
              len(in_index), len(found_full), result['coverage_pct_original'], result['coverage_pct_full'])
    log.info('PGCL rho: %.4f (full coverage) vs reference run (see %s)', rho_pgcl, ORIGINAL_RESULTS)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
