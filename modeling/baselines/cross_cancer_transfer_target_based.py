#!/usr/bin/env python3
"""
C1 (target-based) — Cross-Cancer Transfer Experiment: OSCC PGCL -> LUAD

Reconstructs the manuscript's Fig. 3 / P23-25 analysis: target-based LUAD
compound extraction from ChEMBL (16 LUAD-relevant oncogenes), frozen OSCC
PGCL projection, and Spearman rho against measured pChEMBL.

The original script that produced transfer_results_target_based.json no
longer exists in the repo (only its output JSON survived). This script
reconstructs the same methodology from ChEMBL directly, now that live MySQL
access is available, and reruns it against the corrected (post checkpoint-
selection-fix) canonical PGCL model.

Usage:
    python cross_cancer_transfer_target_based.py --device mps
"""
import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mysql.connector
from scipy import stats
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parent
GENESIS_DIR = MODELING_DIR.parent
RESULTS_DIR = SCRIPT_DIR / 'results' / 'cross_cancer'

sys.path.insert(0, str(MODELING_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

# Same 16-target list recorded in the surviving transfer_results_target_based.json
LUAD_TARGETS = [
    'EGFR', 'ALK', 'KRAS', 'RET', 'MET', 'ERBB2', 'HER2', 'PIK3CA', 'BRAF',
    'STK11', 'NTRK1', 'NTRK2', 'ROS1', 'NF1', 'MTOR', 'AKT1',
]

# Requires a local ChEMBL MySQL mirror (see docs/CHEMBL_MYSQL_SETUP.md).
# Credentials are never hardcoded -- set these environment variables instead:
#   export CHEMBL_MYSQL_HOST=localhost
#   export CHEMBL_MYSQL_USER=root
#   export CHEMBL_MYSQL_PASSWORD=...
#   export CHEMBL_MYSQL_DB=chembl_36
DB_CONFIG = dict(
    host=os.environ.get('CHEMBL_MYSQL_HOST', 'localhost'),
    database=os.environ.get('CHEMBL_MYSQL_DB', 'chembl_36'),
    user=os.environ.get('CHEMBL_MYSQL_USER', 'root'),
    password=os.environ.get('CHEMBL_MYSQL_PASSWORD'),
    port=int(os.environ.get('CHEMBL_MYSQL_PORT', 3306)),
)

IC50_ACTIVE = 10000  # nM

OSCC_INTERNAL = {'spearman_rho': 0.740, 'sd': 0.030}


def extract_luad_target_compounds() -> pd.DataFrame:
    """Query ChEMBL for compounds tested against the 16 LUAD targets."""
    if not DB_CONFIG['password']:
        raise RuntimeError(
            'CHEMBL_MYSQL_PASSWORD is not set. This script requires a local '
            'ChEMBL MySQL mirror -- set CHEMBL_MYSQL_HOST/USER/PASSWORD/DB '
            'environment variables (see docs/CHEMBL_MYSQL_SETUP.md), or skip '
            'this script entirely and use the pre-computed '
            'results/cross_cancer/transfer_results_target_based.json shipped '
            'with this repository.'
        )
    conn = mysql.connector.connect(**DB_CONFIG)
    placeholders = ', '.join(['%s'] * len(LUAD_TARGETS))
    sql = f"""
    SELECT
        md.chembl_id AS compound_chembl_id,
        cs.canonical_smiles AS smiles,
        act.standard_value AS ic50_nM,
        act.pchembl_value AS pchembl,
        csyn.component_synonym AS gene_symbol
    FROM activities act
    INNER JOIN molecule_dictionary md ON act.molregno = md.molregno
    LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
    INNER JOIN assays ass ON act.assay_id = ass.assay_id
    LEFT JOIN target_dictionary td ON ass.tid = td.tid
    LEFT JOIN target_components tco ON td.tid = tco.tid
    LEFT JOIN component_synonyms csyn ON tco.component_id = csyn.component_id
    WHERE csyn.component_synonym IN ({placeholders})
      AND td.organism = 'Homo sapiens'
      AND cs.canonical_smiles IS NOT NULL
      AND act.standard_type = 'IC50'
      AND act.standard_units = 'nM'
      AND act.standard_value IS NOT NULL
      AND act.standard_value <= {IC50_ACTIVE}
    GROUP BY md.chembl_id, act.activity_id, csyn.component_synonym
    """
    log.info('Querying ChEMBL for %d LUAD targets...', len(LUAD_TARGETS))
    df = pd.read_sql(sql, conn, params=tuple(LUAD_TARGETS))
    conn.close()
    log.info('Raw records: %d, unique compounds: %d', len(df), df['compound_chembl_id'].nunique())

    # pChEMBL: prefer direct value, else derive from IC50; most-potent (max pchembl) per compound
    df['pchembl_final'] = df['pchembl']
    missing = df['pchembl_final'].isna() & df['ic50_nM'].notna() & (df['ic50_nM'] > 0)
    df.loc[missing, 'pchembl_final'] = 9.0 - np.log10(df.loc[missing, 'ic50_nM'])
    df = df[df['pchembl_final'].notna() & (df['pchembl_final'] > 0)].copy()

    idx = df.groupby('compound_chembl_id')['pchembl_final'].idxmax()
    best = df.loc[idx, ['compound_chembl_id', 'smiles', 'pchembl_final']].reset_index(drop=True)
    gene_map = df.groupby('compound_chembl_id')['gene_symbol'].agg(
        lambda s: '; '.join(sorted(set(s.dropna())))).to_dict()
    best['gene_symbols'] = best['compound_chembl_id'].map(gene_map)
    log.info('Compounds with valid pChEMBL (most potent per compound): %d', len(best))
    return best


def project_and_score(df: pd.DataFrame, compound_index: dict, emb_mmap, device: str):
    from evaluate import load_model, project_ids

    found = df[df['compound_chembl_id'].isin(compound_index.keys())].copy()
    log.info('Compounds found in embedding index: %d / %d', len(found), len(df))

    ckpt = MODELING_DIR / 'checkpoints' / 'best.pt'
    log.info('Loading canonical PGCL model from %s', ckpt)
    model, _ = load_model(str(ckpt), device=device)

    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)
    vp_train_ids = [cid for cid in splits['train']['positive_ids']
                    if cid in compound_index and compound_index[cid].get('potency_class_num') == 3]
    z_vp = project_ids(model, vp_train_ids, compound_index, emb_mmap, device)
    centroid = z_vp.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    log.info('OSCC VP centroid from %d training compounds', len(vp_train_ids))

    ids = found['compound_chembl_id'].tolist()
    z = project_ids(model, ids, compound_index, emb_mmap, device)
    scores = z @ centroid
    found['pgcl_score'] = scores
    return found, centroid, splits


def ecfp4_baseline(found: pd.DataFrame, compound_index: dict, splits: dict,
                    positives_tsv: Path, n_vp_refs: int = 500) -> np.ndarray:
    pos_df = pd.read_csv(positives_tsv, sep='\t', usecols=['compound_chembl_id', 'smiles'])
    smiles_lookup = dict(zip(pos_df['compound_chembl_id'], pos_df['smiles']))
    vp_train_ids = [cid for cid in splits['train']['positive_ids']
                    if cid in compound_index and compound_index[cid].get('potency_class_num') == 3]
    rng = np.random.default_rng(42)
    if len(vp_train_ids) > n_vp_refs:
        vp_train_ids = list(rng.choice(vp_train_ids, n_vp_refs, replace=False))
    vp_fps = []
    for cid in vp_train_ids:
        smi = smiles_lookup.get(cid)
        if smi:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                vp_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
    log.info('VP reference fingerprints: %d', len(vp_fps))

    scores = np.zeros(len(found))
    for i, smi in enumerate(found['smiles'].tolist()):
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        if mol and vp_fps:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, vp_fps)
            scores[i] = max(sims) if sims else 0.0
    return scores


def per_target_rho(found: pd.DataFrame, top_targets=('EGFR', 'PIK3CA', 'ALK', 'KRAS'), n_boot=2000) -> dict:
    rng = np.random.default_rng(42)
    out = {}
    for t in top_targets:
        mask = found['gene_symbols'].fillna('').str.contains(t)
        sub = found[mask]
        if len(sub) >= 10:
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
            out[t] = {'rho': float(rho), 'p': float(p), 'n': int(len(sub)),
                       'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = extract_luad_target_compounds()
    raw.to_csv(RESULTS_DIR / 'luad_target_based_raw.tsv', sep='\t', index=False)

    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    n_compounds = len(np.load(GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True))
    emb_mmap = np.memmap(GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
                         dtype='float16', mode='r', shape=(n_compounds, 768))

    found, centroid, splits = project_and_score(raw, compound_index, emb_mmap, args.device)

    ecfp4 = ecfp4_baseline(found, compound_index, splits, MODELING_DIR / 'positives.tsv')
    found['ecfp4_score'] = ecfp4
    found[['compound_chembl_id', 'gene_symbols', 'pchembl_final', 'pgcl_score', 'ecfp4_score']].to_csv(
        RESULTS_DIR / 'luad_target_based_scored.tsv', sep='\t', index=False)

    rho_pgcl, p_pgcl = stats.spearmanr(found['pgcl_score'], found['pchembl_final'])
    rho_ecfp4, p_ecfp4 = stats.spearmanr(ecfp4, found['pchembl_final'])

    per_target = per_target_rho(found)

    result = {
        'experiment': 'C1_target_based',
        'luad_targets': LUAD_TARGETS,
        'n_total': len(raw),
        'n_in_index': len(found),
        'pchembl_range': [float(found['pchembl_final'].min()), float(found['pchembl_final'].max())],
        'pgcl': {'spearman_rho': float(rho_pgcl), 'p_value': float(p_pgcl)},
        'ecfp4': {'spearman_rho': float(rho_ecfp4), 'p_value': float(p_ecfp4)},
        'oscc_internal': OSCC_INTERNAL,
        'per_target_rho': per_target,
        'interpretation': (
            'SUBSTANTIAL TRANSFER (pan-cancer potency geometry)' if rho_pgcl > 0.40
            else 'PARTIAL TRANSFER' if rho_pgcl > 0.20
            else 'NO TRANSFER (disease-specific)'
        ),
    }

    print('\n' + '=' * 70)
    print('C1 (target-based): OSCC PGCL -> LUAD (16 targets)')
    print('=' * 70)
    print(f'n_total={len(raw)}  n_in_index={len(found)}')
    print(f'PGCL  Spearman rho = {rho_pgcl:.4f} (p={p_pgcl:.2e})')
    print(f'ECFP4 Spearman rho = {rho_ecfp4:.4f} (p={p_ecfp4:.2e})')
    print(f'OSCC internal (reference) rho = {OSCC_INTERNAL["spearman_rho"]:.3f}')
    for t, v in per_target.items():
        print(f'  {t}: rho={v["rho"]:.3f} (n={v["n"]})')
    print(f'Interpretation: {result["interpretation"]}')
    print('=' * 70)

    out = RESULTS_DIR / 'transfer_results_target_based.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    log.info('Saved -> %s', out)


if __name__ == '__main__':
    main()
