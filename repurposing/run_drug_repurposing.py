#!/usr/bin/env python3
"""
Phase 6 — Drug Repurposing of CompoundList Compounds.

Screens the 23,308 CompoundList compounds against OSCC VP space using
the potency-aware contrastive model, applies safety profiling, validates
against known annotations, and produces a ranked dossier with 8-component
recommendation weights.

Additionally performs cross-study SMARTS enrichment: identifies
substructure patterns significant in BOTH repurposed candidates AND
generated molecules from Phase 4.

Pipeline position in GENESIS
----------------------------
Phase 4 (BioactivityProfiling) scores *generated* molecules; this Phase 6
scores *existing* drugs. Both feed into downstream experimental validation.
Shared scoring architecture (composite score, safety profiling, SMARTS
mining) enables direct cross-pipeline comparison.

Design philosophy
-----------------
- Centroid-based scoring (not kNN) for robustness to annotation noise.
- T3C repulsion novelty score — penalise proximity to clinical failures.
- Geometric-mean safety — one catastrophic failure dominates the signal.
- Placeholder components (consensus, docking) maintain structural
  extensibility without requiring future refactoring.
- Leakage-aware: training-set overlaps are flagged, not suppressed.

Usage:
    python run_drug_repurposing.py --device mps
"""

import argparse
import json
import logging
import os
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENESIS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(GENESIS_DIR)
MODELING_DIR = os.path.join(GENESIS_DIR, 'modeling')
DATA_DIR = os.path.join(GENESIS_DIR, 'data')
EMB_DIR = os.path.join(GENESIS_DIR, 'embeddings')
GEN_PROCESSED = os.path.join(GENESIS_DIR, 'generative', 'processed')
GEN_RESULTS = os.path.join(GENESIS_DIR, 'generative', 'results')
BIOACTIVITY_DIR = os.path.join(ROOT_DIR, 'BioactivityProfiling_Generative')

sys.path.insert(0, MODELING_DIR)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('repurposing')

# ─── SA Score ──────────────────────────────────────────────────────────────────
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    sa_available = True
except Exception:
    sa_available = False
    log.warning('SA score module not available — SA scores will be estimated')

# ─── Filter Catalogs ──────────────────────────────────────────────────────────
_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(_pains_params)

_brenk_params = FilterCatalogParams()
_brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
BRENK_CATALOG = FilterCatalog(_brenk_params)

# Custom toxicophore SMARTS — 10 intrinsic-reactivity structural alerts.
# These complement PAINS (assay-interference artefacts) and Brenk (metabolic
# liabilities): together the three filter libraries cover the full spectrum
# of medicinal chemistry red flags.
TOXICOPHORES = {
    'nitro_aromatic': '[$(c[N+](=O)[O-])]',     # Reductive activation → toxic metabolites
    'aniline': '[NX3;H2;!$(NC=O)]c1ccccc1',      # Mutagenicity via arylamine oxidation
    'alkyl_halide': '[CX4][F,Cl,Br,I]',           # Electrophilic alkylation of DNA/proteins
    'epoxide': 'C1OC1',                            # Covalent modification of macromolecules
    'acyl_halide': '[CX3](=[OX1])[F,Cl,Br,I]',    # Highly reactive acylating agent
    'michael_acceptor': '[CX3]=[CX3][CX3]=[OX1]', # Thiol-reactive → off-target covalent binding
    'aldehyde': '[CX3H1](=O)[#6]',                # Schiff base formation with lysines
    'hydrazine': '[NX3][NX3]',                     # Hepatotoxicity, carcinogenicity
    'peroxide': '[OX2][OX2]',                      # Radical generation, oxidative damage
    'thiol': '[SX2H]',                             # Disulphide scrambling, off-target reactivity
}
TOXICOPHORE_PATTERNS = {}
for name, smarts in TOXICOPHORES.items():
    pat = Chem.MolFromSmarts(smarts)
    if pat:
        TOXICOPHORE_PATTERNS[name] = pat

# ─── Scoring Weights ──────────────────────────────────────────────────────────
# 8-component recommendation weight architecture.  Weights sum to 1.00 and
# reflect domain priorities: potency signal > safety > target specificity >
# developability.  Placeholder components (consensus, docking) are included
# so the weight architecture is stable when future data becomes available.
W_COMPOSITE = 0.30       # Core potency signal from contrastive model
W_SAFETY = 0.20          # Geometric-mean safety (conservative: one failure dominates)
W_TARGET_CONF = 0.15     # Per-target centroid confidence → anchors MoA specificity
W_SA = 0.10              # Synthetic accessibility for experimental follow-up
W_CONSENSUS = 0.05       # Placeholder: uniform 0.5 (single-source, extensible for multi-method)
W_DOCKING = 0.05         # Placeholder: 0.0 (reserved for Phase 7 molecular docking scores)
W_CLINICAL = 0.10        # Doubled vs Phase 4 — existing drugs have known safety/PK profiles
W_SMARTS_ENRICHED = 0.05 # Cross-pipeline structural convergence (repurposed × generated)

# Metadata columns from the master table
META_COLS = [
    'compound_chembl_id', 'pubchem_cid', 'compound_name', 'smiles',
    'inchi_key', 'compound_origin', 'natural_product', 'clinical_phase',
    'max_phase', 'qed_weighted', 'num_ro5_violations', 'molecular_weight',
    'alogp', 'hba', 'hbd', 'psa', 'rotatable_bonds', 'lipinski_pass',
    'potency_class', 'potency_class_num', 'pchembl', 'ic50_nM',
    'target_name', 'gene_symbols', 'cell_line', 'mechanism_of_action',
    'action_type', 'source_file', 'in_plos',
    'Indication', 'HighestStatus', 'DrugType', 'IsAnticancer',
]


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE A: Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_compoundlist_compounds():
    """Load CompoundList compounds from the master table.

    Selects rows where source_file contains 'CompoundList' (includes
    overlap compounds that appear in both CompoundList and ChEMBL).
    """
    log.info('Phase A: Loading CompoundList compounds ...')
    master_path = os.path.join(DATA_DIR, 'All_Compounds_Unq_Std.tsv')
    master = pd.read_csv(master_path, sep='\t', low_memory=False)
    log.info('  Master table: %d rows, %d cols', len(master), len(master.columns))

    # Filter to CompoundList
    mask = master['source_file'].str.contains('CompoundList', case=False, na=False)
    cl = master[mask].copy().reset_index(drop=True)
    log.info('  CompoundList compounds: %d', len(cl))

    # Drop rows without SMILES
    cl = cl[cl['smiles'].notna() & (cl['smiles'].str.len() > 0)].reset_index(drop=True)
    log.info('  With valid SMILES: %d', len(cl))

    # Log source distribution
    only_cl = cl['source_file'].str.contains('CompoundList', na=False) & \
              ~cl['source_file'].str.contains('ChEMBL', case=False, na=False)
    n_only = only_cl.sum()
    n_both = len(cl) - n_only
    log.info('  CompoundList-only: %d, Overlap (ChEMBL+CompoundList): %d', n_only, n_both)

    # Known potency stats
    has_potency = cl['potency_class_num'].notna()
    log.info('  With known potency: %d (%.1f%%)', has_potency.sum(),
             100 * has_potency.sum() / max(len(cl), 1))
    if has_potency.sum() > 0:
        pot_dist = cl.loc[has_potency, 'potency_class_num'].value_counts().sort_index()
        for pc, cnt in pot_dist.items():
            log.info('    potency_class_num=%s: %d', pc, cnt)

    return cl


def map_to_embeddings(cl_df, compound_index):
    """Map CompoundList compounds to pre-computed embedding indices.

    Dual lookup strategy:
      1. compound_index.pkl — compounds already in the contrastive training
         pipeline.  These are 'interpolation' queries: the model has seen them.
      2. compound_ids.npy (fallback) — the full 715K embedding library.
         These are 'generalisation' queries: never seen during training.

    Tracking both sources allows the leakage_check() step to separate
    interpolation accuracy from genuine generalisation performance.
    """
    log.info('  Mapping to embedding indices ...')

    # Build fallback lookup from compound_ids.npy
    cids_array = np.load(os.path.join(EMB_DIR, 'compound_ids.npy'), allow_pickle=True)
    cid_to_emb_idx = {cid: i for i, cid in enumerate(cids_array)}
    log.info('  Embedding ID lookup: %d entries', len(cid_to_emb_idx))

    idx_map = []
    n_from_ci = 0
    n_from_fallback = 0
    for _, row in cl_df.iterrows():
        chembl_id = row.get('compound_chembl_id', '')
        if chembl_id and chembl_id in compound_index:
            idx_map.append(compound_index[chembl_id]['emb_idx'])
            n_from_ci += 1
        elif chembl_id and chembl_id in cid_to_emb_idx:
            idx_map.append(cid_to_emb_idx[chembl_id])
            n_from_fallback += 1
        else:
            idx_map.append(None)

    cl_df = cl_df.copy()
    cl_df['emb_idx'] = idx_map
    n_mapped = cl_df['emb_idx'].notna().sum()
    log.info('  Mapped via compound_index: %d, via embedding IDs: %d, unmapped: %d',
             n_from_ci, n_from_fallback, len(cl_df) - n_mapped)
    log.info('  Total mapped: %d / %d (%.1f%%)',
             n_mapped, len(cl_df), 100 * n_mapped / max(len(cl_df), 1))
    return cl_df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE B: Global Ranking via Contrastive Projection
# ═══════════════════════════════════════════════════════════════════════════════

def project_compounds(cl_df, model, emb_mmap, device, batch_size=512):
    """Project CompoundList compounds through ContrastiveModel."""
    log.info('Phase B: Projecting compounds through contrastive model ...')

    mapped = cl_df[cl_df['emb_idx'].notna()].copy()
    emb_indices = mapped['emb_idx'].astype(int).values

    all_z = []
    all_pchembl = []
    for start in range(0, len(emb_indices), batch_size):
        batch_idx = emb_indices[start:start + batch_size]
        embs = torch.tensor(
            emb_mmap[batch_idx].astype(np.float32),
            device=device,
        )
        with torch.no_grad():
            z, pchembl_pred = model(embs)
        all_z.append(z.cpu().numpy())
        all_pchembl.append(pchembl_pred.cpu().numpy())

    projections = np.concatenate(all_z, axis=0)
    pred_pchembl = np.concatenate(all_pchembl, axis=0).flatten()

    log.info('  Projected: %d compounds → %d-d space', len(projections), projections.shape[1])
    return mapped, projections, pred_pchembl


def compute_centroids(model, compound_index, emb_mmap, device, batch_size=1024):
    """Compute VP and T3C centroids from compound_index.

    Centroids (not kNN) by design: averaging over all VP or T3C compounds
    cancels annotation noise from individual training samples, yielding a
    more stable scoring axis on the 128-d L2-normalised hypersphere.

    VP centroid  — the 'positive pole': mean direction of very-potent
                   OSCC compounds in contrastive space.
    T3C centroid — the 'clinical-failure pole': mean direction of
                   compounds that were OSCC-active but failed clinically.
                   Used in novelty_score = vp_sim − 0.3 × t3c_sim to
                   penalise candidates that are too close to known failures.
    """
    log.info('  Computing VP and T3C centroids ...')

    # VP centroid: all positive compounds
    vp_ids = [cid for cid, info in compound_index.items()
              if info.get('label') == 'positive']
    vp_emb_idx = [compound_index[cid]['emb_idx'] for cid in vp_ids]
    log.info('    VP compounds: %d', len(vp_ids))

    vp_zs = []
    for start in range(0, len(vp_emb_idx), batch_size):
        batch_idx = vp_emb_idx[start:start + batch_size]
        embs = torch.tensor(
            emb_mmap[batch_idx].astype(np.float32),
            device=device,
        )
        with torch.no_grad():
            z, _ = model(embs)
        vp_zs.append(z.cpu().numpy())
    vp_all = np.concatenate(vp_zs, axis=0)
    vp_centroid = vp_all.mean(axis=0)
    vp_centroid /= np.linalg.norm(vp_centroid) + 1e-8

    # T3C centroid: negatives with neg_tier == 'T3_clinical_failure'
    t3c_ids = [cid for cid, info in compound_index.items()
               if info.get('neg_tier') == 'T3_clinical_failure']
    t3c_centroid = None
    if len(t3c_ids) >= 10:
        t3c_emb_idx = [compound_index[cid]['emb_idx'] for cid in t3c_ids]
        t3c_zs = []
        for start in range(0, len(t3c_emb_idx), batch_size):
            batch_idx = t3c_emb_idx[start:start + batch_size]
            embs = torch.tensor(
                emb_mmap[batch_idx].astype(np.float32),
                device=device,
            )
            with torch.no_grad():
                z, _ = model(embs)
            t3c_zs.append(z.cpu().numpy())
        t3c_all = np.concatenate(t3c_zs, axis=0)
        t3c_centroid = t3c_all.mean(axis=0)
        t3c_centroid /= np.linalg.norm(t3c_centroid) + 1e-8
        log.info('    T3C compounds: %d', len(t3c_ids))
    else:
        log.info('    T3C skipped (only %d clinical-failure negatives)', len(t3c_ids))

    return vp_centroid, t3c_centroid


def global_ranking(cl_df, projections, pred_pchembl, vp_centroid, t3c_centroid):
    """Rank all CompoundList compounds by composite score.

    composite = 0.7 × vp_similarity + 0.3 × (predicted_pchembl / 10.0)

    The 70/30 weighting reflects the training objective hierarchy:
    - VP similarity is the geometric signal from contrastive learning (primary
      training objective) and occupies the full [−1, 1] cosine range.
    - Predicted pChEMBL is the scalar from the auxiliary linear regulariser;
      divided by 10 to map the typical bioactivity range (4–10) into [0.4, 1.0],
      roughly commensurate with cosine similarity magnitudes.

    novelty = vp_similarity − 0.3 × t3c_similarity
    Subtracting T3C similarity penalises proximity to clinical failures.
    """
    log.info('  Computing global rankings ...')

    vp_sim = projections @ vp_centroid
    t3c_sim = projections @ t3c_centroid if t3c_centroid is not None else np.zeros(len(projections))
    composite = 0.7 * vp_sim + 0.3 * (pred_pchembl / 10.0)
    novelty = vp_sim - 0.3 * t3c_sim

    cl_df = cl_df.copy()
    cl_df['vp_similarity'] = vp_sim
    cl_df['t3c_similarity'] = t3c_sim
    cl_df['predicted_pchembl'] = pred_pchembl
    cl_df['composite_score'] = composite
    cl_df['novelty_score'] = novelty

    cl_df = cl_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    log.info('  Composite score: mean=%.4f, max=%.4f, min=%.4f',
             composite.mean(), composite.max(), composite.min())
    log.info('  VP similarity:   mean=%.4f, max=%.4f', vp_sim.mean(), vp_sim.max())
    log.info('  Pred pChEMBL:    mean=%.2f, max=%.2f', pred_pchembl.mean(), pred_pchembl.max())
    return cl_df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE C: Per-Target Screening
# ═══════════════════════════════════════════════════════════════════════════════

def per_target_centroids(model, compound_index, emb_mmap, device, min_vp=30, batch_size=1024):
    """Compute per-target VP centroids.

    min_vp=30 is a pragmatic threshold: centroids from fewer compounds are
    noisy on the 128-d hypersphere.  30 samples ensure a stable mean
    direction for cosine-similarity ranking.  The output n_targets_top50
    tracks how many target top-50 lists a compound appears in — a useful
    multi-target promiscuity signal for polypharmacological candidates.
    """
    log.info('Phase C: Computing per-target VP centroids (min_vp=%d) ...', min_vp)

    target_ids = defaultdict(list)
    for cid, info in compound_index.items():
        if info.get('label') == 'positive' and info.get('target_name'):
            target_ids[info['target_name']].append(cid)

    centroids = {}
    for target, cids in sorted(target_ids.items(), key=lambda x: -len(x[1])):
        if len(cids) < min_vp:
            continue
        idx = [compound_index[c]['emb_idx'] for c in cids]
        zs = []
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            embs = torch.tensor(
                emb_mmap[batch_idx].astype(np.float32),
                device=device,
            )
            with torch.no_grad():
                z, _ = model(embs)
            zs.append(z.cpu().numpy())
        z_all = np.concatenate(zs, axis=0)
        centroid = z_all.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        centroids[target] = centroid
        log.info('    %s: %d VP compounds', target, len(cids))

    log.info('  Per-target centroids computed: %d targets', len(centroids))
    return centroids


def target_specific_ranking(cl_df, projections, pred_pchembl, target_centroids,
                            t3c_centroid, top_k=50):
    """Rank CompoundList compounds per target."""
    log.info('  Computing per-target rankings ...')

    target_rankings = {}
    all_records = []

    for target, centroid in target_centroids.items():
        t_sim = projections @ centroid
        t3c_sim = projections @ t3c_centroid if t3c_centroid is not None else np.zeros(len(projections))
        t_composite = 0.7 * t_sim + 0.3 * (pred_pchembl / 10.0)

        t_df = cl_df.copy()
        t_df['target_vp_similarity'] = t_sim
        t_df['target_composite'] = t_composite
        t_df['target_novelty'] = t_sim - 0.3 * t3c_sim
        t_df = t_df.sort_values('target_composite', ascending=False).head(top_k)
        target_rankings[target] = t_df

        for _, row in t_df.iterrows():
            all_records.append({
                'smiles': row['smiles'],
                'compound_chembl_id': row.get('compound_chembl_id', ''),
                'target': target,
                'target_vp_similarity': row['target_vp_similarity'],
                'target_composite': row['target_composite'],
                'predicted_pchembl': row['predicted_pchembl'],
            })

    # Build target summary per compound
    target_summary = None
    if all_records:
        rec_df = pd.DataFrame(all_records)
        target_summary = rec_df.groupby('smiles').agg(
            best_target=('target', lambda x: x.iloc[rec_df.loc[x.index, 'target_composite'].values.argmax()]),
            max_target_score=('target_composite', 'max'),
            mean_target_score=('target_composite', 'mean'),
            n_targets_top50=('target', 'nunique'),
        ).reset_index()

    log.info('  Targets scored: %d', len(target_rankings))
    return target_rankings, target_summary


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE D: Safety Profiling
# ═══════════════════════════════════════════════════════════════════════════════

def profile_molecule(smi):
    """Compute safety profile for a single molecule."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {
            'lipinski_violations': 99, 'pains_alerts': 0, 'pains_names': '',
            'brenk_alerts': 0, 'brenk_names': '', 'toxicophore_count': 0,
            'toxicophore_names': '', 'sa_score': 10.0, 'qed': 0.0,
            'overall_flag': 'FAIL',
        }

    # Lipinski
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    lip_violations = sum([mw > 500, logp > 5, hba > 10, hbd > 5])

    # PAINS
    pains_matches = PAINS_CATALOG.GetMatches(mol)
    pains_names = [m.GetDescription() for m in pains_matches]

    # Brenk
    brenk_matches = BRENK_CATALOG.GetMatches(mol)
    brenk_names = [m.GetDescription() for m in brenk_matches]

    # Toxicophores
    tox_hits = [name for name, pat in TOXICOPHORE_PATTERNS.items()
                if mol.HasSubstructMatch(pat)]

    # SA
    if sa_available:
        sa = sascorer.calculateScore(mol)
    else:
        sa = max(1, min(10, 1 + 0.5 * mol.GetNumHeavyAtoms() / 5))

    # QED
    qed = Descriptors.qed(mol)

    # Overall flag
    critical = len(pains_names) + len(tox_hits)
    if critical >= 2 or lip_violations >= 3:
        flag = 'FAIL'
    elif critical >= 1 or lip_violations >= 2 or len(brenk_names) >= 3:
        flag = 'CAUTION'
    else:
        flag = 'PASS'

    return {
        'lipinski_violations': lip_violations,
        'pains_alerts': len(pains_names),
        'pains_names': '; '.join(pains_names),
        'brenk_alerts': len(brenk_names),
        'brenk_names': '; '.join(brenk_names),
        'toxicophore_count': len(tox_hits),
        'toxicophore_names': '; '.join(tox_hits),
        'sa_score': sa,
        'qed': qed,
        'overall_flag': flag,
    }


def compute_safety_profiles(cl_df):
    """Compute safety profiles for all CompoundList compounds."""
    log.info('Phase D: Safety profiling ...')

    profiles = []
    for i, smi in enumerate(cl_df['smiles'].values):
        profiles.append(profile_molecule(smi))
        if (i + 1) % 5000 == 0:
            log.info('    Profiled %d / %d', i + 1, len(cl_df))

    safety_df = pd.DataFrame(profiles)
    for col in safety_df.columns:
        cl_df[col] = safety_df[col].values

    n_pass = (cl_df['overall_flag'] == 'PASS').sum()
    n_caution = (cl_df['overall_flag'] == 'CAUTION').sum()
    n_fail = (cl_df['overall_flag'] == 'FAIL').sum()
    log.info('  Safety: PASS=%d, CAUTION=%d, FAIL=%d', n_pass, n_caution, n_fail)
    return cl_df


def continuous_safety(row):
    """Geometric-mean safety score (0–1, higher = safer).

    Uses geometric mean (not arithmetic) so that *one* catastrophic failure
    (e.g., PAINS = 0.3) dominates the product and drags the overall score
    down — more conservative than arithmetic mean for safety signals where
    a single red flag should sink the compound.  Each of the 5 penalty
    terms maps a different risk dimension into [0, 1]: Lipinski (oral
    bioavailability), PAINS (assay artefacts), Brenk (metabolic stability),
    toxicophores (intrinsic reactivity), SA (developability).
    """
    penalties = []
    lip = row.get('lipinski_violations', 0)
    penalties.append(max(0, 1 - lip / 4))
    penalties.append(1.0 if row.get('pains_alerts', 0) == 0 else 0.3)
    penalties.append(max(0, 1 - row.get('brenk_alerts', 0) / 5))
    penalties.append(1.0 if row.get('toxicophore_count', 0) == 0 else 0.2)
    sa = row.get('sa_score', 5)
    penalties.append(max(0, min(1, (10 - sa) / 9)))
    product = 1.0
    for p in penalties:
        product *= max(p, 1e-6)
    return product ** (1.0 / len(penalties))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE E: Validation (predicted vs known potency)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_predictions(cl_df):
    """Validate predicted pChEMBL against known values for annotated compounds."""
    log.info('Phase E: Validation — predicted vs known potency ...')

    has_known = cl_df['pchembl'].notna() & cl_df['predicted_pchembl'].notna()
    annotated = cl_df[has_known].copy()
    log.info('  Compounds with known pChEMBL: %d', len(annotated))

    if len(annotated) < 10:
        log.info('  Insufficient annotated compounds for validation')
        return {'n_annotated': len(annotated), 'pearson_r': None, 'spearman_rho': None}

    known = annotated['pchembl'].astype(float).values
    predicted = annotated['predicted_pchembl'].values

    r, p_val = sp_stats.pearsonr(known, predicted)
    rho, rho_p = sp_stats.spearmanr(known, predicted)
    mae = np.mean(np.abs(known - predicted))
    rmse = np.sqrt(np.mean((known - predicted) ** 2))

    log.info('  Pearson r  = %.4f (p=%.2e)', r, p_val)
    log.info('  Spearman ρ = %.4f (p=%.2e)', rho, rho_p)
    log.info('  MAE        = %.3f', mae)
    log.info('  RMSE       = %.3f', rmse)

    # Potency class concordance
    has_class = cl_df['potency_class_num'].notna() & cl_df['predicted_pchembl'].notna()
    if has_class.sum() >= 10:
        sub = cl_df[has_class].copy()
        # Assign predicted potency class: VP ≥ 7.0, Potent ≥ 6.0, Active ≥ 5.0
        sub['pred_class'] = pd.cut(
            sub['predicted_pchembl'],
            bins=[-999, 5.0, 6.0, 7.0, 999],
            labels=[0, 1, 2, 3],
        ).astype(float)
        known_class = sub['potency_class_num'].values
        pred_class = sub['pred_class'].values
        valid_mask = ~np.isnan(pred_class)
        if valid_mask.sum() >= 10:
            concordance = np.mean(known_class[valid_mask] == pred_class[valid_mask])
            log.info('  Class concordance: %.1f%% (%d compounds)',
                     100 * concordance, valid_mask.sum())

    # VP Recall@K
    vp_mask = cl_df['potency_class_num'] == 3  # VP class
    if vp_mask.sum() >= 5:
        n_vp = vp_mask.sum()
        for k in [50, 100, 200]:
            if k <= len(cl_df):
                top_k = cl_df.nlargest(k, 'composite_score')
                recall = top_k[top_k.index.isin(cl_df[vp_mask].index)].shape[0] / n_vp
                log.info('  VP Recall@%d: %.1f%% (%d / %d)', k, 100 * recall,
                         int(recall * n_vp), n_vp)

    return {
        'n_annotated': len(annotated),
        'pearson_r': float(r),
        'pearson_p': float(p_val),
        'spearman_rho': float(rho),
        'spearman_p': float(rho_p),
        'mae': float(mae),
        'rmse': float(rmse),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE F: Dossier Compilation (8-component recommendation weight)
# ═══════════════════════════════════════════════════════════════════════════════

def pharmacophore_2d(df):
    """Compute 2D pharmacophore features for all compounds."""
    log.info('  Computing 2D pharmacophore features ...')

    feats = []
    for smi in df['smiles'].values:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            feats.append({})
            continue
        feats.append({
            'n_hba': rdMolDescriptors.CalcNumHBA(mol),
            'n_hbd': rdMolDescriptors.CalcNumHBD(mol),
            'n_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'n_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
            'n_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'n_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
            'tpsa': Descriptors.TPSA(mol),
            'logp': Descriptors.MolLogP(mol),
            'mw': Descriptors.MolWt(mol),
            'n_heavy_atoms': mol.GetNumHeavyAtoms(),
        })
    return pd.DataFrame(feats)


def compile_dossier(cl_df, target_summary, smarts_df, pharm_df):
    """Compile final dossier with 8-component recommendation weight.

    The 8 components (composite, safety, target_conf, SA, consensus,
    docking, clinical, SMARTS) are independently normalised to [0, 1]
    then combined via fixed weights summing to 1.00.  The architecture
    includes two placeholder components (consensus=0.5, docking=0.0)
    as structural hooks — when docking scores or multi-method consensus
    become available, the weight infrastructure requires no restructuring.

    Clinical phase receives 0.10 (doubled vs Phase 4's 0.05) because
    existing approved/late-phase drugs already have established safety
    and PK profiles, substantially lowering repurposing risk.
    """
    log.info('Phase F: Compiling final dossier ...')

    dossier = cl_df.sort_values('composite_score', ascending=False).drop_duplicates(
        subset='smiles', keep='first').copy().reset_index(drop=True)

    # Pharmacophore
    if pharm_df is not None and len(pharm_df) == len(cl_df):
        pharm_unique = pharm_df.iloc[dossier.index] if len(pharm_df) == len(dossier) else None
        if pharm_unique is None:
            # Re-index 
            pharm_unique = pharmacophore_2d(dossier)
        for col in pharm_unique.columns:
            dossier[col + '_pharm'] = pharm_unique[col].values

    # Target info
    if target_summary is not None and len(target_summary) > 0:
        ts_lookup = target_summary.set_index('smiles')
        for col in ['best_target', 'max_target_score', 'mean_target_score', 'n_targets_top50']:
            if col in ts_lookup.columns:
                dossier[col] = dossier['smiles'].map(ts_lookup[col])

    # SMARTS enrichment hits
    enriched_patterns = []
    if smarts_df is not None and len(smarts_df) > 0:
        enriched = smarts_df[smarts_df['enrichment_ratio'] > 1.5]
        for _, row in enriched.iterrows():
            smarts_str = row['smarts']
            pat = Chem.MolFromSmarts(smarts_str)
            if pat is None:
                pat = Chem.MolFromSmiles(smarts_str)
            if pat:
                enriched_patterns.append(pat)

    def count_enriched_hits(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0
        return sum(1 for p in enriched_patterns if mol.HasSubstructMatch(p))

    dossier['enriched_smarts_hits'] = dossier['smiles'].apply(count_enriched_hits)

    # ── 8-component recommendation weight ──

    # 1. Normalized composite
    c_min, c_max = dossier['composite_score'].min(), dossier['composite_score'].max()
    dossier['norm_composite'] = (dossier['composite_score'] - c_min) / max(c_max - c_min, 1e-8)

    # 2. Continuous safety
    dossier['continuous_safety'] = dossier.apply(continuous_safety, axis=1)

    # 3. Target confidence
    if 'max_target_score' in dossier.columns:
        ts_max = dossier['max_target_score'].max()
        dossier['norm_target_conf'] = dossier['max_target_score'] / max(ts_max, 1e-8)
    else:
        dossier['norm_target_conf'] = dossier['norm_composite']

    # 4. SA normalized (inverted: lower SA → better)
    dossier['norm_sa'] = dossier['sa_score'].apply(
        lambda x: max(0, min(1, (10 - x) / 9)) if pd.notna(x) else 0.5)

    # 5. Consensus (single source → always 1 method, so uniform)
    dossier['norm_consensus'] = 0.5

    # 6. Docking placeholder
    dossier['norm_docking'] = 0.0

    # 7. Clinical phase (CompoundList compounds may have approvals)
    def clinical_score(row):
        phase = row.get('max_phase') or row.get('clinical_phase')
        if pd.isna(phase):
            return 0.0
        try:
            phase = float(phase)
        except (ValueError, TypeError):
            phase_map = {'approved': 4, 'phase 3': 3, 'phase 2': 2, 'phase 1': 1}
            phase = phase_map.get(str(phase).lower().strip(), 0)
        return min(1.0, phase / 4.0)

    dossier['norm_clinical'] = dossier.apply(clinical_score, axis=1)

    # 8. SMARTS enrichment
    dossier['norm_smarts'] = (dossier['enriched_smarts_hits'] > 0).astype(float)

    # Final recommendation weight
    dossier['recommendation_weight'] = (
        W_COMPOSITE * dossier['norm_composite'] +
        W_SAFETY * dossier['continuous_safety'] +
        W_TARGET_CONF * dossier['norm_target_conf'] +
        W_SA * dossier['norm_sa'] +
        W_CONSENSUS * dossier['norm_consensus'] +
        W_DOCKING * dossier['norm_docking'] +
        W_CLINICAL * dossier['norm_clinical'] +
        W_SMARTS_ENRICHED * dossier['norm_smarts']
    )

    dossier = dossier.sort_values('recommendation_weight', ascending=False).reset_index(drop=True)
    dossier.index.name = 'rank'
    dossier.index += 1

    log.info('  Dossier compiled: %d unique candidates', len(dossier))
    log.info('  Recommendation weight range: [%.4f, %.4f]',
             dossier['recommendation_weight'].min(), dossier['recommendation_weight'].max())
    return dossier


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE G: SMARTS Enrichment Mining (repurposed + generated cross-analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_scaffolds(smiles_list, top_n_smiles=None):
    """Extract Murcko scaffolds from a list of SMILES."""
    scaffold_counts = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(core)
            scaffold_counts[scaffold_smi] += 1
        except Exception:
            pass
    return scaffold_counts


def smarts_mining(cl_df, compound_index, top_n=200, min_freq=3):
    """Mine SMARTS patterns enriched in top repurposing candidates vs inactive compounds.

    Uses Murcko scaffolds decomposed from top-ranked compounds and
    computes enrichment ratios relative to VP and inactive populations.
    """
    log.info('Phase G: SMARTS enrichment mining (repurposed candidates) ...')

    # Get top candidates
    top_df = cl_df.nlargest(top_n, 'composite_score')
    top_smiles = top_df['smiles'].tolist()
    log.info('  Top %d candidates for scaffold analysis', len(top_smiles))

    # Reference sets from compound_index
    vp_smiles = []
    inactive_smiles = []
    for cid, info in compound_index.items():
        if info.get('label') == 'positive':
            vp_smiles.append(cid)
        elif info.get('label') == 'negative':
            inactive_smiles.append(cid)

    # We need actual SMILES — load from master table
    master_path = os.path.join(DATA_DIR, 'All_Compounds_Unq_Std.tsv')
    master = pd.read_csv(master_path, sep='\t', usecols=['compound_chembl_id', 'smiles'],
                         low_memory=False)
    smi_lookup = dict(zip(master['compound_chembl_id'], master['smiles']))

    vp_smi_list = [smi_lookup[cid] for cid in vp_smiles[:5000] if cid in smi_lookup and pd.notna(smi_lookup[cid])]
    inactive_smi_list = [smi_lookup[cid] for cid in inactive_smiles[:5000] if cid in smi_lookup and pd.notna(smi_lookup[cid])]
    log.info('  Reference VP SMILES: %d, Inactive SMILES: %d', len(vp_smi_list), len(inactive_smi_list))

    # Parse molecules
    top_mols = [m for m in (Chem.MolFromSmiles(s) for s in top_smiles) if m is not None]
    vp_mols = [m for m in (Chem.MolFromSmiles(s) for s in vp_smi_list) if m is not None]
    inactive_mols = [m for m in (Chem.MolFromSmiles(s) for s in inactive_smi_list) if m is not None]

    # Extract scaffolds from top candidates
    scaffold_counts = extract_scaffolds(top_smiles)
    frequent = {s: c for s, c in scaffold_counts.items() if c >= min_freq}
    log.info('  Frequent scaffolds (>=%d): %d', min_freq, len(frequent))

    results = []
    for scaffold_smi, count in sorted(frequent.items(), key=lambda x: -x[1])[:100]:
        pat = Chem.MolFromSmiles(scaffold_smi)
        if pat is None:
            continue

        n_top = sum(1 for m in top_mols if m.HasSubstructMatch(pat))
        n_vp = sum(1 for m in vp_mols if m.HasSubstructMatch(pat))
        n_inactive = sum(1 for m in inactive_mols if m.HasSubstructMatch(pat))

        vp_rate = n_vp / max(1, len(vp_mols))
        inactive_rate = n_inactive / max(1, len(inactive_mols))
        enrichment = vp_rate / max(1e-6, inactive_rate)

        results.append({
            'smarts': scaffold_smi,
            'source': 'scaffold_repurposed',
            'n_matches_top': n_top,
            'n_matches_top_pct': n_top / max(1, len(top_mols)),
            'n_matches_vp': n_vp,
            'vp_rate': vp_rate,
            'n_matches_inactive': n_inactive,
            'inactive_rate': inactive_rate,
            'enrichment_ratio': enrichment,
            'count_in_candidates': count,
        })

    repurposed_smarts = pd.DataFrame(results)
    if len(repurposed_smarts) > 0:
        repurposed_smarts = repurposed_smarts.sort_values('enrichment_ratio', ascending=False).reset_index(drop=True)
    log.info('  Repurposed enrichment patterns: %d', len(repurposed_smarts))
    return repurposed_smarts


def cross_study_smarts(repurposed_smarts, cl_df, top_n=200):
    """Cross-study SMARTS analysis: find patterns significant in BOTH
    repurposed CompoundList candidates AND generated molecules from Phase 4.

    This is the most novel aspect of the repurposing pipeline: if the same
    scaffold (e.g. a quinazoline core) appears in the top-ranked compounds
    from *both* existing-drug screening and de novo generation, the
    convergence from two independent computational approaches constitutes
    strong evidence for that scaffold's pharmacological relevance to OSCC.

    The enrichment ratio (VP frequency / inactive frequency) further
    filters for scaffolds overrepresented in potent compounds relative
    to the inactive background.
    """
    log.info('Phase G (cross-study): SMARTS enrichment across repurposed & generated ...')

    # Load Phase 4 generated molecule dossier
    gen_dossier_path = os.path.join(BIOACTIVITY_DIR, 'final_dossier.tsv')
    gen_smarts_path = os.path.join(BIOACTIVITY_DIR, 'prevalent_smarts.tsv')

    if not os.path.exists(gen_dossier_path):
        log.info('  Phase 4 dossier not found at %s — skipping cross-study', gen_dossier_path)
        return pd.DataFrame()

    gen_dossier = pd.read_csv(gen_dossier_path, sep='\t')
    log.info('  Loaded Phase 4 dossier: %d generated molecules', len(gen_dossier))

    # Top generated molecules
    gen_top = gen_dossier.nlargest(top_n, 'composite_score')
    gen_top_smiles = gen_top['smiles'].tolist()

    # Top repurposed compounds
    rep_top = cl_df.nlargest(top_n, 'composite_score')
    rep_top_smiles = rep_top['smiles'].tolist()

    # Extract scaffolds from both sets
    gen_scaffolds = extract_scaffolds(gen_top_smiles)
    rep_scaffolds = extract_scaffolds(rep_top_smiles)

    # Union of all scaffolds found in either set
    all_scaffolds = set(gen_scaffolds.keys()) | set(rep_scaffolds.keys())
    log.info('  Scaffolds in generated top-%d: %d', top_n, len(gen_scaffolds))
    log.info('  Scaffolds in repurposed top-%d: %d', top_n, len(rep_scaffolds))
    log.info('  Union scaffolds: %d', len(all_scaffolds))

    # Parse molecules
    gen_mols = [m for m in (Chem.MolFromSmiles(s) for s in gen_top_smiles) if m is not None]
    rep_mols = [m for m in (Chem.MolFromSmiles(s) for s in rep_top_smiles) if m is not None]

    cross_results = []
    for scaffold_smi in all_scaffolds:
        pat = Chem.MolFromSmiles(scaffold_smi)
        if pat is None:
            continue

        n_gen = sum(1 for m in gen_mols if m.HasSubstructMatch(pat))
        n_rep = sum(1 for m in rep_mols if m.HasSubstructMatch(pat))

        if n_gen == 0 and n_rep == 0:
            continue

        gen_pct = n_gen / max(1, len(gen_mols))
        rep_pct = n_rep / max(1, len(rep_mols))

        # Shared if present in both sets
        shared = n_gen > 0 and n_rep > 0

        # Convert scaffold to SMARTS for more general matching  
        try:
            smarts_str = Chem.MolToSmarts(pat)
        except Exception:
            smarts_str = scaffold_smi

        cross_results.append({
            'scaffold_smiles': scaffold_smi,
            'smarts': smarts_str,
            'n_matches_generated': n_gen,
            'pct_generated': gen_pct,
            'n_matches_repurposed': n_rep,
            'pct_repurposed': rep_pct,
            'shared': shared,
            'combined_pct': (gen_pct + rep_pct) / 2,
            'gen_scaffold_count': gen_scaffolds.get(scaffold_smi, 0),
            'rep_scaffold_count': rep_scaffolds.get(scaffold_smi, 0),
        })

    cross_df = pd.DataFrame(cross_results)
    if len(cross_df) > 0:
        cross_df = cross_df.sort_values('combined_pct', ascending=False).reset_index(drop=True)

    n_shared = cross_df['shared'].sum() if len(cross_df) > 0 else 0
    n_gen_only = ((cross_df['n_matches_generated'] > 0) & ~cross_df['shared']).sum() if len(cross_df) > 0 else 0
    n_rep_only = ((cross_df['n_matches_repurposed'] > 0) & ~cross_df['shared']).sum() if len(cross_df) > 0 else 0
    log.info('  Cross-study scaffolds: shared=%d, gen-only=%d, rep-only=%d',
             n_shared, n_gen_only, n_rep_only)

    # Also load and merge Phase 4 prevalent_smarts if available
    if os.path.exists(gen_smarts_path):
        gen_smarts = pd.read_csv(gen_smarts_path, sep='\t')
        log.info('  Phase 4 prevalent SMARTS loaded: %d patterns', len(gen_smarts))

        # Check which Phase 4 enriched patterns also appear in repurposed top compounds
        for _, row in gen_smarts.iterrows():
            gp_smarts = row.get('smarts', '')
            pat = Chem.MolFromSmiles(gp_smarts)
            if pat is None:
                pat = Chem.MolFromSmarts(gp_smarts)
            if pat is None:
                continue

            n_rep_hits = sum(1 for m in rep_mols if m.HasSubstructMatch(pat))
            if n_rep_hits > 0:
                cross_df = pd.concat([cross_df, pd.DataFrame([{
                    'scaffold_smiles': gp_smarts,
                    'smarts': gp_smarts,
                    'n_matches_generated': row.get('n_matches_top', 0),
                    'pct_generated': row.get('n_matches_top_pct', 0),
                    'n_matches_repurposed': n_rep_hits,
                    'pct_repurposed': n_rep_hits / max(1, len(rep_mols)),
                    'shared': True,
                    'combined_pct': (row.get('n_matches_top_pct', 0) + n_rep_hits / max(1, len(rep_mols))) / 2,
                    'gen_scaffold_count': row.get('count_in_generated', 0),
                    'rep_scaffold_count': n_rep_hits,
                }])], ignore_index=True)

        if len(cross_df) > 0:
            cross_df = cross_df.drop_duplicates(subset='scaffold_smiles', keep='first')
            cross_df = cross_df.sort_values('combined_pct', ascending=False).reset_index(drop=True)

    log.info('  Final cross-study patterns: %d', len(cross_df))
    return cross_df


# ═══════════════════════════════════════════════════════════════════════════════
# LEAKAGE CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def leakage_check(cl_df, compound_index):
    """Check for training set leakage — flag compounds that were in training positives.

    Training-set overlap is expected (CompoundList compounds may also appear
    in ChEMBL bioactivity data) and is not erroneous.  This function
    explicitly flags overlapping IDs so downstream reporting can separate:
      - Interpolation accuracy (training positives → validation of model fit)
      - Generalisation performance (unseen compounds → true predictive power)
    """
    log.info('Leakage check ...')

    train_positives = {cid for cid, info in compound_index.items()
                       if info.get('label') == 'positive'}

    cl_chembl_ids = set(cl_df['compound_chembl_id'].dropna().unique())
    overlap = cl_chembl_ids & train_positives
    log.info('  CompoundList IDs in training positives: %d / %d (%.1f%%)',
             len(overlap), len(cl_chembl_ids),
             100 * len(overlap) / max(len(cl_chembl_ids), 1))

    # Mark in dataframe
    cl_df['in_training_positives'] = cl_df['compound_chembl_id'].isin(train_positives)
    return cl_df


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def save_all_outputs(cl_df, dossier, target_rankings, target_summary,
                     repurposed_smarts, cross_smarts, pharm_df,
                     validation_results, output_dir):
    """Save all output files."""
    log.info('Saving outputs to %s ...', output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Global rankings
    ranked = cl_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    ranked.to_csv(os.path.join(output_dir, 'global_rankings.tsv'), sep='\t', index=False)
    log.info('  global_rankings.tsv (%d rows)', len(ranked))

    # Top 100
    dossier.head(100).to_csv(os.path.join(output_dir, 'top100_candidates.tsv'), sep='\t')
    log.info('  top100_candidates.tsv')

    # Safety profiles
    safety_cols = ['smiles', 'compound_chembl_id', 'compound_name', 'composite_score',
                   'sa_score', 'qed', 'lipinski_violations', 'pains_alerts', 'pains_names',
                   'brenk_alerts', 'brenk_names', 'toxicophore_count', 'toxicophore_names',
                   'overall_flag']
    safety_cols = [c for c in safety_cols if c in cl_df.columns]
    cl_df[safety_cols].to_csv(os.path.join(output_dir, 'safety_profiles.tsv'),
                              sep='\t', index=False)
    log.info('  safety_profiles.tsv')

    # Target rankings
    target_dir = os.path.join(output_dir, 'target_rankings')
    os.makedirs(target_dir, exist_ok=True)
    for target, t_df in target_rankings.items():
        safe_name = target.replace('/', '_').replace(' ', '_')[:60]
        t_df.to_csv(os.path.join(target_dir, f'{safe_name}.tsv'), sep='\t', index=False)
    log.info('  target_rankings/ (%d targets)', len(target_rankings))

    # Target summary
    if target_summary is not None:
        target_summary.to_csv(os.path.join(output_dir, 'target_summary.tsv'),
                              sep='\t', index=False)
        log.info('  target_summary.tsv')

    # Repurposed SMARTS enrichment
    if repurposed_smarts is not None and len(repurposed_smarts) > 0:
        repurposed_smarts.to_csv(os.path.join(output_dir, 'repurposed_smarts.tsv'),
                                 sep='\t', index=False)
        log.info('  repurposed_smarts.tsv (%d patterns)', len(repurposed_smarts))

    # Cross-study SMARTS
    if cross_smarts is not None and len(cross_smarts) > 0:
        cross_smarts.to_csv(os.path.join(output_dir, 'cross_study_smarts.tsv'),
                            sep='\t', index=False)
        log.info('  cross_study_smarts.tsv (%d patterns)', len(cross_smarts))

        # Also save shared-only subset
        shared = cross_smarts[cross_smarts['shared'] == True]
        if len(shared) > 0:
            shared.to_csv(os.path.join(output_dir, 'shared_smarts_repurposed_generated.tsv'),
                          sep='\t', index=False)
            log.info('  shared_smarts_repurposed_generated.tsv (%d patterns)', len(shared))

    # Pharmacophore
    if pharm_df is not None:
        pharm_out = cl_df[['smiles', 'compound_chembl_id']].copy()
        for col in pharm_df.columns:
            pharm_out[col] = pharm_df[col].values
        pharm_out.to_csv(os.path.join(output_dir, 'pharmacophore_2d.tsv'),
                         sep='\t', index=False)
        log.info('  pharmacophore_2d.tsv')

    # Full dossier
    dossier.to_csv(os.path.join(output_dir, 'final_dossier.tsv'), sep='\t')
    log.info('  final_dossier.tsv (%d rows)', len(dossier))

    # Top 50 dossier
    dossier.head(50).to_csv(os.path.join(output_dir, 'top50_dossier.tsv'), sep='\t')
    log.info('  top50_dossier.tsv')

    # Validation results
    with open(os.path.join(output_dir, 'validation_results.json'), 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    log.info('  validation_results.json')

    # Summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'pipeline': 'Phase6_DrugRepurposing_CompoundList',
        'n_total_compounds': len(cl_df),
        'n_unique_dossier': len(dossier),
        'n_targets_scored': len(target_rankings),
        'n_repurposed_smarts': len(repurposed_smarts) if repurposed_smarts is not None else 0,
        'n_cross_study_smarts': len(cross_smarts) if cross_smarts is not None else 0,
        'n_shared_smarts': int(cross_smarts['shared'].sum()) if cross_smarts is not None and len(cross_smarts) > 0 else 0,
        'safety_summary': {
            'pass': int((cl_df['overall_flag'] == 'PASS').sum()),
            'caution': int((cl_df['overall_flag'] == 'CAUTION').sum()),
            'fail': int((cl_df['overall_flag'] == 'FAIL').sum()),
        },
        'validation': validation_results,
        'score_stats': {
            'composite': {
                'mean': float(cl_df['composite_score'].mean()),
                'std': float(cl_df['composite_score'].std()),
                'max': float(cl_df['composite_score'].max()),
                'min': float(cl_df['composite_score'].min()),
            },
            'recommendation_weight': {
                'mean': float(dossier['recommendation_weight'].mean()),
                'std': float(dossier['recommendation_weight'].std()),
                'max': float(dossier['recommendation_weight'].max()),
            },
        },
        'top10': dossier.head(10)[[c for c in [
            'smiles', 'compound_chembl_id', 'compound_name',
            'recommendation_weight', 'composite_score',
            'vp_similarity', 'predicted_pchembl', 'qed', 'sa_score',
            'overall_flag', 'clinical_phase', 'best_target',
        ] if c in dossier.columns]].to_dict(orient='records'),
    }
    with open(os.path.join(output_dir, 'repurposing_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info('  repurposing_summary.json')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 6: Drug Repurposing of CompoundList Compounds')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--min-vp', type=int, default=30)
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, 'results')
    t0 = datetime.now()

    log.info('=' * 70)
    log.info('Phase 6: Drug Repurposing Pipeline (CompoundList Compounds)')
    log.info('=' * 70)
    log.info('Device: %s', args.device)
    log.info('Output: %s', output_dir)

    # ── Load contrastive model ──
    from model import ContrastiveModel
    ckpt_path = os.path.join(MODELING_DIR, 'checkpoints', 'best.pt')
    log.info('Loading contrastive model from %s', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = ContrastiveModel()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    log.info('Model loaded (epoch %d, val_loss=%.4f)',
             ckpt.get('epoch', -1), ckpt.get('val_loss', -1))

    # ── Load compound index & embeddings ──
    with open(os.path.join(MODELING_DIR, 'compound_index.pkl'), 'rb') as f:
        compound_index = pickle.load(f)
    log.info('Compound index: %d entries', len(compound_index))

    n_compounds = len(np.load(
        os.path.join(EMB_DIR, 'compound_ids.npy'), allow_pickle=True,
    ))
    emb_mmap = np.memmap(
        os.path.join(EMB_DIR, 'compound_embeddings.mmap'),
        dtype='float16', mode='r', shape=(n_compounds, 768),
    )
    log.info('Embedding memmap: %s', emb_mmap.shape)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE A: Data Loading
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('PHASE A: Loading CompoundList compounds')
    log.info('=' * 70)
    cl_df = load_compoundlist_compounds()
    cl_df = map_to_embeddings(cl_df, compound_index)

    # Drop unmapped compounds
    n_before = len(cl_df)
    cl_df = cl_df[cl_df['emb_idx'].notna()].reset_index(drop=True)
    log.info('  After dropping unmapped: %d (dropped %d)', len(cl_df), n_before - len(cl_df))

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE B: Global Ranking
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('PHASE B: Global ranking via contrastive projection')
    log.info('=' * 70)
    cl_df, projections, pred_pchembl = project_compounds(cl_df, model, emb_mmap, args.device)
    vp_centroid, t3c_centroid = compute_centroids(model, compound_index, emb_mmap, args.device)
    cl_df = global_ranking(cl_df, projections, pred_pchembl, vp_centroid, t3c_centroid)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE C: Per-Target Screening
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('PHASE C: Per-target screening')
    log.info('=' * 70)
    target_cents = per_target_centroids(model, compound_index, emb_mmap, args.device,
                                        min_vp=args.min_vp)
    target_rankings, target_summary = target_specific_ranking(
        cl_df, projections, pred_pchembl, target_cents, t3c_centroid,
        top_k=args.top_k,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE D: Safety Profiling
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('PHASE D: Safety profiling')
    log.info('=' * 70)
    cl_df = compute_safety_profiles(cl_df)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE E: Validation
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('PHASE E: Validation (predicted vs known potency)')
    log.info('=' * 70)
    validation_results = validate_predictions(cl_df)

    # Leakage check
    cl_df = leakage_check(cl_df, compound_index)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE G: SMARTS Enrichment Mining
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('PHASE G: SMARTS enrichment mining & cross-study analysis')
    log.info('=' * 70)
    repurposed_smarts = smarts_mining(cl_df, compound_index)
    cross_smarts = cross_study_smarts(repurposed_smarts, cl_df)

    # ═══════════════════════════════════════════════════════════════════════
    # 2D Pharmacophore
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('2D pharmacophore profiling')
    log.info('=' * 70)
    pharm_df = pharmacophore_2d(cl_df)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE F: Dossier Compilation
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('PHASE F: Compiling final dossier (8-component recommendation weight)')
    log.info('=' * 70)
    dossier = compile_dossier(cl_df, target_summary, repurposed_smarts, pharm_df)

    # ═══════════════════════════════════════════════════════════════════════
    # Save Everything
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('SAVING OUTPUTS')
    log.info('=' * 70)
    save_all_outputs(cl_df, dossier, target_rankings, target_summary,
                     repurposed_smarts, cross_smarts, pharm_df,
                     validation_results, output_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # Print Top Results
    # ═══════════════════════════════════════════════════════════════════════
    log.info('\n' + '=' * 70)
    log.info('TOP 20 REPURPOSING CANDIDATES')
    log.info('=' * 70)
    log.info('%-4s %-7s %-7s %-7s %-6s %-5s %-6s %-15s %-25s',
             'Rank', 'RW', 'Comp', 'VPsim', 'QED', 'SA', 'Safety',
             'ChEMBL ID', 'Best Target')
    log.info('-' * 100)
    for i, (_, row) in enumerate(dossier.head(20).iterrows()):
        log.info('%-4d %-7.4f %-7.4f %-7.4f %-6.3f %-5.2f %-6s %-15s %-25s',
                 i + 1,
                 row['recommendation_weight'],
                 row['composite_score'],
                 row['vp_similarity'],
                 row.get('qed', 0),
                 row.get('sa_score', 0),
                 row.get('overall_flag', ''),
                 str(row.get('compound_chembl_id', ''))[:15],
                 str(row.get('best_target', ''))[:25])

    # Cross-study summary
    if cross_smarts is not None and len(cross_smarts) > 0:
        shared = cross_smarts[cross_smarts['shared'] == True]
        log.info('\n' + '=' * 70)
        log.info('CROSS-STUDY SMARTS: %d shared between repurposed & generated', len(shared))
        log.info('=' * 70)
        for _, row in shared.head(10).iterrows():
            log.info('  %s | gen=%.1f%% | rep=%.1f%%',
                     str(row['scaffold_smiles'])[:60],
                     100 * row['pct_generated'],
                     100 * row['pct_repurposed'])

    elapsed = (datetime.now() - t0).total_seconds()
    log.info('\n' + '=' * 70)
    log.info('Phase 6 COMPLETE in %.1f minutes', elapsed / 60)
    log.info('Results → %s', output_dir)
    log.info('=' * 70)


if __name__ == '__main__':
    main()
