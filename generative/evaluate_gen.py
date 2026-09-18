#!/usr/bin/env python3
"""
Phase 3 — Unified Evaluation Framework for Generative Models.

10-metric evaluation battery applied consistently to all methods:
  1. Validity      — % of generated SELFIES that decode to valid SMILES
  2. Uniqueness    — % unique among valid molecules
  3. Novelty       — % not in training set
  4. OSCC Potency  — cosine similarity to VP centroid in 128-d space
  5. pChEMBL pred  — predicted potency from contrastive model
  6. QED           — quantitative estimate of drug-likeness
  7. SA Score      — synthetic accessibility (1=easy, 10=hard)
  8. Diversity     — average pairwise Tanimoto distance
  9. Scaffold Nov. — % with scaffolds unseen in training
 10. FCD           — Fréchet ChemNet Distance to training set

Key fix baked in (from UPDATE_1):
  - MolDeBERTa with use_fast=False (was ChemBERTa → potency=0)

Usage:
    from generative.evaluate_gen import evaluate_molecules
    results = evaluate_molecules(smiles_list, method_name='VAE')
"""
import os
import sys
import json
import logging
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold

GENESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELING_DIR = os.path.join(GENESIS_DIR, 'modeling')
GEN_DIR = os.path.join(GENESIS_DIR, 'generative', 'processed')
RESULTS_DIR = os.path.join(GENESIS_DIR, 'generative', 'results')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ── SA Score (Ertl & Schuffenhauer, JCICS 2009) ────────────────────

def _sa_score(mol):
    """Estimate synthetic accessibility score."""
    try:
        from rdkit.Chem import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer
        return sascorer.calculateScore(mol)
    except (ImportError, Exception):
        n_rings = Descriptors.RingCount(mol)
        n_stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        mw = Descriptors.MolWt(mol)
        score = 1.0 + 0.5 * n_rings + 0.3 * len(n_stereo) + (mw / 500.0)
        return min(score, 10.0)


def get_murcko_scaffold(smi):
    """Get Murcko scaffold from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def compute_fingerprints(smiles_list, radius=2, nbits=2048):
    """Compute Morgan fingerprints for a list of SMILES."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(fp)
    return fps


def pairwise_tanimoto_distances(fps, n_sample=5000):
    """Compute average pairwise Tanimoto distance from a sample of fingerprints."""
    if len(fps) <= 1:
        return 0.0
    if len(fps) > n_sample:
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(fps), size=n_sample, replace=False)
        fps = [fps[i] for i in idxs]

    n = len(fps)
    total_dist = 0.0
    count = 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total_dist += sum(1.0 - s for s in sims)
        count += len(sims)

    return total_dist / count if count > 0 else 0.0


def _load_training_smiles():
    """Load training set SMILES for novelty comparison."""
    meta_path = os.path.join(GEN_DIR, 'compound_metadata.tsv')
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path, sep='\t')
        train_smiles = set(meta[meta['split'] == 'train']['smiles'].dropna().values)
        return train_smiles
    return set()


def _load_training_scaffolds():
    """Load training set scaffolds for scaffold novelty."""
    meta_path = os.path.join(GEN_DIR, 'compound_metadata.tsv')
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path, sep='\t')
        train_smiles = meta[meta['split'] == 'train']['smiles'].dropna().values
        scaffolds = set()
        for smi in train_smiles:
            scaf = get_murcko_scaffold(smi)
            if scaf:
                scaffolds.add(scaf)
        return scaffolds
    return set()


def _load_centroid():
    """Load 128-d VP centroid for OSCC potency scoring."""
    centroid_path = os.path.join(GENESIS_DIR, 'generative', 'processed', 'vp_centroid.json')
    with open(centroid_path) as f:
        data = json.load(f)
    return np.array(data['oscc_vp_centroid'], dtype=np.float32)


def _predict_potency(smiles_list, device='cpu'):
    """Predict 128-d projections and pChEMBL for generated SMILES.

    Uses MolDeBERTa → ProjectionHead → PotencyRegressor pipeline.
    """
    import torch

    sys.path.insert(0, MODELING_DIR)
    from model import ContrastiveModel

    ckpt_path = os.path.join(MODELING_DIR, 'checkpoints', 'best.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = ContrastiveModel()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)

    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(
            'SaeedLab/MolDeBERTa-base-123M-mtr', use_fast=False)
        embedder = AutoModel.from_pretrained(
            'SaeedLab/MolDeBERTa-base-123M-mtr')
        embedder.eval().to(device)
    except Exception:
        log.warning('MolDeBERTa not available, returning zero projections')
        return (np.zeros((len(smiles_list), 128), dtype=np.float32),
                np.zeros(len(smiles_list), dtype=np.float32))

    projections = []
    pchembl_preds = []
    batch_size = 64

    for start in range(0, len(smiles_list), batch_size):
        batch_smi = smiles_list[start:start + batch_size]
        try:
            inputs = tokenizer(batch_smi, return_tensors='pt', padding=True,
                               truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = embedder(**inputs)
                embs = outputs.last_hidden_state[:, 0, :]
                z, pchembl = model(embs)
            projections.append(z.cpu().numpy())
            pchembl_preds.append(pchembl.cpu().numpy())
        except Exception:
            for smi in batch_smi:
                try:
                    inp = tokenizer([smi], return_tensors='pt', padding=True,
                                    truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        out = embedder(**inp)
                        emb = out.last_hidden_state[:, 0, :]
                        z_i, p_i = model(emb)
                    projections.append(z_i.cpu().numpy())
                    pchembl_preds.append(p_i.cpu().numpy())
                except Exception:
                    projections.append(np.zeros((1, 128), dtype=np.float32))
                    pchembl_preds.append(np.zeros(1, dtype=np.float32))

    projections = np.concatenate(projections, axis=0)
    pchembl_preds = np.concatenate(pchembl_preds, axis=0)
    return projections, pchembl_preds


def evaluate_molecules(smiles_list, method_name='unknown', device='cpu',
                       compute_fcd=False, compute_potency=True):
    """Run full 10-metric evaluation on generated SMILES."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log.info('=== Evaluating %s: %d generated molecules ===', method_name, len(smiles_list))

    n_total = len(smiles_list)
    if n_total == 0:
        log.warning('No molecules to evaluate')
        return {}

    # ── 1. Validity ──────────────────────────────────────────────────
    valid_smiles = []
    valid_mols = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon = Chem.MolToSmiles(mol, canonical=True)
                valid_smiles.append(canon)
                valid_mols.append(mol)
        except Exception:
            pass

    validity = len(valid_smiles) / n_total
    log.info('1. Validity: %.1f%% (%d/%d)', 100 * validity, len(valid_smiles), n_total)

    if len(valid_smiles) == 0:
        log.warning('No valid molecules — aborting evaluation')
        return {'validity': 0.0, 'n_generated': n_total, 'method': method_name}

    # ── 2. Uniqueness ────────────────────────────────────────────────
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / len(valid_smiles)
    log.info('2. Uniqueness: %.1f%% (%d/%d valid)',
             100 * uniqueness, len(unique_smiles), len(valid_smiles))

    # ── 3. Novelty ───────────────────────────────────────────────────
    train_smiles = _load_training_smiles()
    novel_smiles = [s for s in unique_smiles if s not in train_smiles]
    novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0
    log.info('3. Novelty: %.1f%% (%d/%d unique)',
             100 * novelty, len(novel_smiles), len(unique_smiles))

    # ── 6. QED ───────────────────────────────────────────────────────
    qed_scores = []
    for mol in valid_mols:
        try:
            qed_scores.append(QED.qed(mol))
        except Exception:
            qed_scores.append(0.0)
    qed_mean = float(np.mean(qed_scores))
    log.info('6. QED: mean=%.3f (+/-%.3f)', qed_mean, np.std(qed_scores))

    # ── 7. SA Score ──────────────────────────────────────────────────
    sa_scores = [_sa_score(mol) for mol in valid_mols]
    sa_mean = float(np.mean(sa_scores))
    log.info('7. SA Score: mean=%.2f (+/-%.2f) [lower=easier]', sa_mean, np.std(sa_scores))

    # ── 8. Diversity ─────────────────────────────────────────────────
    fps = compute_fingerprints(valid_smiles)
    diversity = pairwise_tanimoto_distances(fps)
    log.info('8. Diversity: %.3f avg Tanimoto distance', diversity)

    # ── 9. Scaffold Novelty ──────────────────────────────────────────
    train_scaffolds = _load_training_scaffolds()
    gen_scaffolds = [get_murcko_scaffold(s) for s in unique_smiles]
    gen_scaffolds_valid = [s for s in gen_scaffolds if s is not None]
    scaffold_novel = [s for s in gen_scaffolds_valid if s not in train_scaffolds]
    scaffold_novelty = len(scaffold_novel) / len(gen_scaffolds_valid) if gen_scaffolds_valid else 0
    log.info('9. Scaffold Novelty: %.1f%% (%d/%d)',
             100 * scaffold_novelty, len(scaffold_novel), len(gen_scaffolds_valid))

    # ── 4 & 5. OSCC Potency + pChEMBL ───────────────────────────────
    oscc_potency_mean = 0.0
    pchembl_mean = 0.0
    vp_fraction = 0.0
    if compute_potency:
        try:
            centroid = _load_centroid()
            projections, pchembl_preds = _predict_potency(valid_smiles, device=device)
            oscc_scores = projections @ centroid
            oscc_potency_mean = float(np.mean(oscc_scores))
            pchembl_mean = float(np.mean(pchembl_preds))
            vp_fraction = float(np.mean(oscc_scores > 0.7))
            log.info('4. OSCC Potency: mean=%.3f, VP-like fraction=%.1f%%',
                     oscc_potency_mean, 100 * vp_fraction)
            log.info('5. Predicted pChEMBL: mean=%.2f (+/-%.2f)',
                     pchembl_mean, np.std(pchembl_preds))
        except Exception as e:
            log.warning('Potency prediction failed: %s', e)

    # ── 10. FCD ──────────────────────────────────────────────────────
    fcd_score = None
    if compute_fcd:
        try:
            from fcd import get_fcd
            fcd_score = get_fcd(list(train_smiles)[:10000], valid_smiles[:10000])
            log.info('10. FCD: %.3f', fcd_score)
        except ImportError:
            log.warning('FCD package not installed, skipping')
        except Exception as e:
            log.warning('FCD computation failed: %s', e)

    # ── Compile results ──────────────────────────────────────────────
    results = {
        'method': method_name,
        'n_generated': n_total,
        'n_valid': len(valid_smiles),
        'n_unique': len(unique_smiles),
        'n_novel': len(novel_smiles),
        'metrics': {
            'validity': float(validity),
            'uniqueness': float(uniqueness),
            'novelty': float(novelty),
            'oscc_potency': float(oscc_potency_mean),
            'pchembl_pred': float(pchembl_mean),
            'vp_fraction': float(vp_fraction),
            'qed': float(qed_mean),
            'sa_score': float(sa_mean),
            'diversity': float(diversity),
            'scaffold_novelty': float(scaffold_novelty),
        },
        'distributions': {
            'qed': {'mean': float(np.mean(qed_scores)), 'std': float(np.std(qed_scores)),
                    'p25': float(np.percentile(qed_scores, 25)),
                    'p75': float(np.percentile(qed_scores, 75))},
            'sa': {'mean': float(np.mean(sa_scores)), 'std': float(np.std(sa_scores))},
        },
    }
    if fcd_score is not None:
        results['metrics']['fcd'] = float(fcd_score)

    out_path = os.path.join(RESULTS_DIR, f'eval_{method_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info('Results -> %s', out_path)

    return results


def compare_methods(method_names=None):
    """Compare evaluation results across methods."""
    if method_names is None:
        if not os.path.exists(RESULTS_DIR):
            log.warning('Results directory not found')
            return None
        files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('eval_') and f.endswith('.json')]
        method_names = [f[5:-5] for f in files]

    rows = []
    for name in method_names:
        path = os.path.join(RESULTS_DIR, f'eval_{name}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            row = {'method': name}
            row.update(data['metrics'])
            rows.append(row)

    if not rows:
        log.warning('No evaluation results found')
        return None

    df = pd.DataFrame(rows).set_index('method')
    print('\n=== Method Comparison ===')
    print(df.to_string(float_format='%.3f'))
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true', help='Compare all methods')
    args = parser.parse_args()

    if args.compare:
        compare_methods()
    else:
        print('Usage: python evaluate_gen.py --compare')
        print('Or: from generative.evaluate_gen import evaluate_molecules')
