#!/usr/bin/env python3
"""
NMI Baseline A1 — Standard (Flat) SupCon vs. PGCL (Ordinal SupCon).

Trains the same ProjectionHead architecture with flat pair weights (w_ij = 1.0
for all positive pairs) instead of ordinal potency-proximity weights.
This directly tests whether the ordinal extension contributes beyond standard
SupCon (Khosla et al., NeurIPS 2020).

Usage:
    python GENESIS/modeling/baselines/run_supcon_baseline.py --device mps
    python GENESIS/modeling/baselines/run_supcon_baseline.py --skip-train --device mps

Outputs:
    GENESIS/modeling/baselines/results/flat_supcon/
        best.pt            — best checkpoint
        eval_results.json  — all 10 evaluation test results
        comparison.json    — side-by-side vs PGCL baseline
"""
import os
import sys
import json
import subprocess
import argparse
import shutil
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parent
GENESIS_DIR = MODELING_DIR.parent
RESULTS_DIR = SCRIPT_DIR / 'results' / 'flat_supcon'

sys.path.insert(0, str(MODELING_DIR))


# Scope C fresh multiseed PGCL results (seeds 42/123/456, 2026-09-14, retrained
# with the fixed checkpoint-selection logic — best.pt = last completed epoch,
# not min val_loss). These are the targets the flat baseline must be compared
# against.
# cross_pathway_auc/pchembl_mse/plos_ndcg_100/approved_drug_ndcg_50/balanced_accuracy
# are unused by build_comparison_table and have no fresh equivalent — left unchanged.
PGCL_RESULTS = {
    'vp_recall_100':       {'mean': 0.0260, 'std': 0.0005},
    'global_spearman_rho': {'mean': 0.5553, 'std': 0.0028},
    'cliff_accuracy':      {'mean': 0.6638, 'std': 0.0034},
    'target_switch_auc_roc': {'mean': 0.9668, 'std': 0.0012},
    'cross_pathway_auc':   {'mean': 0.860, 'std': 0.025},
    'pchembl_mse':         {'mean': 0.42, 'std': 0.08},
    'plos_ndcg_100':       {'mean': 0.82, 'std': 0.04},
    'approved_drug_ndcg_50': {'mean': 0.78, 'std': 0.06},
    'balanced_accuracy':   {'mean': 0.87, 'std': 0.02},
    'ndcg_100':            {'mean': 0.9410, 'std': 0.0248},
}


def train_flat_supcon(device: str, seed: int = 42, epochs: int = 40):
    """Train flat SupCon model using train.py with --flat-supcon flag."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = RESULTS_DIR / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = RESULTS_DIR / 'train.log'
    python = sys.executable

    # train.py always writes to the SHARED modeling/checkpoints/best.pt regardless
    # of --flat-supcon or which seed is being trained, which would clobber the
    # canonical PGCL model living there. Protect it: back up the canonical
    # SEED-42 checkpoint (not this call's `seed`, which may be 123/456 for
    # multiseed Flat SupCon runs) before training, restore immediately after --
    # even on failure -- so the shared path never stays corrupted or ends up
    # holding the wrong PGCL seed.
    shared_ckpt = MODELING_DIR / 'checkpoints' / 'best.pt'
    pgcl_backup = MODELING_DIR / 'checkpoints' / 'pgcl_best_seed42.pt'
    if shared_ckpt.exists() and not pgcl_backup.exists():
        shutil.copy2(shared_ckpt, pgcl_backup)
        print(f'Backed up canonical PGCL checkpoint -> {pgcl_backup}')

    cmd = [
        python, str(MODELING_DIR / 'train.py'),
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch-size', '512',
        '--batch-pos', '128',
        '--lr', '3e-4',
        '--temperature', '0.07',
        '--proj-dim', '128',
        '--patience', '7',
        '--min-epochs', '25',
        '--device', device,
        '--flat-supcon',          # KEY: flat pair weights (no ordinal weighting)
    ]

    print(f'Training flat SupCon baseline (seed={seed})...')
    print(f'Command: {" ".join(cmd)}')
    print(f'Log: {log_path}')

    with open(log_path, 'w') as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if ret.returncode != 0:
        print(f'ERROR: training failed (exit={ret.returncode}). Check {log_path}')
        if pgcl_backup.exists():
            shutil.copy2(pgcl_backup, shared_ckpt)
            print('Restored canonical PGCL checkpoint after failed run.')
        return False

    # Copy checkpoint from default location to baseline results
    default_ckpt = MODELING_DIR / 'checkpoints' / 'best.pt'
    if default_ckpt.exists():
        shutil.copy(default_ckpt, ckpt_dir / 'best.pt')
        print(f'Checkpoint saved to {ckpt_dir / "best.pt"}')
    else:
        print('WARNING: best.pt not found in default checkpoint dir')

    # CRITICAL: restore the canonical PGCL model into the shared scratch path
    # so any other process reading modeling/checkpoints/best.pt never sees
    # the Flat SupCon weights left behind by this run.
    if pgcl_backup.exists():
        shutil.copy2(pgcl_backup, shared_ckpt)
        print(f'Restored canonical PGCL checkpoint -> {shared_ckpt}')

    return True

    return True


def evaluate_flat_supcon(device: str):
    """Run the full evaluation battery on the flat SupCon checkpoint."""
    import pickle
    import torch
    from model import ContrastiveModel
    from evaluate import (
        load_model, project_ids,
        test_potency_ordering, test_activity_cliff, test_target_switch,
        test_oscc_specificity, test_vp_retrieval, test_potency_graded_ranking,
    )

    ckpt_path = RESULTS_DIR / 'checkpoints' / 'best.pt'
    if not ckpt_path.exists():
        print(f'ERROR: checkpoint not found at {ckpt_path}. Run training first.')
        return None

    INDEX_FILE = MODELING_DIR / 'compound_index.pkl'
    SPLITS_FILE = MODELING_DIR / 'splits.json'
    EMB_MMAP = GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap'
    EMB_IDS = GENESIS_DIR / 'embeddings' / 'compound_ids.npy'

    print('Loading data artifacts...')
    with open(INDEX_FILE, 'rb') as f:
        compound_index = pickle.load(f)
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    n_compounds = len(np.load(EMB_IDS, allow_pickle=True))
    emb_mmap = np.memmap(EMB_MMAP, dtype='float16', mode='r', shape=(n_compounds, 768))

    print(f'Loading flat SupCon model from {ckpt_path}...')
    model, _ = load_model(str(ckpt_path), device=device)

    results = {}

    # Run evaluation battery (same 6 core tests as PGCL)
    print('\n=== Flat SupCon Evaluation Battery ===')
    results['potency_ordering'] = test_potency_ordering(model, splits, compound_index, emb_mmap, device)
    results['activity_cliff']   = test_activity_cliff(model, splits, compound_index, emb_mmap, device)
    results['target_switch']    = test_target_switch(model, splits, compound_index, emb_mmap, device)
    results['oscc_specificity'] = test_oscc_specificity(model, splits, compound_index, emb_mmap, device)
    results['vp_retrieval']     = test_vp_retrieval(model, splits, compound_index, emb_mmap, device)
    results['potency_graded']   = test_potency_graded_ranking(model, splits, compound_index, emb_mmap, device)

    # Save raw results
    out_path = RESULTS_DIR / 'eval_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')

    return results


def build_comparison_table(flat_results: dict) -> dict:
    """Compare flat SupCon vs. PGCL on all key metrics."""
    comparison = {
        'method_a': 'PGCL (Ordinal SupCon) [ours]',
        'method_b': 'Standard SupCon (flat weights) [baseline]',
        'metrics': {}
    }

    metric_map = [
        ('VP Recall@100',         'vp_retrieval',      'vp_recall_100',       'vp_recall_100'),
        ('Potency Spearman ρ',    'potency_ordering',  'global_spearman_rho', 'global_spearman_rho'),
        ('Cliff Pair Accuracy',   'activity_cliff',    'cliff_accuracy',      'cliff_accuracy'),
        ('Target-Switch AUROC',   'target_switch',     'target_switch_auc_roc', 'target_switch_auc_roc'),
        ('NDCG@100 (potency)',    'potency_graded',    'ndcg_100',            'ndcg_100'),
    ]

    for label, section, pgcl_key, flat_key in metric_map:
        pgcl_val = PGCL_RESULTS.get(pgcl_key, {})
        flat_section = flat_results.get(section, {})
        flat_val = flat_section.get(flat_key, None)

        pgcl_mean = pgcl_val.get('mean', None)
        if flat_val is not None and pgcl_mean is not None:
            delta = float(flat_val) - pgcl_mean
            pct_change = 100 * delta / (abs(pgcl_mean) + 1e-12)
        else:
            delta = None
            pct_change = None

        comparison['metrics'][label] = {
            'PGCL_mean': pgcl_mean,
            'PGCL_std':  pgcl_val.get('std', None),
            'FlatSupCon': float(flat_val) if flat_val is not None else None,
            'delta': delta,
            'pct_change': pct_change,
            'PGCL_wins': (delta < 0) if delta is not None else None,
        }

    return comparison


def print_comparison(comparison: dict):
    """Print side-by-side comparison table."""
    print('\n' + '='*80)
    print('NMI BASELINE A1: Standard SupCon vs. PGCL (Ordinal SupCon)')
    print('='*80)
    print(f'{"Metric":<30} {"PGCL (Ours)":>15} {"Flat SupCon":>15} {"Δ":>10} {"PGCL wins?":>12}')
    print('-'*80)

    for label, vals in comparison['metrics'].items():
        pgcl_str = f'{vals["PGCL_mean"]:.3f}±{vals["PGCL_std"]:.3f}' if vals['PGCL_mean'] is not None else 'N/A'
        flat_str = f'{vals["FlatSupCon"]:.3f}' if vals['FlatSupCon'] is not None else 'N/A'
        delta_str = f'{vals["delta"]:+.3f}' if vals['delta'] is not None else 'N/A'
        wins_str = '✓' if vals['PGCL_wins'] else '✗' if vals['PGCL_wins'] is not None else 'N/A'
        print(f'{label:<30} {pgcl_str:>15} {flat_str:>15} {delta_str:>10} {wins_str:>12}')

    wins = sum(1 for v in comparison['metrics'].values() if v.get('PGCL_wins'))
    total = sum(1 for v in comparison['metrics'].values() if v.get('PGCL_wins') is not None)
    print('-'*80)
    print(f'PGCL wins: {wins}/{total} metrics (ordinal weighting contribution)')
    print('='*80)


def main():
    parser = argparse.ArgumentParser(description='NMI Baseline A1: Flat SupCon vs. PGCL')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training, only evaluate existing checkpoint')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        success = train_flat_supcon(args.device, seed=args.seed, epochs=args.epochs)
        if not success:
            sys.exit(1)

    flat_results = evaluate_flat_supcon(args.device)
    if flat_results is None:
        sys.exit(1)

    comparison = build_comparison_table(flat_results)
    print_comparison(comparison)

    comp_path = RESULTS_DIR / 'comparison.json'
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f'\nComparison saved to {comp_path}')


if __name__ == '__main__':
    main()
