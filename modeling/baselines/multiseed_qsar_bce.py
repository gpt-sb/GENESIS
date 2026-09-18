#!/usr/bin/env python3
"""Multi-seed QSAR (RF/XGB/LR) and BCE classifier baselines, mirroring
multiseed_flat_supcon.py, for a fair 3-seed comparison basis against PGCL's
3-seed mean. Both write to seed-specific paths only -- neither touches the
shared modeling/checkpoints/ path (QSAR has no torch checkpoint at all; BCE
writes to its own dedicated results/bce_classifier/checkpoints/best.pt), so
no PGCL-protection guard is needed here.
"""
import sys
import json
import shutil
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

SEEDS = [42, 123, 456]


def run_qsar_multiseed():
    from run_qsar_baselines import run_rf, run_xgb, run_logistic, RESULTS_DIR

    multiseed_dir = RESULTS_DIR / 'multiseed'
    (multiseed_dir / 'logs').mkdir(parents=True, exist_ok=True)

    print('Loading QSAR data artifacts (once, shared across seeds)...')
    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)
    df_pos = pd.read_csv(MODELING_DIR / 'positives.tsv', sep='\t', low_memory=False,
                          usecols=['compound_chembl_id', 'smiles'])
    smiles_lookup = {str(r['compound_chembl_id']): str(r['smiles'])
                     for _, r in df_pos.iterrows()
                     if pd.notna(r['smiles']) and str(r['compound_chembl_id']) in compound_index}

    all_results = {}
    for seed in SEEDS:
        out_path = multiseed_dir / 'logs' / f'eval_seed{seed}.json'
        if out_path.exists():
            print(f'QSAR seed {seed}: already done, loading cached -> {out_path}')
            with open(out_path) as f:
                all_results[seed] = json.load(f)
            continue

        print(f'\n=== QSAR seed={seed} ===')
        res = {
            'rf':  run_rf(compound_index, smiles_lookup, splits, seed=seed),
            'xgb': run_xgb(compound_index, smiles_lookup, splits, seed=seed),
            'lr':  run_logistic(compound_index, smiles_lookup, splits, seed=seed),
        }
        with open(out_path, 'w') as f:
            json.dump(res, f, indent=2)
        all_results[seed] = res
        print(f'QSAR seed {seed} saved -> {out_path}')

    metrics = {}
    for seed, res in all_results.items():
        for method in ('rf', 'xgb', 'lr'):
            for key in ('pchembl_spearman_rho', 'binary_auroc', 'vp_recall_100'):
                v = res.get(method, {}).get(key)
                if v is not None:
                    metrics.setdefault(f'{method}_{key}', []).append(v)

    summary = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'values': v, 'n_seeds': len(v)}
               for k, v in metrics.items()}
    summary_path = multiseed_dir / 'logs' / 'multiseed_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nQSAR multi-seed summary ({len(all_results)}/{len(SEEDS)} seeds) -> {summary_path}')
    for k, v in summary.items():
        print(f'  {k:<28s} {v["mean"]:.4f} ± {v["std"]:.4f}')


def run_bce_multiseed(device='mps'):
    from run_bce_baseline import train_bce, evaluate_bce, RESULTS_DIR

    multiseed_dir = RESULTS_DIR / 'multiseed'
    (multiseed_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (multiseed_dir / 'logs').mkdir(parents=True, exist_ok=True)

    all_results = {}
    for seed in SEEDS:
        ckpt_dst = multiseed_dir / 'checkpoints' / f'best_seed{seed}.pt'
        eval_dst = multiseed_dir / 'logs' / f'eval_seed{seed}.json'
        if ckpt_dst.exists() and eval_dst.exists():
            print(f'BCE seed {seed}: already done, loading cached -> {eval_dst}')
            with open(eval_dst) as f:
                all_results[seed] = json.load(f)
            continue

        print(f'\n=== BCE seed={seed} ===')
        ok = train_bce(device, seed=seed, epochs=40)
        if not ok:
            print(f'BCE seed {seed} FAILED — skipping')
            continue
        results = evaluate_bce(device)
        if not results:
            print(f'BCE seed {seed} evaluation FAILED — skipping')
            continue

        shutil.copy2(RESULTS_DIR / 'checkpoints' / 'best.pt', ckpt_dst)
        with open(eval_dst, 'w') as f:
            json.dump(results, f, indent=2)
        all_results[seed] = results
        print(f'BCE seed {seed} preserved -> {ckpt_dst}, {eval_dst}')

    metrics = {}
    for seed, res in all_results.items():
        flat = {
            'vp_recall_100': res['vp_retrieval']['vp_recall_100'],
            'global_spearman_rho': res['potency_ordering']['global_spearman_rho'],
            'cliff_accuracy': res['activity_cliff']['cliff_accuracy'],
            'target_switch_auc_roc': res['target_switch']['target_switch_auc_roc'],
            'ndcg_100': res['potency_graded']['ndcg_100'],
        }
        for k, v in flat.items():
            metrics.setdefault(k, []).append(v)

    summary = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'values': v, 'n_seeds': len(v)}
               for k, v in metrics.items()}
    summary_path = multiseed_dir / 'logs' / 'multiseed_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nBCE multi-seed summary ({len(all_results)}/{len(SEEDS)} seeds) -> {summary_path}')
    for k, v in summary.items():
        print(f'  {k:<25s} {v["mean"]:.4f} ± {v["std"]:.4f}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Multi-seed QSAR + BCE baselines')
    p.add_argument('--device', default='mps')
    p.add_argument('--only', choices=['qsar', 'bce', 'both'], default='both')
    args = p.parse_args()

    if args.only in ('qsar', 'both'):
        run_qsar_multiseed()
    if args.only in ('bce', 'both'):
        run_bce_multiseed(args.device)
