#!/usr/bin/env python3
"""Multi-seed Flat SupCon training + evaluation, mirroring modeling/multiseed_train_eval.py
for PGCL, to give a fair 3-seed comparison basis against PGCL's 3-seed mean.

run_supcon_baseline.py only supports a single --seed and writes to fixed shared
paths (results/flat_supcon/{checkpoints/best.pt,eval_results.json}) that would be
clobbered by each subsequent seed. This driver preserves each seed's checkpoint
and eval into results/flat_supcon/multiseed/ before the next seed overwrites them.
"""
import sys
import json
import shutil
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_supcon_baseline import train_flat_supcon, evaluate_flat_supcon, RESULTS_DIR

MULTISEED_DIR = RESULTS_DIR / 'multiseed'
SEEDS = [42, 123, 456]


def main(device='mps'):
    (MULTISEED_DIR / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (MULTISEED_DIR / 'logs').mkdir(parents=True, exist_ok=True)

    all_results = {}
    for seed in SEEDS:
        ckpt_dst = MULTISEED_DIR / 'checkpoints' / f'best_seed{seed}.pt'
        eval_dst = MULTISEED_DIR / 'logs' / f'eval_seed{seed}.json'

        if ckpt_dst.exists() and eval_dst.exists():
            print(f'Seed {seed}: already done, loading cached eval -> {eval_dst}')
            with open(eval_dst) as f:
                all_results[seed] = json.load(f)
            continue

        print(f'\n{"="*60}\n  Flat SupCon: seed={seed}\n{"="*60}')
        ok = train_flat_supcon(device, seed=seed, epochs=40)
        if not ok:
            print(f'Seed {seed} FAILED — skipping')
            continue

        results = evaluate_flat_supcon(device)
        if results is None:
            print(f'Seed {seed} evaluation FAILED — skipping')
            continue

        # Preserve this seed's checkpoint + eval before the next seed overwrites
        # the shared results/flat_supcon/{checkpoints/best.pt,eval_results.json}.
        shutil.copy2(RESULTS_DIR / 'checkpoints' / 'best.pt', ckpt_dst)
        with open(eval_dst, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Seed {seed} preserved -> {ckpt_dst}, {eval_dst}')
        all_results[seed] = results

    # Aggregate the 5 headline metrics across all completed seeds.
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

    summary = {
        k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'values': v, 'n_seeds': len(v)}
        for k, v in metrics.items()
    }
    summary_path = MULTISEED_DIR / 'logs' / 'multiseed_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"="*60}\n  Flat SupCon multi-seed summary ({len(all_results)}/{len(SEEDS)} seeds)\n{"="*60}')
    for k, v in summary.items():
        print(f'  {k:<25s} {v["mean"]:.4f} ± {v["std"]:.4f}  (n={v["n_seeds"]})')
    print(f'\nSummary -> {summary_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Multi-seed Flat SupCon training + evaluation')
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    main(args.device)
