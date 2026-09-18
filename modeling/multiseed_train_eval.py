#!/usr/bin/env python3
"""
GENESIS — Phase 5a: Multi-seed training, evaluation, and bootstrap CI.

Trains the contrastive model with seeds {42, 123, 456}, evaluates each,
then computes bootstrap confidence intervals for all key metrics.

Results → GENESIS/modeling/validation/multiseed/

Usage:
    python GENESIS/modeling/multiseed_train_eval.py --device mps
    python GENESIS/modeling/multiseed_train_eval.py --skip-train --device mps  # eval only
"""
import os
import sys
import json
import time
import pickle
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR
GENESIS_DIR = SCRIPT_DIR.parent
VALIDATION_DIR = MODELING_DIR / 'validation' / 'multiseed'

sys.path.insert(0, str(MODELING_DIR))


def run_training(seed, device, epochs=40, min_epochs=25, patience=7):
    """Run training for a specific seed."""
    python = sys.executable
    log_dir = VALIDATION_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'train_seed{seed}.log'

    cmd = [
        python, str(MODELING_DIR / 'train.py'),
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch-size', '512',
        '--batch-pos', '128',
        '--lr', '3e-4',
        '--temperature', '0.07',
        '--proj-dim', '128',
        '--patience', str(patience),
        '--min-epochs', str(min_epochs),
        '--device', device,
    ]

    print(f"\n{'='*60}")
    print(f"  Training seed={seed} ({epochs} epochs, min={min_epochs})")
    print(f"{'='*60}")
    t0 = time.time()

    with open(log_path, 'w') as lf:
        proc = subprocess.run(cmd, cwd=str(GENESIS_DIR.parent),
                              stdout=lf, stderr=subprocess.STDOUT)

    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"  → {status} [{elapsed/60:.1f} min]")

    if proc.returncode == 0:
        # Copy best checkpoint to validation dir
        src = MODELING_DIR / 'checkpoints' / 'best.pt'
        dst_dir = VALIDATION_DIR / 'checkpoints'
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f'best_seed{seed}.pt'
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Checkpoint → {dst}")

        # Copy training history
        hist_src = MODELING_DIR / 'logs' / f'history_seed{seed}.json'
        hist_dst = VALIDATION_DIR / 'logs' / f'history_seed{seed}.json'
        if hist_src.exists():
            shutil.copy2(hist_src, hist_dst)

    return proc.returncode


def run_evaluation(seed, device):
    """Run full evaluation for a specific seed's checkpoint."""
    from evaluate import run_full_evaluation

    ckpt_path = VALIDATION_DIR / 'checkpoints' / f'best_seed{seed}.pt'
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n  Evaluating seed={seed}...")
    results = run_full_evaluation(str(ckpt_path), device=device)

    out_path = VALIDATION_DIR / 'logs' / f'evaluation_seed{seed}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  → {out_path}")
    return results


def bootstrap_evaluation(seed, device, n_bootstrap=1000):
    """Bootstrap CI for evaluation metrics."""
    from evaluate import load_model, project_ids

    ckpt_path = VALIDATION_DIR / 'checkpoints' / f'best_seed{seed}.pt'
    if not ckpt_path.exists():
        return None

    print(f"\n  Bootstrap CI for seed={seed} ({n_bootstrap} iterations)...")
    model, ckpt = load_model(str(ckpt_path), device)

    # Load data
    with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
        compound_index = pickle.load(f)
    with open(MODELING_DIR / 'splits.json') as f:
        splits = json.load(f)

    n_compounds = len(np.load(
        GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True))
    emb_mmap = np.memmap(
        GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
        dtype='float16', mode='r', shape=(n_compounds, 768)
    )

    # Pre-compute projections
    test_pos_ids = [cid for cid in splits['test']['positive_ids']
                    if cid in compound_index]
    test_neg_ids = [cid for cid in splits['test']['negative_ids']
                    if cid in compound_index]

    z_pos = project_ids(model, test_pos_ids, compound_index, emb_mmap, device)
    potencies = np.array([compound_index[cid]['potency_class_num']
                          for cid in test_pos_ids])

    # VP centroid from training set
    vp_ids = [cid for cid in splits['train']['positive_ids']
              if cid in compound_index
              and compound_index[cid].get('potency_class_num') == 3]
    z_vp = project_ids(model, vp_ids, compound_index, emb_mmap, device)
    centroid = z_vp.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    dists = 1.0 - z_pos @ centroid

    # T3C negatives (target-switch test)
    t3c_ids = [cid for cid in test_neg_ids
               if compound_index[cid].get('neg_tier') == 'T3C']
    z_t3c = project_ids(model, t3c_ids, compound_index, emb_mmap, device) \
        if t3c_ids else None

    # T2 negatives (Cohen's d)
    t2_ids = [cid for cid in test_neg_ids
              if compound_index[cid].get('neg_tier') == 'T2']
    z_t2 = project_ids(model, t2_ids, compound_index, emb_mmap, device) \
        if t2_ids else None

    from scipy import stats
    from sklearn.metrics import roc_auc_score

    bootstrap_results = {
        'spearman_rho': [],
        'target_switch_auc': [],
        'cohens_d': [],
    }

    rng = np.random.RandomState(seed)
    n_pos = len(test_pos_ids)

    for _ in range(n_bootstrap):
        idx = rng.choice(n_pos, size=n_pos, replace=True)
        boot_dists = dists[idx]
        boot_potencies = potencies[idx]

        # Spearman ρ
        rho, _ = stats.spearmanr(-boot_dists, boot_potencies)
        if not np.isnan(rho):
            bootstrap_results['spearman_rho'].append(float(rho))

        # Target switch AUC
        if z_t3c is not None and len(z_t3c) > 0:
            pos_scores = z_pos[idx] @ centroid
            n_t3c = len(z_t3c)
            t3c_idx = rng.choice(n_t3c, size=n_t3c, replace=True)
            t3c_scores = z_t3c[t3c_idx] @ centroid
            y_true = np.concatenate([np.ones(n_pos), np.zeros(n_t3c)])
            y_scores = np.concatenate([pos_scores, t3c_scores])
            try:
                auc = roc_auc_score(y_true, y_scores)
                bootstrap_results['target_switch_auc'].append(float(auc))
            except ValueError:
                pass

        # Cohen's d
        if z_t2 is not None and len(z_t2) > 0:
            sim_pos = z_pos[idx] @ centroid
            n_t2 = len(z_t2)
            t2_idx = rng.choice(n_t2, size=n_t2, replace=True)
            sim_t2 = z_t2[t2_idx] @ centroid
            pooled_std = np.sqrt((sim_pos.std()**2 + sim_t2.std()**2) / 2)
            d = (sim_pos.mean() - sim_t2.mean()) / (pooled_std + 1e-12)
            bootstrap_results['cohens_d'].append(float(d))

    # Compute CIs
    ci_results = {}
    for metric, values in bootstrap_results.items():
        if values:
            arr = np.array(values)
            ci_results[metric] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'ci_2.5': float(np.percentile(arr, 2.5)),
                'ci_97.5': float(np.percentile(arr, 97.5)),
                'n_bootstrap': len(values),
            }

    out_path = VALIDATION_DIR / 'logs' / f'bootstrap_seed{seed}.json'
    with open(out_path, 'w') as f:
        json.dump(ci_results, f, indent=2)
    print(f"  Bootstrap CI → {out_path}")
    return ci_results


def main():
    parser = argparse.ArgumentParser(description='GENESIS Phase 5a: Multi-seed validation')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--min-epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training, only evaluate existing checkpoints')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    args = parser.parse_args()

    (VALIDATION_DIR / 'logs').mkdir(parents=True, exist_ok=True)
    (VALIDATION_DIR / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # Save run config
    config = vars(args)
    config['script'] = str(Path(__file__).name)
    config['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(VALIDATION_DIR / 'run_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ── Multi-seed training ──────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "█" * 60)
        print("  PHASE 5a: Multi-Seed Training")
        print("█" * 60)
        for seed in args.seeds:
            rc = run_training(seed, args.device, args.epochs,
                              args.min_epochs, args.patience)
            if rc != 0:
                print(f"  ⚠ Seed {seed} failed — continuing with remaining seeds")

    # ── Evaluation per seed ──────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  PHASE 5a: Evaluation per Seed")
    print("█" * 60)
    all_results = {}
    for seed in args.seeds:
        results = run_evaluation(seed, args.device)
        if results:
            all_results[str(seed)] = results

    # ── Cross-seed summary ───────────────────────────────────────────
    if len(all_results) > 1:
        metrics = {}
        for seed_str, res in all_results.items():
            for k, v in res.items():
                if isinstance(v, (int, float)) and v is not None:
                    metrics.setdefault(k, []).append(v)

        summary = {}
        print(f"\n  {'Metric':<35s} {'Mean':>10s} {'±SD':>10s} {'Seeds':>6s}")
        print(f"  {'-'*61}")
        for k, vals in sorted(metrics.items()):
            if len(vals) >= 2:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                summary[k] = {
                    'mean': float(mean_v),
                    'std': float(std_v),
                    'values': vals,
                    'n_seeds': len(vals),
                }
                print(f"  {k:<35s} {mean_v:10.4f} {std_v:10.4f} {len(vals):6d}")

        summary_path = VALIDATION_DIR / 'logs' / 'multiseed_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary → {summary_path}")

    # ── Bootstrap CI ─────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  PHASE 5a: Bootstrap Confidence Intervals")
    print("█" * 60)
    all_bootstrap = {}
    for seed in args.seeds:
        ci = bootstrap_evaluation(seed, args.device,
                                  n_bootstrap=args.n_bootstrap)
        if ci:
            all_bootstrap[str(seed)] = ci

    # Aggregate bootstrap across seeds
    if all_bootstrap:
        agg = {}
        for seed_str, ci in all_bootstrap.items():
            for metric, vals in ci.items():
                agg.setdefault(metric, {
                    'means': [], 'ci_lows': [], 'ci_highs': []
                })
                agg[metric]['means'].append(vals['mean'])
                agg[metric]['ci_lows'].append(vals['ci_2.5'])
                agg[metric]['ci_highs'].append(vals['ci_97.5'])

        bootstrap_summary = {}
        print(f"\n  {'Metric':<25s} {'Mean':>8s} {'95% CI':>22s}")
        print(f"  {'-'*55}")
        for metric, data in sorted(agg.items()):
            mean_v = np.mean(data['means'])
            ci_lo = np.mean(data['ci_lows'])
            ci_hi = np.mean(data['ci_highs'])
            bootstrap_summary[metric] = {
                'mean': float(mean_v),
                'ci_95': [float(ci_lo), float(ci_hi)],
                'per_seed_means': data['means'],
            }
            print(f"  {metric:<25s} {mean_v:8.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")

        bs_path = VALIDATION_DIR / 'logs' / 'bootstrap_summary.json'
        with open(bs_path, 'w') as f:
            json.dump(bootstrap_summary, f, indent=2)
        print(f"\n  Bootstrap summary → {bs_path}")

    # CRITICAL: after training multiple seeds in sequence, the shared
    # modeling/checkpoints/best.pt holds whichever seed trained LAST, not
    # necessarily seed 42 (the designated canonical seed for all downstream
    # phases). Always restore seed 42 here so nothing downstream silently
    # picks up a non-canonical seed's weights.
    canonical_src = VALIDATION_DIR / 'checkpoints' / 'best_seed42.pt'
    shared_ckpt = MODELING_DIR / 'checkpoints' / 'best.pt'
    if canonical_src.exists():
        shutil.copy2(canonical_src, shared_ckpt)
        pgcl_backup = MODELING_DIR / 'checkpoints' / 'pgcl_best_seed42.pt'
        shutil.copy2(canonical_src, pgcl_backup)
        print(f"\n  Restored canonical seed-42 checkpoint -> {shared_ckpt}")

    print("\n✓ Multi-seed training + bootstrap CI complete")


if __name__ == '__main__':
    main()
