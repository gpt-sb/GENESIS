#!/usr/bin/env python3
"""
GENESIS — Phase 5b: Ablation Studies.

Systematic ablation of contrastive model hyperparameters to validate
or improve the Phase 2 configuration before proceeding to generative models.

Ablation space:
  Temperature:    τ ∈ {0.05, 0.07, 0.10}
  Projection dim: d ∈ {64, 128, 256}
  Loss variants:  full, no_ranking, no_potency
  Curriculum:     curriculum vs uniform sampling (TODO: requires train.py flag)

Results → GENESIS/modeling/validation/ablations/

Usage:
    python GENESIS/modeling/run_ablations.py --device mps
    python GENESIS/modeling/run_ablations.py --configs dim_256 no_ranking --device mps
    python GENESIS/modeling/run_ablations.py --skip-train --device mps  # eval only
"""
import os
import sys
import json
import time
import subprocess
import shutil
import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR
GENESIS_DIR = SCRIPT_DIR.parent
ABLATION_DIR = MODELING_DIR / 'validation' / 'ablations'

sys.path.insert(0, str(MODELING_DIR))

# ── Ablation Configurations ──────────────────────────────────────────
# temp_0.07 (d=128, full) = baseline, same as Phase 2 single-seed.
# We still include it for clean comparison under identical eval harness.

ABLATION_CONFIGS = [
    # Temperature sweep (d=128, full loss)
    {'name': 'temp_0.05', 'temperature': 0.05, 'proj_dim': 128, 'loss_variant': 'full'},
    {'name': 'temp_0.07', 'temperature': 0.07, 'proj_dim': 128, 'loss_variant': 'full'},
    {'name': 'temp_0.10', 'temperature': 0.10, 'proj_dim': 128, 'loss_variant': 'full'},

    # Projection dimension sweep (τ=0.07, full loss)
    {'name': 'dim_64',  'temperature': 0.07, 'proj_dim': 64,  'loss_variant': 'full'},
    # dim_128 already covered by temp_0.07
    {'name': 'dim_256', 'temperature': 0.07, 'proj_dim': 256, 'loss_variant': 'full'},

    # Loss component ablation (τ=0.07, d=128)
    {'name': 'no_ranking',  'temperature': 0.07, 'proj_dim': 128, 'loss_variant': 'no_ranking'},
    {'name': 'no_potency',  'temperature': 0.07, 'proj_dim': 128, 'loss_variant': 'no_potency'},

    # Curriculum ablation (τ=0.07, d=128, full loss)
    {'name': 'no_curriculum', 'temperature': 0.07, 'proj_dim': 128, 'loss_variant': 'full',
     'uniform_sampling': True},
]


def run_ablation(config, device, epochs=40, seed=42, min_epochs=25, patience=7):
    """Train one ablation configuration."""
    name = config['name']
    out_dir = ABLATION_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    # train.py writes every ablation config to the SHARED modeling/checkpoints/
    # best.pt, which would clobber the canonical PGCL model. Protect it: back
    # up the canonical SEED-42 checkpoint (not this call's `seed`, in case
    # ablations are ever run with a different seed) before the first ablation,
    # restore after each config's run.
    shared_ckpt = MODELING_DIR / 'checkpoints' / 'best.pt'
    pgcl_backup = MODELING_DIR / 'checkpoints' / 'pgcl_best_seed42.pt'
    if shared_ckpt.exists() and not pgcl_backup.exists():
        shutil.copy2(shared_ckpt, pgcl_backup)
        print(f'Backed up canonical PGCL checkpoint -> {pgcl_backup}')
    log_path = out_dir / 'train.log'

    cmd = [
        python, str(MODELING_DIR / 'train.py'),
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch-size', '512',
        '--batch-pos', '128',
        '--lr', '3e-4',
        '--temperature', str(config['temperature']),
        '--proj-dim', str(config['proj_dim']),
        '--patience', str(patience),
        '--min-epochs', str(min_epochs),
        '--device', device,
    ]

    if config['loss_variant'] == 'no_ranking':
        cmd.append('--no-ranking')
    elif config['loss_variant'] == 'no_potency':
        cmd.append('--no-potency')

    if config.get('uniform_sampling', False):
        cmd.append('--uniform-sampling')

    print(f"\n{'='*60}")
    print(f"  Ablation: {name}")
    print(f"  τ={config['temperature']}, d={config['proj_dim']}, loss={config['loss_variant']}")
    print(f"{'='*60}")
    t0 = time.time()

    with open(log_path, 'w') as lf:
        proc = subprocess.run(cmd, cwd=str(GENESIS_DIR.parent),
                              stdout=lf, stderr=subprocess.STDOUT)

    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"  → {status} [{elapsed/60:.1f} min]")

    if proc.returncode == 0:
        # Copy checkpoint to ablation output dir
        src = MODELING_DIR / 'checkpoints' / 'best.pt'
        if src.exists():
            shutil.copy2(src, out_dir / 'best.pt')

        # Copy training history
        hist_src = MODELING_DIR / 'logs' / f'history_seed{seed}.json'
        if hist_src.exists():
            shutil.copy2(hist_src, out_dir / 'history.json')

        # Save this ablation's config
        with open(out_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # CRITICAL: restore the canonical PGCL model into the shared scratch path
    # regardless of success/failure, so it's never left holding this ablation's
    # weights for the next config's run (or any other consumer) to pick up.
    if pgcl_backup.exists():
        shutil.copy2(pgcl_backup, shared_ckpt)

    return proc.returncode


def evaluate_ablation(config, device):
    """Evaluate one ablation checkpoint."""
    name = config['name']
    out_dir = ABLATION_DIR / name
    ckpt_path = out_dir / 'best.pt'

    if not ckpt_path.exists():
        print(f"  Skipping {name}: no checkpoint")
        return None

    print(f"\n  Evaluating {name}...")
    from evaluate import run_full_evaluation
    results = run_full_evaluation(str(ckpt_path), device=device)

    out_path = out_dir / 'evaluation.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  → {out_path}")
    return results


def compare_results(all_results):
    """Print comparison table and identify the winner."""
    # Core metrics from GENESIS evaluation battery
    key_metrics = [
        'global_spearman_rho',
        'cliff_accuracy',
        'target_switch_auc_roc',
        'cohens_d',
        'vp_retrieval_precision_at_100',
        'potency_graded_ndcg',
    ]
    # Fallback: use whatever metrics we find
    available = set()
    for data in all_results.values():
        available.update(data['metrics'].keys())
    key_metrics = [m for m in key_metrics if m in available]

    if not key_metrics:
        key_metrics = sorted(available)[:6]

    print(f"\n{'='*110}")
    print("ABLATION RESULTS COMPARISON")
    print(f"{'='*110}")

    # Header
    header = f"  {'Config':<18s} {'τ':>5s} {'d':>5s} {'loss':>12s}"
    for m in key_metrics:
        short = m.replace('global_', '').replace('_', ' ')[:14]
        header += f" {short:>14s}"
    print(header)
    print(f"  {'-'*len(header)}")

    # Rows
    best_scores = {m: (-np.inf, '') for m in key_metrics}
    for name in sorted(all_results.keys()):
        data = all_results[name]
        cfg = data['config']
        row = f"  {name:<18s} {cfg['temperature']:>5.2f} {cfg['proj_dim']:>5d} {cfg['loss_variant']:>12s}"
        for m in key_metrics:
            v = data['metrics'].get(m)
            if v is not None:
                row += f" {v:14.4f}"
                if v > best_scores[m][0]:
                    best_scores[m] = (v, name)
            else:
                row += f" {'N/A':>14s}"
        print(row)

    # Winner analysis
    print(f"\n  {'Metric':<35s} {'Best Config':>18s} {'Value':>10s}")
    print(f"  {'-'*63}")
    winner_counts = {}
    for m in key_metrics:
        val, winner = best_scores[m]
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        print(f"  {m:<35s} {winner:>18s} {val:10.4f}")

    overall_winner = max(winner_counts, key=winner_counts.get)
    print(f"\n  Overall winner (majority metrics): {overall_winner} "
          f"({winner_counts[overall_winner]}/{len(key_metrics)} metrics)")

    # Decision gate
    baseline = all_results.get('temp_0.07', {}).get('metrics', {})
    if baseline and overall_winner != 'temp_0.07':
        winner_metrics = all_results[overall_winner]['metrics']
        winner_cfg = all_results[overall_winner]['config']
        print(f"\n  ⚠ DECISION GATE: {overall_winner} outperforms baseline (temp_0.07)")
        print(f"    Winner config: τ={winner_cfg['temperature']}, "
              f"d={winner_cfg['proj_dim']}, loss={winner_cfg['loss_variant']}")
        print(f"    → ACTION: Re-train Phase 2 with winner config before Phase 3")
        if winner_cfg['proj_dim'] != 128:
            print(f"    → WARNING: proj_dim changed to {winner_cfg['proj_dim']} — "
                  f"Phase 3 generative scripts need dimension updates")
    else:
        print(f"\n  ✓ DECISION GATE: Baseline config (τ=0.07, d=128, full) is optimal")
        print(f"    → Phase 3 can proceed with current checkpoint")

    return overall_winner, winner_counts


def main():
    parser = argparse.ArgumentParser(description='GENESIS Phase 5b: Ablation studies')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--min-epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training, only evaluate existing checkpoints')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Specific config names to run (default: all)')
    args = parser.parse_args()

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    configs = ABLATION_CONFIGS
    if args.configs:
        configs = [c for c in configs if c['name'] in args.configs]
        if not configs:
            print(f"No configs matched: {args.configs}")
            print(f"Available: {[c['name'] for c in ABLATION_CONFIGS]}")
            sys.exit(1)

    # Save run config
    run_config = {
        'script': str(Path(__file__).name),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'configs_to_run': [c['name'] for c in configs],
    }
    with open(ABLATION_DIR / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    # ── Training ─────────────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "█" * 60)
        print("  PHASE 5b: Ablation Training")
        print(f"  {len(configs)} configurations × seed {args.seed}")
        print("█" * 60)
        for config in configs:
            rc = run_ablation(config, args.device, args.epochs, args.seed,
                              args.min_epochs, args.patience)
            if rc != 0:
                print(f"  ⚠ {config['name']} FAILED — continuing")

    # ── Evaluation ───────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  PHASE 5b: Ablation Evaluation")
    print("█" * 60)

    all_results = {}
    for config in configs:
        results = evaluate_ablation(config, args.device)
        if results:
            all_results[config['name']] = {
                'config': config,
                'metrics': results,
            }

    # ── Comparison & Decision Gate ───────────────────────────────────
    if len(all_results) >= 2:
        winner, counts = compare_results(all_results)
    elif len(all_results) == 1:
        name = list(all_results.keys())[0]
        print(f"\n  Only one config evaluated ({name}) — no comparison possible")

    # Save all results
    if all_results:
        summary_path = ABLATION_DIR / 'ablation_results.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Full results → {summary_path}")

    print("\n✓ Ablation studies complete")


if __name__ == '__main__':
    main()
