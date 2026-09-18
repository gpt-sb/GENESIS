#!/usr/bin/env python3
"""
GENESIS — Multi-Seed Generative Comparison.

Re-runs all 6 generative methods (VAE, cVAE, Diffusion, RL-VAE, CGLF, CRAVE)
with seeds {42, 123, 456}, evaluates each, and aggregates metrics with bootstrap CIs.

Dependency chain (per seed):
  VAE ──┬──→ Diffusion
        ├──→ RL-VAE ──→ CRAVE
        └──→ CGLF
  cVAE  (independent)

Seed 42 can be skipped (--skip-seed42) to reuse existing results.

Output directory:
  GENESIS/generative/multiseed/
    seed_42/   (generated_*.txt, eval_*.json, history_*.json)
    seed_123/
    seed_456/
    aggregate/ (multiseed_comparison.tsv, multiseed_summary.json)

Usage:
    python GENESIS/generative/multiseed_gen.py --device mps --skip-seed42
    python GENESIS/generative/multiseed_gen.py --aggregate-only
"""
import os
import sys
import json
import time
import shutil
import subprocess
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent          # GENESIS/generative/
GENESIS_DIR = SCRIPT_DIR.parent                       # GENESIS/
GEN_DIR = SCRIPT_DIR                                  # GENESIS/generative/
CHECKPOINT_DIR = GEN_DIR / 'checkpoints'
RESULTS_DIR = GEN_DIR / 'results'
LOG_DIR = GEN_DIR / 'logs'
MULTISEED_DIR = GEN_DIR / 'multiseed'

PYTHON = sys.executable

SEEDS = [42, 123, 456]

# ── Method definitions ───────────────────────────────────────────────

METHODS = [
    {
        'name': 'vae',
        'script': 'vae.py',
        'args': ['--mode', 'vae', '--epochs', '100'],
        'checkpoint': 'best_vae.pt',
        'generated': 'generated_vae.txt',
        'history': 'history_vae.json',
        'eval_name': 'vae',
        'depends_on': [],
    },
    {
        'name': 'cvae',
        'script': 'vae.py',
        'args': ['--mode', 'cvae', '--epochs', '100'],
        'checkpoint': 'best_cvae.pt',
        'generated': 'generated_cvae.txt',
        'history': 'history_cvae.json',
        'eval_name': 'cvae',
        'depends_on': [],
    },
    {
        'name': 'diffusion',
        'script': 'diffusion.py',
        'args': ['--epochs', '200'],
        'checkpoint': 'best_diffusion.pt',
        'generated': 'generated_diffusion.txt',
        'history': 'history_diffusion.json',
        'eval_name': 'diffusion',
        'depends_on': ['vae'],  # loads best_vae.pt
    },
    {
        'name': 'rl_vae',
        'script': 'rl_finetune.py',
        'args': ['--base-model', 'vae', '--epochs', '50'],
        'checkpoint': 'best_rl_vae.pt',
        'generated': 'generated_rl_vae.txt',
        'history': 'history_rl_vae.json',
        'eval_name': 'rl_vae',
        'depends_on': ['vae'],  # loads best_vae.pt
    },
    {
        'name': 'cglf',
        'script': 'cglf.py',
        'args': ['--stage1-epochs', '50', '--stage2-epochs', '200'],
        'checkpoint': 'best_cglf.pt',
        'generated': 'generated_cglf.txt',
        'history': 'history_cglf.json',
        'eval_name': 'cglf',
        'extra_checkpoints': ['cglf_decoder.pt'],
        'depends_on': ['vae'],  # loads best_vae.pt
    },
    {
        'name': 'crave',
        'script': 'crave_gen.py',
        'args': ['--epochs', '30'],
        'checkpoint': 'best_crave.pt',
        'generated': ['generated_crave.txt', 'generated_crave_unfiltered.txt'],
        'history': 'history_crave.json',
        'eval_names': ['crave_filtered', 'crave_unfiltered'],
        'depends_on': ['rl_vae'],  # loads best_rl_vae.pt
    },
]


# ── Helpers ──────────────────────────────────────────────────────────

def seed_dir(seed):
    return MULTISEED_DIR / f'seed_{seed}'


def run_method(method, seed, device, cwd):
    """Run a single generative method with the given seed."""
    script = str(GEN_DIR / method['script'])
    cmd = [PYTHON, script] + method['args'] + ['--seed', str(seed), '--device', device]

    log.info('  CMD: %s', ' '.join(cmd))
    t0 = time.time()

    log_path = seed_dir(seed) / f"run_{method['name']}.log"
    with open(log_path, 'w') as lf:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT)

    elapsed = time.time() - t0
    status = 'OK' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'
    log.info('  → %s [%.1f min]', status, elapsed / 60)

    if proc.returncode != 0:
        log.error('  Check log: %s', log_path)

    return proc.returncode == 0


def copy_seed42_existing():
    """Copy existing seed=42 results to multiseed/seed_42/."""
    dst = seed_dir(42)
    dst.mkdir(parents=True, exist_ok=True)

    copied = []

    for method in METHODS:
        # Checkpoints
        ckpt_src = CHECKPOINT_DIR / method['checkpoint']
        if ckpt_src.exists():
            shutil.copy2(ckpt_src, dst / method['checkpoint'])
            copied.append(method['checkpoint'])

        for extra in method.get('extra_checkpoints', []):
            src = CHECKPOINT_DIR / extra
            if src.exists():
                shutil.copy2(src, dst / extra)
                copied.append(extra)

        # Generated files
        gen_files = method.get('generated', [])
        if isinstance(gen_files, str):
            gen_files = [gen_files]
        for gf in gen_files:
            src = RESULTS_DIR / gf
            if src.exists():
                shutil.copy2(src, dst / gf)
                copied.append(gf)

        # History
        hist_src = LOG_DIR / method['history']
        if hist_src.exists():
            shutil.copy2(hist_src, dst / method['history'])
            copied.append(method['history'])

        # Eval JSONs
        eval_names = method.get('eval_names', [method.get('eval_name', method['name'])])
        for en in eval_names:
            eval_src = RESULTS_DIR / f'eval_{en}.json'
            if eval_src.exists():
                shutil.copy2(eval_src, dst / f'eval_{en}.json')
                copied.append(f'eval_{en}.json')

    log.info('  Copied %d files for seed=42', len(copied))
    return copied


def backup_and_restore_checkpoints(seed, phase):
    """
    Before training a new seed: back up current checkpoints.
    After training: save new checkpoints to seed dir, restore originals.
    """
    backup_dir = MULTISEED_DIR / '_checkpoint_backup'

    if phase == 'backup':
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in CHECKPOINT_DIR.glob('*.pt'):
            shutil.copy2(f, backup_dir / f.name)
        log.info('  Backed up %d checkpoints', len(list(CHECKPOINT_DIR.glob('*.pt'))))

    elif phase == 'save_and_restore':
        dst = seed_dir(seed)
        # Copy new checkpoints to seed dir
        for method in METHODS:
            ckpt = CHECKPOINT_DIR / method['checkpoint']
            if ckpt.exists():
                shutil.copy2(ckpt, dst / method['checkpoint'])
            for extra in method.get('extra_checkpoints', []):
                src = CHECKPOINT_DIR / extra
                if src.exists():
                    shutil.copy2(src, dst / extra)

        # Copy generated files & histories
        for method in METHODS:
            gen_files = method.get('generated', [])
            if isinstance(gen_files, str):
                gen_files = [gen_files]
            for gf in gen_files:
                src = RESULTS_DIR / gf
                if src.exists():
                    shutil.copy2(src, dst / gf)
            hist = LOG_DIR / method['history']
            if hist.exists():
                shutil.copy2(hist, dst / method['history'])

        # Restore original checkpoints
        for f in backup_dir.glob('*.pt'):
            shutil.copy2(f, CHECKPOINT_DIR / f.name)
        log.info('  Saved seed=%d outputs & restored original checkpoints', seed)


def evaluate_seed(seed, device):
    """Run evaluation on all methods for a given seed."""
    dst = seed_dir(seed)

    sys.path.insert(0, str(GEN_DIR))
    from evaluate_gen import evaluate_molecules

    for method in METHODS:
        gen_files = method.get('generated', [])
        if isinstance(gen_files, str):
            gen_files = [gen_files]
        eval_names = method.get('eval_names', [method.get('eval_name', method['name'])])

        for gf, en in zip(gen_files, eval_names):
            gen_path = dst / gf
            if not gen_path.exists():
                log.warning('  Skipping %s: %s not found', en, gen_path)
                continue

            with open(gen_path) as f:
                smiles = [line.strip() for line in f if line.strip()]

            if not smiles:
                log.warning('  Skipping %s: empty file', en)
                continue

            log.info('  Evaluating %s (seed=%d): %d SMILES', en, seed, len(smiles))

            # Temporarily redirect eval output to seed dir
            orig_results_dir = os.environ.get('GENESIS_EVAL_RESULTS_DIR')
            os.environ['GENESIS_EVAL_RESULTS_DIR'] = str(dst)

            results = evaluate_molecules(smiles, method_name=en, device=device,
                                         compute_potency=True)

            # The function writes eval_{name}.json to RESULTS_DIR internally,
            # so also copy it to seed dir explicitly
            eval_src = RESULTS_DIR / f'eval_{en}.json'
            eval_dst = dst / f'eval_{en}.json'
            if eval_src.exists() and str(eval_src) != str(eval_dst):
                shutil.copy2(eval_src, eval_dst)

            # Also write directly in case the env var isn't honored
            with open(eval_dst, 'w') as f:
                json.dump(results, f, indent=2)

            if orig_results_dir:
                os.environ['GENESIS_EVAL_RESULTS_DIR'] = orig_results_dir
            elif 'GENESIS_EVAL_RESULTS_DIR' in os.environ:
                del os.environ['GENESIS_EVAL_RESULTS_DIR']

            log.info('    → %s', eval_dst)


def aggregate_results(seeds):
    """Aggregate metrics across seeds with bootstrap CIs."""
    agg_dir = MULTISEED_DIR / 'aggregate'
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Collect all eval_names
    all_eval_names = []
    for method in METHODS:
        all_eval_names.extend(
            method.get('eval_names', [method.get('eval_name', method['name'])]))

    # Metric keys of interest
    metric_keys = [
        'validity', 'uniqueness', 'novelty', 'oscc_potency',
        'pchembl_pred', 'vp_fraction', 'qed', 'sa_score',
        'diversity', 'scaffold_novelty',
    ]

    # Load per-seed results
    all_data = {}  # {eval_name: {metric: [seed_values]}}
    per_seed_data = {}  # {seed: {eval_name: {metrics}}}

    for seed in seeds:
        per_seed_data[seed] = {}
        for en in all_eval_names:
            eval_path = seed_dir(seed) / f'eval_{en}.json'
            if not eval_path.exists():
                log.warning('Missing: %s', eval_path)
                continue

            with open(eval_path) as f:
                data = json.load(f)

            metrics = data.get('metrics', {})
            per_seed_data[seed][en] = metrics

            if en not in all_data:
                all_data[en] = {k: [] for k in metric_keys}
            for k in metric_keys:
                if k in metrics:
                    all_data[en][k].append(metrics[k])

    # Compute mean ± SD + bootstrap 95% CI
    summary_rows = []
    summary_json = {}

    for en in all_eval_names:
        if en not in all_data:
            continue
        row = {'method': en}
        method_summary = {}

        for k in metric_keys:
            vals = all_data[en][k]
            if len(vals) < 2:
                row[k] = f'{vals[0]:.4f}' if vals else 'N/A'
                method_summary[k] = {
                    'mean': vals[0] if vals else None,
                    'sd': None, 'ci_lo': None, 'ci_hi': None,
                    'values': vals,
                }
                continue

            arr = np.array(vals)
            mean = arr.mean()
            sd = arr.std(ddof=1) if len(arr) > 1 else 0.0

            # Bootstrap 95% CI
            rng = np.random.RandomState(42)
            boot_means = []
            for _ in range(1000):
                boot_sample = rng.choice(arr, size=len(arr), replace=True)
                boot_means.append(boot_sample.mean())
            boot_means = np.array(boot_means)
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)

            row[k] = f'{mean:.4f} ± {sd:.4f}'
            row[f'{k}_ci'] = f'[{ci_lo:.4f}, {ci_hi:.4f}]'

            method_summary[k] = {
                'mean': float(mean),
                'sd': float(sd),
                'ci_lo': float(ci_lo),
                'ci_hi': float(ci_hi),
                'values': [float(v) for v in vals],
            }

        summary_rows.append(row)
        summary_json[en] = method_summary

    # Save comparison TSV
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        tsv_path = agg_dir / 'multiseed_comparison.tsv'
        df.to_csv(tsv_path, sep='\t', index=False)
        log.info('Comparison table → %s', tsv_path)

        # Also print a clean table
        print('\n' + '=' * 80)
        print('  MULTI-SEED GENERATIVE COMPARISON  (seeds: %s)', seeds)
        print('=' * 80)
        display_cols = ['method'] + metric_keys
        display_df = df[[c for c in display_cols if c in df.columns]]
        print(display_df.to_string(index=False))
        print()

    # Save JSON summary
    full_summary = {
        'seeds': seeds,
        'n_seeds': len(seeds),
        'methods': summary_json,
        'per_seed': {str(s): per_seed_data[s] for s in seeds if s in per_seed_data},
    }
    json_path = agg_dir / 'multiseed_summary.json'
    with open(json_path, 'w') as f:
        json.dump(full_summary, f, indent=2)
    log.info('Summary JSON → %s', json_path)

    # Paper-ready table (mean ± SD format)
    paper_rows = []
    for en in all_eval_names:
        if en not in summary_json:
            continue
        row = {'Method': en}
        for k in metric_keys:
            info = summary_json[en].get(k, {})
            mean = info.get('mean')
            sd = info.get('sd')
            if mean is not None and sd is not None:
                # Format percentages vs raw values
                if k in ('validity', 'uniqueness', 'novelty', 'vp_fraction', 'scaffold_novelty'):
                    row[k] = f'{100*mean:.1f} ± {100*sd:.1f}'
                else:
                    row[k] = f'{mean:.3f} ± {sd:.3f}'
            elif mean is not None:
                if k in ('validity', 'uniqueness', 'novelty', 'vp_fraction', 'scaffold_novelty'):
                    row[k] = f'{100*mean:.1f}'
                else:
                    row[k] = f'{mean:.3f}'
            else:
                row[k] = 'N/A'
        paper_rows.append(row)

    if paper_rows:
        paper_df = pd.DataFrame(paper_rows)
        paper_path = agg_dir / 'table2_with_ci.tsv'
        paper_df.to_csv(paper_path, sep='\t', index=False)
        log.info('Paper-ready Table 2 → %s', paper_path)

    return summary_json


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GENESIS Multi-Seed Generative Comparison')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Seeds to evaluate (default: 42 123 456)')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--skip-seed42', action='store_true',
                        help='Reuse existing seed=42 results instead of retraining')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Skip training, just aggregate existing results')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Skip training, re-evaluate generated files for all seeds')
    args = parser.parse_args()

    MULTISEED_DIR.mkdir(parents=True, exist_ok=True)
    cwd = str(GENESIS_DIR.parent)  # research root (parent of GENESIS/)

    seeds = args.seeds

    if args.aggregate_only:
        log.info('=== Aggregate Only Mode ===')
        aggregate_results(seeds)
        return

    if args.evaluate_only:
        log.info('=== Evaluate Only Mode ===')
        for seed in seeds:
            log.info('Evaluating seed=%d ...', seed)
            evaluate_seed(seed, args.device)
        aggregate_results(seeds)
        return

    # ── Full pipeline ────────────────────────────────────────────────
    total_t0 = time.time()

    for seed in seeds:
        seed_t0 = time.time()
        log.info('\n' + '=' * 70)
        log.info('  SEED = %d', seed)
        log.info('=' * 70)

        sd = seed_dir(seed)
        sd.mkdir(parents=True, exist_ok=True)

        # Seed 42: copy existing results
        if seed == 42 and args.skip_seed42:
            log.info('  Reusing existing seed=42 results')
            copy_seed42_existing()

            # Re-evaluate to ensure eval JSONs are present
            log.info('  Verifying evaluations ...')
            evaluate_seed(seed, args.device)
            log.info('  Seed 42 complete [%.1f min]', (time.time() - seed_t0) / 60)
            continue

        # New seed: back up current checkpoints
        backup_and_restore_checkpoints(seed, 'backup')

        # Train each method in dependency order
        success = {}
        for method in METHODS:
            log.info('\n  --- %s (seed=%d) ---', method['name'].upper(), seed)

            # Check dependencies
            deps_ok = all(success.get(dep, False) for dep in method['depends_on'])
            if not deps_ok:
                log.warning('  Skipping %s: dependency failed', method['name'])
                success[method['name']] = False
                continue

            ok = run_method(method, seed, args.device, cwd)
            success[method['name']] = ok

        # Save outputs and restore original checkpoints
        backup_and_restore_checkpoints(seed, 'save_and_restore')

        # Evaluate
        log.info('\n  --- EVALUATION (seed=%d) ---', seed)
        evaluate_seed(seed, args.device)

        log.info('  Seed %d complete [%.1f min]', seed, (time.time() - seed_t0) / 60)

    # ── Aggregate ────────────────────────────────────────────────────
    log.info('\n' + '=' * 70)
    log.info('  AGGREGATION')
    log.info('=' * 70)
    aggregate_results(seeds)

    total_elapsed = time.time() - total_t0
    log.info('\n✓ Multi-seed generative comparison complete [%.1f hours]',
             total_elapsed / 3600)

    # Clean up backup
    backup_dir = MULTISEED_DIR / '_checkpoint_backup'
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


if __name__ == '__main__':
    main()
