#!/usr/bin/env python3
"""
Phase 3 — Run All Generative Methods & Comparative Evaluation.

Orchestrates the full pipeline:
  1. VAE training
  2. cVAE with centroid conditioning
  3. Latent Diffusion
  4. RL Fine-Tuning
  5. CGLF (novel method)
  6. CRAVE (enhanced RL + multi-strategy + Pareto)

Then runs unified evaluation and generates comparison table.

Usage:
    python generative/run_all.py --device mps
    python generative/run_all.py --evaluate-only
"""
import os
import sys
import json
import subprocess
import logging
import argparse

GENESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(GENESIS_DIR, 'generative')
RESULTS_DIR = os.path.join(GEN_DIR, 'results')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

PYTHON = sys.executable


def run_method(script, args_list, method_name):
    """Run a generative method script."""
    cmd = [PYTHON, script] + args_list
    log.info('=== Running %s ===', method_name)
    log.info('Command: %s', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error('%s failed with return code %d', method_name, result.returncode)
        return False
    log.info('%s complete', method_name)
    return True


def evaluate_all(device='cpu'):
    """Run unified evaluation on all generated molecule sets."""
    sys.path.insert(0, GEN_DIR)
    from evaluate_gen import evaluate_molecules, compare_methods

    methods = {
        'vae': 'generated_vae.txt',
        'cvae': 'generated_cvae.txt',
        'diffusion': 'generated_diffusion.txt',
        'rl_vae': 'generated_rl_vae.txt',
        'cglf': 'generated_cglf.txt',
        'crave_filtered': 'generated_crave.txt',
    }

    for name, filename in methods.items():
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            log.warning('Skipping %s: %s not found', name, path)
            continue

        with open(path) as f:
            smiles = [line.strip() for line in f if line.strip()]

        log.info('Evaluating %s: %d SMILES', name, len(smiles))
        evaluate_molecules(smiles, method_name=name, device=device, compute_potency=True)

    df = compare_methods()
    if df is not None:
        df.to_csv(os.path.join(RESULTS_DIR, 'method_comparison.tsv'), sep='\t')
        log.info('Comparison table -> method_comparison.tsv')

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--skip-vae', action='store_true', help='Skip VAE if already trained')
    parser.add_argument('--skip-cvae', action='store_true')
    parser.add_argument('--vae-epochs', type=int, default=100)
    parser.add_argument('--diffusion-epochs', type=int, default=200)
    parser.add_argument('--rl-epochs', type=int, default=50)
    parser.add_argument('--cglf-s1-epochs', type=int, default=50)
    parser.add_argument('--cglf-s2-epochs', type=int, default=200)
    parser.add_argument('--crave-epochs', type=int, default=30)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.evaluate_only:
        evaluate_all(args.device)
        return

    scripts = GEN_DIR

    # 1. VAE
    if not args.skip_vae:
        run_method(os.path.join(scripts, 'vae.py'),
                   ['--mode', 'vae', '--epochs', str(args.vae_epochs),
                    '--device', args.device], 'VAE')

    # 2. cVAE
    if not args.skip_cvae:
        run_method(os.path.join(scripts, 'vae.py'),
                   ['--mode', 'cvae', '--epochs', str(args.vae_epochs),
                    '--device', args.device], 'cVAE')

    # 3. Latent Diffusion (requires trained VAE)
    run_method(os.path.join(scripts, 'diffusion.py'),
               ['--epochs', str(args.diffusion_epochs),
                '--device', args.device], 'Latent Diffusion')

    # 4. RL Fine-Tuning (requires trained VAE)
    run_method(os.path.join(scripts, 'rl_finetune.py'),
               ['--base-model', 'vae', '--epochs', str(args.rl_epochs),
                '--device', args.device], 'RL Fine-Tuning')

    # 5. CGLF (requires trained VAE encoder)
    run_method(os.path.join(scripts, 'cglf.py'),
               ['--stage1-epochs', str(args.cglf_s1_epochs),
                '--stage2-epochs', str(args.cglf_s2_epochs),
                '--device', args.device], 'CGLF')

    # 6. CRAVE (requires trained RL-VAE)
    run_method(os.path.join(scripts, 'crave_gen.py'),
               ['--epochs', str(args.crave_epochs),
                '--device', args.device], 'CRAVE')

    # 7. Evaluate all
    log.info('=== Final Comparative Evaluation ===')
    evaluate_all(args.device)


if __name__ == '__main__':
    main()
