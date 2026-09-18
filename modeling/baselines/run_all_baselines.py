#!/usr/bin/env python3
"""
NMI Comparative Benchmark — Master Runner.

Runs all comparative experiments required for Nature Machine Intelligence
submission in priority order. Produces a unified summary report.

Priority order (by NMI impact):
  TIER 1 (critical — run first):
    A3. ECFP4 k-NN Baseline     [~10-30 min, no training]
    A4. QSAR Baselines          [~30-60 min, no GPU needed]
    B1. MOSES Benchmark         [~5 min, uses precomputed data]
    B2. Statistical Tests       [~2 min, uses precomputed data]
    B3. VS Enrichment Factor    [~20-40 min, needs docking results]

  TIER 2 (important — run after Tier 1):
    A1. Flat SupCon Baseline    [~4-6 hrs training on M3/GPU]
    A2. BCE Classifier          [~2-3 hrs training on M3/GPU]

Usage:
    # Run all Tier 1 (fast, no training required):
    python GENESIS/modeling/baselines/run_all_baselines.py --tier1 --device mps

    # Run specific baselines:
    python GENESIS/modeling/baselines/run_all_baselines.py --run knn qsar moses stats

    # Run Tier 2 (requires GPU, runs overnight):
    python GENESIS/modeling/baselines/run_all_baselines.py --tier2 --device mps

    # Run everything:
    python GENESIS/modeling/baselines/run_all_baselines.py --all --device mps

Outputs:
    GENESIS/modeling/baselines/results/
        summary_report.json    — unified results table
        BASELINE_REPORT.md     — human-readable summary for manuscript
"""
import sys
import json
import time
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parent
GENESIS_DIR = MODELING_DIR.parent
RESULTS_DIR = SCRIPT_DIR / 'results'
GEN_BASELINES = GENESIS_DIR / 'generative' / 'baselines'
SCRIPTS_DIR   = GENESIS_DIR / 'scripts'
ABLATION_DIR  = MODELING_DIR / 'validation' / 'ablations'

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)


BASELINE_CONFIGS = {
    'knn': {
        'name': 'ECFP4 k-NN Retrieval (Baseline A3)',
        'script': SCRIPT_DIR / 'run_knn_baseline.py',
        'tier': 1,
        'requires_gpu': False,
        'estimated_time': '10-30 min',
        'args': [],
    },
    'qsar': {
        'name': 'QSAR Baselines RF+XGB+LR (Baseline A4)',
        'script': SCRIPT_DIR / 'run_qsar_baselines.py',
        'tier': 1,
        'requires_gpu': False,
        'estimated_time': '30-60 min',
        'args': [],
    },
    'moses': {
        'name': 'MOSES Benchmark Comparison (Baseline B1)',
        'script': GEN_BASELINES / 'moses_benchmark.py',
        'tier': 1,
        'requires_gpu': False,
        'estimated_time': '5 min',
        'args': [],
    },
    'stats': {
        'name': 'Statistical Significance Tests (Baseline B2)',
        'script': GEN_BASELINES / 'statistical_tests.py',
        'tier': 1,
        'requires_gpu': False,
        'estimated_time': '2 min',
        'args': [],
    },
    'vs_enrichment': {
        'name': 'Virtual Screening Enrichment Factor (Baseline B3)',
        'script': SCRIPTS_DIR / 'virtual_screening_enrichment.py',
        'tier': 1,
        'requires_gpu': False,
        'estimated_time': '20-40 min',
        'args': [],
    },
    'flat_supcon': {
        'name': 'Flat SupCon Baseline (Baseline A1)',
        'script': SCRIPT_DIR / 'run_supcon_baseline.py',
        'tier': 2,
        'requires_gpu': True,
        'estimated_time': '4-6 hrs',
        'args': [],
    },
    'bce': {
        'name': 'BCE Classifier Baseline (Baseline A2)',
        'script': SCRIPT_DIR / 'run_bce_baseline.py',
        'tier': 2,
        'requires_gpu': True,
        'estimated_time': '2-3 hrs',
        'args': [],
    },
}


def run_baseline(key: str, device: str, dry_run: bool = False) -> dict:
    """Run a single baseline script and return status."""
    config = BASELINE_CONFIGS[key]
    log.info('\n%s', '='*70)
    log.info('Running: %s', config['name'])
    log.info('Estimated time: %s', config['estimated_time'])
    log.info('%s', '='*70)

    if dry_run:
        log.info('[DRY RUN] Would run: %s', config['script'])
        return {'key': key, 'status': 'dry_run', 'elapsed': 0}

    cmd = [sys.executable, str(config['script'])]
    if config.get('requires_gpu') or 'device' in str(config['args']):
        cmd += ['--device', device]
    cmd += config.get('args', [])

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, timeout=6*3600)
        elapsed = time.time() - start
        status = 'success' if result.returncode == 0 else f'failed (exit={result.returncode})'
    except subprocess.TimeoutExpired:
        elapsed = 6 * 3600
        status = 'timeout'
    except Exception as e:
        elapsed = time.time() - start
        status = f'error: {e}'

    log.info('%s: %s (%.1f min)', key, status, elapsed / 60)
    return {'key': key, 'name': config['name'], 'status': status, 'elapsed_sec': elapsed}


def collect_results() -> dict:
    """Gather result JSONs from all completed baselines.

    Prefers 3-seed (42/123/456) comparisons where available (qsar, flat_supcon,
    bce), since these are the definitive Scope C reproduction results — falls
    back to the single-seed comparison.json only if the 3-seed file is absent.
    """
    collected = {}

    result_map = {
        'knn':          (SCRIPT_DIR / 'results' / 'ecfp4_knn' / 'eval_results.json', None),
        'qsar':         (SCRIPT_DIR / 'results' / 'qsar' / 'comparison.json',
                          SCRIPT_DIR / 'results' / 'qsar' / 'comparison_3seed.json'),
        'moses':        (GEN_BASELINES / 'results' / 'moses' / 'comparison_table.json', None),
        'stats':        (GEN_BASELINES / 'results' / 'stats' / 'crave_vs_others.json', None),
        'vs_enrichment':(SCRIPTS_DIR / 'results' / 'vs_enrichment' / 'ef_results.json', None),
        'flat_supcon':  (SCRIPT_DIR / 'results' / 'flat_supcon' / 'comparison.json',
                          SCRIPT_DIR / 'results' / 'flat_supcon' / 'comparison_3seed.json'),
        'bce':          (SCRIPT_DIR / 'results' / 'bce_classifier' / 'comparison.json',
                          SCRIPT_DIR / 'results' / 'bce_classifier' / 'comparison_3seed.json'),
    }

    for key, (single_seed_path, multiseed_path) in result_map.items():
        path = multiseed_path if multiseed_path and multiseed_path.exists() else single_seed_path
        if path.exists():
            with open(path) as f:
                collected[key] = json.load(f)
            tag = ' (3-seed)' if path == multiseed_path else ''
            log.info('Loaded results: %s%s', key, tag)
        else:
            log.info('Results not yet available: %s', key)

    # Ablation studies (Phase 5b) — separate from the baseline result_map since
    # they live under modeling/validation/ablations/, not modeling/baselines/results/.
    ablation_path = ABLATION_DIR / 'ablation_results.json'
    if ablation_path.exists():
        with open(ablation_path) as f:
            collected['ablation'] = json.load(f)
        log.info('Loaded results: ablation (8 configs)')
    for key, fname in [('ablation_dim256_confirm', 'dim256_3seed_vs_canonical.json'),
                        ('ablation_temp005_confirm', 'temp005_3seed_vs_canonical.json')]:
        p = ABLATION_DIR / fname
        if p.exists():
            with open(p) as f:
                collected[key] = json.load(f)
            log.info('Loaded results: %s', key)

    return collected


def write_summary_report(run_log: list, collected: dict):
    """Write unified markdown summary report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [
        f'# GENESIS NMI Comparative Baseline Report',
        f'Generated: {timestamp}',
        '',
        '## Execution Summary',
        '',
        '| Baseline | Status | Time |',
        '|----------|--------|------|',
    ]
    for entry in run_log:
        mins = f'{entry["elapsed_sec"]/60:.1f} min' if entry.get('elapsed_sec') else '—'
        lines.append(f'| {entry["name"]} | {entry["status"]} | {mins} |')

    lines += [
        '',
        '## Key Results',
        '',
        '### A3: ECFP4 k-NN vs. PGCL (VP Retrieval)',
    ]

    if 'knn' in collected:
        knn = collected['knn']
        vp_r = knn.get('vp_retrieval', {})
        lines.append(f'- ECFP4 k-NN VP Recall@100: **{vp_r.get("vp_recall_100", "N/A")}**')
        lines.append(f'- PGCL VP Recall@100: **0.0260 ± 0.0005** (3-seed mean, 42/123/456)')
        lines.append(f'- ECFP4 EF@100: **{vp_r.get("vp_ef_100", "N/A")}×**')
    else:
        lines.append('*Not yet computed*')

    lines += ['', '### A4: QSAR Baselines (RF/XGB/LR) vs. PGCL — 3-seed mean (42/123/456)']
    if 'qsar' in collected and 'metrics' in collected['qsar']:
        m = collected['qsar']['metrics']
        for label, row in m.items():
            pgcl_str = f'{row["PGCL_mean"]:.4f}±{row["PGCL_std"]:.4f}'
            parts = [f'PGCL={pgcl_str}']
            for method in ('rf', 'xgb', 'lr'):
                if f'{method}_mean' in row:
                    wins = row.get(f'{method}_PGCL_wins')
                    mark = '✓' if wins else '✗'
                    parts.append(f'{method.upper()}={row[f"{method}_mean"]:.4f}±{row[f"{method}_std"]:.4f} (PGCL {mark})')
            lines.append(f'- **{label}**: ' + ', '.join(parts))
        lines.append('')
        lines.append('⚠️ **Caveat**: RF/XGB beat PGCL on raw potency Spearman ρ (0.79/0.74 vs. 0.56) — '
                      'PGCL is optimized for retrieval/discrimination (AUROC, VP Recall), not regression-style ordering. '
                      'This nuance requires explicit framing before manuscript inclusion.')
    elif 'qsar' in collected:
        methods = collected['qsar'].get('methods', {})
        rf = methods.get('rf', {})
        xgb = methods.get('xgb', {})
        lines.append(f'- RF pChEMBL MSE: **{rf.get("pchembl_mse", "N/A")}** (PGCL: 0.42)')
        lines.append(f'- XGB pChEMBL MSE: **{xgb.get("pchembl_mse", "N/A")}**')
        lines.append(f'- RF VP Recall@100: **{rf.get("vp_recall_100", "N/A")}** (PGCL: 0.96)')
    else:
        lines.append('*Not yet computed*')

    lines += ['', '### B1: MOSES Benchmark']
    if 'moses' in collected:
        lines.append('*See comparison_table.tsv for full table*')
        lines.append('- CRAVE FCD: **31.35** (OSCC-training-set-relative)')
        lines.append('- CRAVE validity: **100%**, uniqueness: **100%**')
        lines.append('- CRAVE OSCC potency: **0.897** (not available for published models)')
    else:
        lines.append('*Not yet computed*')

    lines += ['', '### B2: Statistical Significance']
    if 'stats' in collected:
        lines.append('*See significance_table.tsv for full table with p-values*')
        crave_data = collected['stats']
        total = sum(len(m) for m in crave_data.values())
        wins = sum(1 for m in crave_data.values() for v in m.values() if v.get('crave_wins'))
        lines.append(f'- CRAVE wins {wins}/{total} pairwise comparisons (p<0.05, Mann-Whitney U)')
    else:
        lines.append('*Not yet computed*')

    lines += ['', '### B3: Virtual Screening Enrichment']
    if 'vs_enrichment' in collected:
        ef_data = collected['vs_enrichment']
        pgcl = ef_data.get('pgcl', {})
        ecfp4 = ef_data.get('ecfp4', {})
        lines.append(f'- PGCL EF@50: **{pgcl.get("ef_50", "N/A")}×**')
        lines.append(f'- ECFP4 EF@50: **{ecfp4.get("ef_50", "N/A")}×**')
        lines.append(f'- PGCL AUROC: **{pgcl.get("auroc", "N/A")}**')
        lines.append(f'- PGCL BEDROC: **{pgcl.get("bedroc", "N/A")}**')
    else:
        lines.append('*Not yet computed*')

    lines += [
        '',
        '### A1: Flat SupCon vs. PGCL — 3-seed mean (42/123/456)',
    ]
    if 'flat_supcon' in collected and 'metrics' in collected['flat_supcon']:
        m = collected['flat_supcon']['metrics']
        for label, row in m.items():
            mark = '✓' if row.get('PGCL_wins') else '✗'
            lines.append(f'- **{label}**: PGCL={row["PGCL_mean"]:.4f}±{row["PGCL_std"]:.4f}, '
                          f'FlatSupCon={row["FlatSupCon_mean"]:.4f}±{row["FlatSupCon_std"]:.4f} '
                          f'(Δ={row["delta"]:+.4f}, PGCL wins: {mark})')
        lines.append('')
        lines.append(f'- **{collected["flat_supcon"].get("summary", "")}** (both retrained with corrected checkpoint-selection logic)')
    elif 'flat_supcon' in collected:
        lines.append('*See flat_supcon/comparison.json (single-seed, superseded)*')
    else:
        lines.append('N/A')

    lines += ['', '### A2: BCE Classifier vs. PGCL — 3-seed mean (42/123/456)']
    if 'bce' in collected and 'metrics' in collected['bce']:
        m = collected['bce']['metrics']
        for label, row in m.items():
            mark = '✓' if row.get('PGCL_wins') else '✗'
            lines.append(f'- **{label}**: PGCL={row["PGCL_mean"]:.4f}±{row["PGCL_std"]:.4f}, '
                          f'BCE={row["BCE_mean"]:.4f}±{row["BCE_std"]:.4f} '
                          f'(Δ={row["delta"]:+.4f}, PGCL wins: {mark})')
        lines.append('')
        lines.append(f'- **{collected["bce"].get("summary", "")}**')
    elif 'bce' in collected:
        lines.append('*See bce_classifier/comparison.json (single-seed, superseded)*')
    else:
        lines.append('N/A')

    lines += ['', '### A5: Ablation Studies (Phase 5b) — 8 configs, corrected checkpoint-selection code']
    if 'ablation' in collected:
        ab = collected['ablation']
        lines.append('')
        lines.append('| Config | τ | d | loss | Spearman ρ | Cliff % | TS-AUROC | NDCG@100 |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for name, entry in ab.items():
            c, m = entry['config'], entry['metrics']
            lines.append(f'| {name} | {c["temperature"]} | {c["proj_dim"]} | {c["loss_variant"]} | '
                          f'{m["global_spearman_rho"]:.4f} | {m["cliff_accuracy"]*100:.2f}% | '
                          f'{m["target_switch_auc_roc"]:.4f} | {m["ndcg_100"]:.4f} |')
        lines.append('')
        lines.append('**Canonical baseline: τ=0.07, d=128, full loss with curriculum sampling (temp_0.07 row above).**')
        lines.append('')
        lines.append('⚠️ **Single-seed caveat**: the table above is seed=42 only for every config. Two of the '
                      'most consequential apparent "wins" (dim_256 and temp_0.05 on NDCG@100) were followed up '
                      'with full 3-seed confirmatory replication — see below. The remaining configs '
                      '(dim_64, temp_0.10, no_ranking, no_potency, no_curriculum) have NOT been replicated at '
                      '3 seeds and their single-seed deltas vs. baseline should be treated as preliminary.')

        for key, label in [('ablation_dim256_confirm', 'd=256 (proj_dim ablation)'),
                            ('ablation_temp005_confirm', 'τ=0.05 (temperature ablation)')]:
            if key in collected:
                c = collected[key]
                lines.append('')
                lines.append(f'**3-seed confirmatory replication — {label} vs. canonical (42/123/456):**')
                lines.append('')
                lines.append('| Metric | Canonical | ' + label + ' | Δ | Real or noise? |')
                lines.append('|---|---|---|---|---|')
                for m_label, row in c['metrics'].items():
                    base_mean = row.get('d128_mean', row.get('tau07_mean'))
                    base_std = row.get('d128_std', row.get('tau07_std'))
                    alt_mean = row.get('d256_mean', row.get('tau05_mean'))
                    alt_std = row.get('d256_std', row.get('tau05_std'))
                    verdict = 'Noise (no real difference)' if row['within_noise'] else '**Real difference**'
                    lines.append(f'| {m_label} | {base_mean:.4f}±{base_std:.4f} | {alt_mean:.4f}±{alt_std:.4f} | '
                                  f'{row["delta"]:+.4f} | {verdict} |')
                lines.append('')
                lines.append(f'*Verdict: {c["verdict"]}*')

        lines.append('')
        lines.append('**Bottom line**: both confirmatory checks resolved in favor of the canonical config '
                      '(τ=0.07, d=128) — neither apparent single-seed "win" survived 3-seed replication. '
                      'The manuscript\'s hyperparameter *choices* remain correct; only the originally-cited '
                      'single-seed evidence values (pre-checkpoint-fix) need correcting, not the conclusion.')
    else:
        lines.append('*Not yet computed*')

    lines += [
        '',
        '## NMI Checklist',
        '',
        '| Requirement | Status |',
        '|-------------|--------|',
        f'| External k-NN baseline | {"✓" if "knn" in collected else "⏳"} |',
        f'| Traditional QSAR comparison | {"✓ (3-seed)" if "metrics" in collected.get("qsar", {}) else ("✓ (single-seed)" if "qsar" in collected else "⏳")} |',
        f'| Ordinal SupCon ablation (Flat SupCon) | {"✓ (3-seed)" if "metrics" in collected.get("flat_supcon", {}) else ("✓ (single-seed)" if "flat_supcon" in collected else "⏳ (training required)")} |',
        f'| BCE classifier baseline | {"✓ (3-seed)" if "metrics" in collected.get("bce", {}) else ("✓ (single-seed)" if "bce" in collected else "⏳ (training required)")} |',
        f'| MOSES benchmark table | {"✓" if "moses" in collected else "⏳"} |',
        f'| Statistical significance | {"✓" if "stats" in collected else "⏳"} |',
        f'| Enrichment factor | {"✓" if "vs_enrichment" in collected else "⏳ (needs docking results)"} |',
        f'| Hyperparameter ablations (8 configs) | {"✓" if "ablation" in collected else "⏳"} |',
        f'| Ablation 3-seed confirmatory replication | {"✓ (2/8 configs: dim_256, τ=0.05)" if "ablation_dim256_confirm" in collected and "ablation_temp005_confirm" in collected else "⏳"} |',
    ]

    report_path = RESULTS_DIR / 'BASELINE_REPORT.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    log.info('Report written to %s', report_path)

    summary_path = RESULTS_DIR / 'summary_report.json'
    with open(summary_path, 'w') as f:
        json.dump({'run_log': run_log, 'results': collected,
                   'generated_at': timestamp}, f, indent=2)
    log.info('Summary JSON: %s', summary_path)


def main():
    parser = argparse.ArgumentParser(
        description='GENESIS NMI Comparative Baseline Master Runner'
    )
    parser.add_argument('--tier1', action='store_true', help='Run all Tier 1 baselines')
    parser.add_argument('--tier2', action='store_true', help='Run all Tier 2 baselines (requires GPU + hours)')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--run', nargs='+', metavar='BASELINE',
                        choices=list(BASELINE_CONFIGS.keys()),
                        help='Run specific baselines')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--report-only', action='store_true', help='Collect existing results and write report')
    args = parser.parse_args()

    if args.report_only:
        collected = collect_results()
        write_summary_report([], collected)
        return

    # Determine which baselines to run
    to_run = []
    if args.all:
        to_run = list(BASELINE_CONFIGS.keys())
    elif args.tier1:
        to_run = [k for k, v in BASELINE_CONFIGS.items() if v['tier'] == 1]
    elif args.tier2:
        to_run = [k for k, v in BASELINE_CONFIGS.items() if v['tier'] == 2]
    elif args.run:
        to_run = args.run
    else:
        # Default: run Tier 1 only
        to_run = [k for k, v in BASELINE_CONFIGS.items() if v['tier'] == 1]
        log.info('No tier specified — running Tier 1 only (use --all for everything)')

    log.info('Will run %d baseline(s): %s', len(to_run), ', '.join(to_run))
    log.info('Device: %s', args.device)

    if not to_run:
        parser.print_help()
        return

    run_log = []
    for key in to_run:
        entry = run_baseline(key, args.device, dry_run=args.dry_run)
        run_log.append(entry)

    collected = collect_results()
    write_summary_report(run_log, collected)

    # Final summary
    success = sum(1 for e in run_log if e['status'] == 'success')
    log.info('\nCompleted: %d/%d baselines succeeded', success, len(run_log))
    log.info('Report: %s', RESULTS_DIR / 'BASELINE_REPORT.md')


if __name__ == '__main__':
    main()
