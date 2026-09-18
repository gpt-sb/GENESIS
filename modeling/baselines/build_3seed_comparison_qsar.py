#!/usr/bin/env python3
"""Build the final 3-seed-vs-3-seed PGCL vs QSAR (RF/XGB/LR) comparison table,
replacing the single-seed comparison.json with a properly powered result."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

pgcl_full = json.load(open(BASE / 'validation/multiseed/logs/multiseed_summary.json'))
qsar = json.load(open(BASE / 'baselines/results/qsar/multiseed/logs/multiseed_summary.json'))

# PGCL reference (n=3 seeds 42/123/456), pchembl_mse has no PGCL equivalent (not a regressor)
pgcl = {
    'pchembl_spearman_rho': pgcl_full['global_spearman_rho'],
    'binary_auroc':         pgcl_full['target_switch_auc_roc'],
    'vp_recall_100':        pgcl_full['vp_recall_100'],
}

metric_map = [
    ('Potency Spearman ρ', 'pchembl_spearman_rho', True),
    ('Binary AUROC',       'binary_auroc',          True),
    ('VP Recall@100',      'vp_recall_100',         True),
]

methods = ['rf', 'xgb', 'lr']
comparison = {
    'method_a': 'PGCL (Ordinal SupCon) [ours] -- 3-seed mean (42/123/456)',
    'method_b': 'QSAR (RF/XGB/LR on ECFP4) [baseline] -- 3-seed mean (42/123/456)',
    'metrics': {},
}

header = f'{"Metric":<25} {"PGCL (3-seed)":>16} ' + ' '.join(f'{m.upper() + " (3-seed)":>20}' for m in methods)
print(header)
print('-' * (25 + 17 + 21 * len(methods)))

wins = {m: 0 for m in methods}
total = 0
for label, key, higher_better in metric_map:
    p = pgcl[key]
    total += 1
    row = f'{label:<25} {p["mean"]:.4f}±{p["std"]:.4f}  '
    comparison['metrics'][label] = {'PGCL_mean': p['mean'], 'PGCL_std': p['std']}
    for m in methods:
        mkey = f'{m}_{key}'
        if mkey not in qsar:
            row += f'{"N/A":>20} '
            continue
        b = qsar[mkey]
        delta = b['mean'] - p['mean']
        pgcl_wins = delta < 0 if higher_better else delta > 0
        wins[m] += int(pgcl_wins)
        comparison['metrics'][label][f'{m}_mean'] = b['mean']
        comparison['metrics'][label][f'{m}_std'] = b['std']
        comparison['metrics'][label][f'{m}_delta'] = delta
        comparison['metrics'][label][f'{m}_PGCL_wins'] = pgcl_wins
        row += f'{b["mean"]:.4f}±{b["std"]:.4f} ({"✓" if pgcl_wins else "✗"})'.rjust(20) + ' '
    print(row)

print('-' * (25 + 17 + 21 * len(methods)))
for m in methods:
    print(f'PGCL wins vs {m.upper()}: {wins[m]}/{total} metrics')

comparison['summary'] = {m: f'PGCL wins {wins[m]}/{total} metrics' for m in methods}
out_path = BASE / 'baselines/results/qsar/comparison_3seed.json'
with open(out_path, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f'\nSaved -> {out_path}')
