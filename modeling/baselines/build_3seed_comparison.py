#!/usr/bin/env python3
"""Build the final 3-seed-vs-3-seed PGCL vs Flat SupCon comparison table,
replacing the single-seed comparison.json with a properly powered result."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

pgcl = json.load(open(BASE / 'validation/multiseed/logs/multiseed_summary.json'))
flat = json.load(open(BASE / 'baselines/results/flat_supcon/multiseed/logs/multiseed_summary.json'))

metric_map = [
    ('VP Recall@100', 'vp_recall_100'),
    ('Potency Spearman ρ', 'global_spearman_rho'),
    ('Cliff Pair Accuracy', 'cliff_accuracy'),
    ('Target-Switch AUROC', 'target_switch_auc_roc'),
    ('NDCG@100 (potency)', 'ndcg_100'),
]

comparison = {
    'method_a': 'PGCL (Ordinal SupCon) [ours] -- 3-seed mean (42/123/456)',
    'method_b': 'Standard SupCon (flat weights) [baseline] -- 3-seed mean (42/123/456)',
    'metrics': {},
}

print(f'{"Metric":<25} {"PGCL (3-seed)":>16} {"FlatSupCon (3-seed)":>20} {"Δ":>9} {"PGCL wins?":>11}')
print('-' * 85)
wins = 0
total = 0
for label, key in metric_map:
    p = pgcl[key]
    f = flat[key]
    delta = f['mean'] - p['mean']
    pgcl_wins = delta < 0
    wins += int(pgcl_wins)
    total += 1
    comparison['metrics'][label] = {
        'PGCL_mean': p['mean'], 'PGCL_std': p['std'],
        'FlatSupCon_mean': f['mean'], 'FlatSupCon_std': f['std'],
        'delta': delta,
        'pct_change': 100 * delta / (abs(p['mean']) + 1e-12),
        'PGCL_wins': pgcl_wins,
    }
    print(f'{label:<25} {p["mean"]:.4f}±{p["std"]:.4f}  {f["mean"]:.4f}±{f["std"]:.4f}      {delta:+.4f}  {"✓" if pgcl_wins else "✗":>10}')

print('-' * 85)
print(f'PGCL wins: {wins}/{total} metrics (3-seed vs 3-seed, both using corrected checkpoint-selection)')

comparison['summary'] = f'PGCL wins {wins}/{total} metrics'
out_path = BASE / 'baselines/results/flat_supcon/comparison_3seed.json'
with open(out_path, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f'\nSaved -> {out_path}')
