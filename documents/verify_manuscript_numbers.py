#!/usr/bin/env python3
"""Phase A verifier: re-derives a sample of manuscript_ground_truth.json entries
directly from their source files and flags any mismatch. Run before Phase C."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = json.load(open(Path(__file__).resolve().parent / 'manuscript_ground_truth.json'))

errors = []
checks = 0


def check(label, actual, expected, tol=1e-3):
    global checks
    checks += 1
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        ok = abs(actual - expected) <= tol
    else:
        ok = actual == expected
    status = 'OK' if ok else 'MISMATCH'
    print(f'[{status}] {label}: ledger={expected} raw={actual}')
    if not ok:
        errors.append(label)


def main():
    d = json.load(open(ROOT / 'modeling/validation/multiseed/logs/multiseed_summary.json'))
    check('pgcl_core.global_spearman_rho.mean', d['global_spearman_rho']['mean'],
          LEDGER['pgcl_core_3seed']['global_spearman_rho']['mean'], tol=1e-4)
    check('pgcl_core.cliff_accuracy.mean', d['cliff_accuracy']['mean'],
          LEDGER['pgcl_core_3seed']['cliff_accuracy']['mean'], tol=1e-4)
    check('pgcl_core.ndcg_100.mean', d['ndcg_100']['mean'],
          LEDGER['pgcl_core_3seed']['ndcg_100']['mean'], tol=1e-4)

    fp = json.load(open(ROOT / 'modeling/baselines/results/pgcl_fullpool_ndcg.json'))
    check('fullpool.spearman_rho.mean', fp['spearman_rho']['mean'],
          LEDGER['pgcl_fullpool_ndcg']['spearman_rho']['mean'], tol=1e-4)

    ab = json.load(open(ROOT / 'modeling/validation/ablations/ablation_results.json'))
    check('ablation.dim_256.ndcg_100', ab['dim_256']['metrics']['ndcg_100'],
          LEDGER['ablations_8config']['configs']['dim_256']['ndcg_100'], tol=1e-4)
    check('ablation.no_potency.rho', ab['no_potency']['metrics']['global_spearman_rho'],
          LEDGER['ablations_8config']['configs']['no_potency']['rho'], tol=1e-4)

    gm = json.load(open(ROOT / 'generative/multiseed/aggregate/multiseed_summary.json'))
    crave_vals = gm['methods']['crave_filtered']['oscc_potency']['values']
    check('generative.crave_filtered.per_seed_potency', crave_vals,
          LEDGER['generative_multiseed_5seed']['methods']['crave_filtered']['per_seed_potency'])
    check('generative.crave_filtered.mean', gm['methods']['crave_filtered']['oscc_potency']['mean'],
          0.7712, tol=1e-3)

    bio = json.load(open(ROOT.parent / 'BioactivityProfiling_Generative/profiling_summary.json'))
    check('phase4.n_unique_molecules', bio['n_unique_molecules'],
          LEDGER['phase4_bioactivity_profiling']['n_unique_molecules'])
    check('phase4.safety_pass', bio['safety_summary']['pass'],
          LEDGER['phase4_bioactivity_profiling']['safety_summary']['pass'])

    rep = json.load(open(ROOT / 'repurposing/results/repurposing_summary.json'))
    check('phase6.n_total_compounds', rep['n_total_compounds'],
          LEDGER['phase6_drug_repurposing']['n_total_compounds'])
    check('phase6.top1.chembl_id', rep['top10'][0]['compound_chembl_id'],
          LEDGER['phase6_drug_repurposing']['top3'][0]['chembl_id'])

    dock = json.load(open(ROOT / 'docking/results/docking_summary.json'))
    check('phase7a.successful_docks', dock['successful_docks'],
          LEDGER['phase7_docking']['phase7a_single_target']['successful'])
    pdb_ids = {t['target']: t['pdb_id'] for t in dock['per_target']}
    check('phase7.pdb_ids', pdb_ids, LEDGER['phase7_docking']['targets_confirmed_8'])

    admet = json.load(open(ROOT / 'admet/results/admet_summary.json'))
    check('phase8.triple_validated_elite', admet['triple_validated_elite']['count'],
          LEDGER['phase8_admet']['triple_validated_elite']['count'])

    lo = json.load(open(ROOT / 'leadopt/results/leadopt_summary.json'))
    check('phase9b.total_unique_analogs', lo['total_unique_analogs'],
          LEDGER['phase9b_lead_optimization']['total_unique_analogs'])

    import pandas as pd
    mr = pd.read_csv(ROOT / 'final_ranking/results/master_ranked_all.tsv', sep='\t', low_memory=False)
    check('phase10.n_total_ranked', len(mr), LEDGER['phase10_final_ranking']['n_total_ranked'])
    tiers = mr['validation_tier'].value_counts().to_dict()
    check('phase10.tier1_count', tiers.get('Tier-1 (fully validated)'),
          LEDGER['phase10_final_ranking']['validation_tier_distribution']['Tier-1_fully_validated'])
    top1 = mr.sort_values('genesis_score', ascending=False).iloc[0]
    check('phase10.top1.cpd_id', top1['cpd_id'],
          LEDGER['phase10_final_ranking']['top3_overall'][0]['cpd_id'])

    print(f'\n{checks} checks run, {len(errors)} mismatches.')
    if errors:
        print('MISMATCHES:', errors)
        raise SystemExit(1)
    print('ALL CHECKS PASSED — ledger verified against raw source files.')


if __name__ == '__main__':
    main()
