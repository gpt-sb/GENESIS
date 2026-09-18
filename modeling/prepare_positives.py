#!/usr/bin/env python3
"""
Phase 1.1 — Extract ChEMBL positive cohort with potency grades.

Filters All_Compounds_Unq_Std.tsv to ChEMBL-sourced compounds with
potency_class_num in {1,2,3}. Computes pchembl_filled where missing.
Output: GENESIS/modeling/positives.tsv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent        # GENESIS/modeling/
GENESIS_DIR = SCRIPT_DIR.parent                     # GENESIS/

INPUT = GENESIS_DIR / 'data' / 'All_Compounds_Unq_Std.tsv'
OUTPUT = SCRIPT_DIR / 'positives.tsv'

KEEP_COLS = [
    'compound_chembl_id', 'smiles', 'potency_class_num', 'pchembl', 'ic50_nM',
    'target_name', 'gene_symbols', 'cell_line', 'IsAnticancer', 'max_phase',
    'in_plos', 'qed_weighted', 'molecular_weight', 'alogp',
]


def main():
    log.info('Reading %s', INPUT)
    df = pd.read_csv(INPUT, sep='\t', low_memory=False)
    log.info('Total rows: %d', len(df))

    # Filter: source_file contains "ChEMBL" AND potency_class_num in {1,2,3}
    mask_source = df['source_file'].str.contains('ChEMBL', na=False)
    mask_potency = df['potency_class_num'].isin([1, 2, 3])
    pos = df[mask_source & mask_potency].copy()
    log.info('Positives after filter: %d', len(pos))

    # Select columns
    pos = pos[KEEP_COLS].copy()

    # Ensure numeric types
    for col in ['potency_class_num', 'pchembl', 'ic50_nM', 'IsAnticancer',
                'max_phase', 'in_plos', 'qed_weighted', 'molecular_weight', 'alogp']:
        pos[col] = pd.to_numeric(pos[col], errors='coerce')

    # Compute pchembl_filled
    # Where pchembl is NaN but ic50_nM is available: pchembl = 9 - log10(ic50_nM)
    pos['pchembl_filled'] = pos['pchembl'].copy()
    need_fill = pos['pchembl_filled'].isna() & pos['ic50_nM'].notna() & (pos['ic50_nM'] > 0)
    pos.loc[need_fill, 'pchembl_filled'] = 9.0 - np.log10(pos.loc[need_fill, 'ic50_nM'])

    # Handle ic50_nM == 0 (leave pchembl_filled as NaN)
    zero_ic50 = pos['ic50_nM'] == 0.0
    log.info('Compounds with ic50_nM == 0: %d (pchembl_filled left NaN)', zero_ic50.sum())

    # Log stats
    log.info('--- Potency distribution ---')
    for cls in [1, 2, 3]:
        n = (pos['potency_class_num'] == cls).sum()
        label = {1: 'active', 2: 'potent', 3: 'very_potent'}[cls]
        log.info('  %s (class %d): %d compounds', label, cls, n)

    log.info('--- pchembl_filled coverage ---')
    filled = pos['pchembl_filled'].notna().sum()
    log.info('  Has pchembl_filled: %d / %d (%.1f%%)', filled, len(pos), 100 * filled / len(pos))
    log.info('  pchembl_filled stats: mean=%.2f, median=%.2f, std=%.2f',
             pos['pchembl_filled'].mean(), pos['pchembl_filled'].median(), pos['pchembl_filled'].std())

    log.info('--- Benchmark sets ---')
    log.info('  in_plos == 1: %d', (pos['in_plos'] == 1).sum())
    log.info('  max_phase == 4: %d', (pos['max_phase'] == 4.0).sum())

    # Save
    pos.to_csv(OUTPUT, sep='\t', index=False)
    log.info('Saved %d rows × %d cols → %s', len(pos), len(pos.columns), OUTPUT)


if __name__ == '__main__':
    main()
