#!/usr/bin/env python3
"""
Phase 1.5 — Leakage-safe split creation.

Groups compounds by (scaffold, target_family), then splits into
train/val/test ensuring no scaffold group appears in multiple splits.
Reserves PLOS and approved-drug benchmarks.
Output: GENESIS/modeling/splits.json
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent        # GENESIS/modeling/
GENESIS_DIR = SCRIPT_DIR.parent                     # GENESIS/

POS_FILE = SCRIPT_DIR / 'positives.tsv'
NEG_FILE = SCRIPT_DIR / 'negatives.tsv'
INDEX_FILE = SCRIPT_DIR / 'compound_index.pkl'
SCAFFOLD_FILE = SCRIPT_DIR / 'scaffold_groups.pkl'
FAMILY_FILE = SCRIPT_DIR / 'target_families.pkl'
OUTPUT = SCRIPT_DIR / 'splits.json'

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def main():
    np.random.seed(SEED)

    # Load data
    pos = pd.read_csv(POS_FILE, sep='\t', low_memory=False)
    neg = pd.read_csv(NEG_FILE, sep='\t', low_memory=False)

    with open(INDEX_FILE, 'rb') as f:
        compound_index = pickle.load(f)

    with open(SCAFFOLD_FILE, 'rb') as f:
        scaffold_data = pickle.load(f)
    cid_to_scaffold = scaffold_data['compound_to_scaffold']

    with open(FAMILY_FILE, 'rb') as f:
        cid_to_family = pickle.load(f)

    log.info('Compound index: %d entries', len(compound_index))

    # --- Reserve benchmark sets (removed from train/val/test entirely) ---
    plos_ids = set(pos.loc[pos['in_plos'] == 1, 'compound_chembl_id'].dropna())
    approved_ids = set(pos.loc[pos['max_phase'] == 4.0, 'compound_chembl_id'].dropna())
    benchmark_ids = plos_ids | approved_ids
    log.info('PLOS benchmark: %d, Approved benchmark: %d, Union: %d',
             len(plos_ids), len(approved_ids), len(benchmark_ids))

    # --- Get all positive IDs for splitting (exclude benchmarks) ---
    pos_ids_all = set(pos['compound_chembl_id'].dropna())
    pos_ids_splittable = pos_ids_all - benchmark_ids
    log.info('Positives for splitting: %d (excluded %d benchmarks)',
             len(pos_ids_splittable), len(benchmark_ids))

    # --- Group positives by scaffold ---
    scaffold_groups = defaultdict(list)
    # sorted(): pos_ids_splittable is a set, whose iteration order is
    # randomized per-process by PYTHONHASHSEED — without sorting, this
    # made np.random.shuffle(scaffolds) below non-reproducible across runs
    # even with a fixed SEED.
    for cid in sorted(pos_ids_splittable):
        scaffold = cid_to_scaffold.get(cid, 'UNKNOWN')
        scaffold_groups[scaffold].append(cid)

    scaffolds = list(scaffold_groups.keys())
    np.random.shuffle(scaffolds)
    log.info('Unique scaffolds (positives): %d', len(scaffolds))

    # --- Stratified scaffold splitting ---
    pos_potency = dict(zip(pos['compound_chembl_id'], pos['potency_class_num']))

    train_ids, val_ids, test_ids = [], [], []
    train_counts = {1: 0, 2: 0, 3: 0}
    val_counts = {1: 0, 2: 0, 3: 0}
    test_counts = {1: 0, 2: 0, 3: 0}
    total_per_class = defaultdict(int)
    for cid in pos_ids_splittable:
        p = int(pos_potency.get(cid, 0))
        if p in [1, 2, 3]:
            total_per_class[p] += 1

    # Sort scaffolds by size (largest first) for better balance
    scaffolds_sorted = sorted(scaffolds, key=lambda s: len(scaffold_groups[s]), reverse=True)

    for scaffold in scaffolds_sorted:
        cids = scaffold_groups[scaffold]
        group_counts = {1: 0, 2: 0, 3: 0}
        for cid in cids:
            p = int(pos_potency.get(cid, 0))
            if p in [1, 2, 3]:
                group_counts[p] += 1

        n_total = len(train_ids) + len(val_ids) + len(test_ids) + len(cids)
        if n_total == 0:
            train_ids.extend(cids)
            for p in [1, 2, 3]:
                train_counts[p] += group_counts[p]
            continue

        train_ratio = len(train_ids) / max(n_total, 1)
        val_ratio = len(val_ids) / max(n_total, 1)
        test_ratio = len(test_ids) / max(n_total, 1)

        train_deficit = TRAIN_FRAC - train_ratio
        val_deficit = VAL_FRAC - val_ratio
        test_deficit = TEST_FRAC - test_ratio

        best = max([('train', train_deficit), ('val', val_deficit), ('test', test_deficit)],
                   key=lambda x: x[1])

        if best[0] == 'train':
            train_ids.extend(cids)
            for p in [1, 2, 3]:
                train_counts[p] += group_counts[p]
        elif best[0] == 'val':
            val_ids.extend(cids)
            for p in [1, 2, 3]:
                val_counts[p] += group_counts[p]
        else:
            test_ids.extend(cids)
            for p in [1, 2, 3]:
                test_counts[p] += group_counts[p]

    log.info('--- Positive split sizes ---')
    log.info('Train: %d (%.1f%%)', len(train_ids), 100 * len(train_ids) / len(pos_ids_splittable))
    log.info('Val:   %d (%.1f%%)', len(val_ids), 100 * len(val_ids) / len(pos_ids_splittable))
    log.info('Test:  %d (%.1f%%)', len(test_ids), 100 * len(test_ids) / len(pos_ids_splittable))

    log.info('--- Potency balance ---')
    for p in [1, 2, 3]:
        label = {1: 'active', 2: 'potent', 3: 'very_potent'}[p]
        total = total_per_class[p]
        log.info('  %s: train=%d(%.0f%%) val=%d(%.0f%%) test=%d(%.0f%%)',
                 label,
                 train_counts[p], 100 * train_counts[p] / max(total, 1),
                 val_counts[p], 100 * val_counts[p] / max(total, 1),
                 test_counts[p], 100 * test_counts[p] / max(total, 1))

    # --- Assign negatives to splits based on scaffold ---
    neg_train, neg_val, neg_test = [], [], []
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    neg_ids = list(neg['compound_chembl_id'].dropna().unique())
    np.random.shuffle(neg_ids)

    train_scaffolds = set(cid_to_scaffold.get(cid, '') for cid in train_ids)
    val_scaffolds = set(cid_to_scaffold.get(cid, '') for cid in val_ids)
    test_scaffolds = set(cid_to_scaffold.get(cid, '') for cid in test_ids)

    neg_unassigned = []
    for cid in neg_ids:
        scaffold = cid_to_scaffold.get(cid, 'UNKNOWN')
        if scaffold in train_scaffolds:
            neg_train.append(cid)
        elif scaffold in val_scaffolds:
            neg_val.append(cid)
        elif scaffold in test_scaffolds:
            neg_test.append(cid)
        else:
            neg_unassigned.append(cid)

    # Distribute unassigned negatives proportionally
    np.random.shuffle(neg_unassigned)
    n = len(neg_unassigned)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    neg_train.extend(neg_unassigned[:n_train])
    neg_val.extend(neg_unassigned[n_train:n_train + n_val])
    neg_test.extend(neg_unassigned[n_train + n_val:])

    log.info('--- Negative split sizes ---')
    log.info('Train: %d, Val: %d, Test: %d', len(neg_train), len(neg_val), len(neg_test))

    # --- Verify no scaffold leakage ---
    train_all_scaffolds = set(cid_to_scaffold.get(cid, '') for cid in train_ids)
    val_all_scaffolds = set(cid_to_scaffold.get(cid, '') for cid in val_ids)
    test_all_scaffolds = set(cid_to_scaffold.get(cid, '') for cid in test_ids)

    tv_leak = train_all_scaffolds & val_all_scaffolds - {'UNKNOWN', 'NO_SMILES', 'INVALID_SMILES', 'SCAFFOLD_ERROR'}
    tt_leak = train_all_scaffolds & test_all_scaffolds - {'UNKNOWN', 'NO_SMILES', 'INVALID_SMILES', 'SCAFFOLD_ERROR'}
    vt_leak = val_all_scaffolds & test_all_scaffolds - {'UNKNOWN', 'NO_SMILES', 'INVALID_SMILES', 'SCAFFOLD_ERROR'}
    log.info('--- Scaffold leakage check ---')
    log.info('Train-Val overlap: %d scaffolds', len(tv_leak))
    log.info('Train-Test overlap: %d scaffolds', len(tt_leak))
    log.info('Val-Test overlap: %d scaffolds', len(vt_leak))
    if tv_leak or tt_leak or vt_leak:
        log.warning('SCAFFOLD LEAKAGE DETECTED — review split logic!')

    # --- Cell-line stress test: hold out KB cell line ---
    kb_ids = set(pos.loc[
        pos['cell_line'].fillna('').str.contains('KB', case=False, na=False),
        'compound_chembl_id'
    ].dropna())
    cellline_stress = list(kb_ids & set(test_ids))
    log.info('Cell-line stress test (KB in test): %d compounds', len(cellline_stress))

    # --- Save splits ---
    splits = {
        'train': {
            'positive_ids': sorted(train_ids),
            'negative_ids': sorted(neg_train),
        },
        'val': {
            'positive_ids': sorted(val_ids),
            'negative_ids': sorted(neg_val),
        },
        'test': {
            'positive_ids': sorted(test_ids),
            'negative_ids': sorted(neg_test),
        },
        'plos_benchmark': sorted(plos_ids),
        'approved_benchmark': sorted(approved_ids),
        'cellline_stress_kb': sorted(cellline_stress),
        'metadata': {
            'seed': SEED,
            'train_frac': TRAIN_FRAC,
            'val_frac': VAL_FRAC,
            'test_frac': TEST_FRAC,
            'n_positives': len(pos_ids_all),
            'n_benchmarks_excluded': len(benchmark_ids),
            'n_negatives': len(neg_ids),
            'split_method': 'scaffold_group_greedy',
        }
    }

    with open(OUTPUT, 'w') as f:
        json.dump(splits, f, indent=2)
    log.info('Saved → %s', OUTPUT)


if __name__ == '__main__':
    main()
