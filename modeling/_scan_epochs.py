#!/usr/bin/env python3
"""Scan candidate epoch checkpoints to find the true best on downstream metrics
(not just val_loss). Diagnostic script for the checkpoint-selection bug investigation.
"""
import sys
import json
import pickle
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate import (
    load_model, test_potency_ordering, test_activity_cliff,
    test_target_switch, test_vp_retrieval, test_potency_graded_ranking,
)

MODELING_DIR = SCRIPT_DIR
GENESIS_DIR = SCRIPT_DIR.parent
CHECKPOINT_DIR = MODELING_DIR / 'checkpoints'
DEVICE = 'mps'

with open(MODELING_DIR / 'compound_index.pkl', 'rb') as f:
    compound_index = pickle.load(f)
with open(MODELING_DIR / 'splits.json') as f:
    splits = json.load(f)
n_compounds = len(np.load(GENESIS_DIR / 'embeddings' / 'compound_ids.npy', allow_pickle=True))
emb_mmap = np.memmap(GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap',
                      dtype='float16', mode='r', shape=(n_compounds, 768))

candidates = [28, 31, 34, 37, 40]
results = {}
for epoch in candidates:
    ckpt_path = CHECKPOINT_DIR / f'epoch_{epoch:03d}.pt'
    if not ckpt_path.exists():
        print(f'MISSING: {ckpt_path}')
        continue
    model, ckpt = load_model(str(ckpt_path), device=DEVICE)
    r = {}
    r.update(test_potency_ordering(model, splits, compound_index, emb_mmap, DEVICE))
    r.update(test_activity_cliff(model, splits, compound_index, emb_mmap, DEVICE))
    r.update(test_target_switch(model, splits, compound_index, emb_mmap, DEVICE))
    r.update(test_vp_retrieval(model, splits, compound_index, emb_mmap, DEVICE))
    r.update(test_potency_graded_ranking(model, splits, compound_index, emb_mmap, DEVICE))
    results[epoch] = r
    print(f"epoch {epoch:3d} | val_loss(ckpt)={ckpt.get('val_loss'):.4f} | "
          f"spearman={r['global_spearman_rho']:.4f} | cliff={r['cliff_accuracy']:.4f} | "
          f"ts_auroc={r['target_switch_auc_roc']:.4f} | vp_recall100={r['vp_recall_100']:.4f} | "
          f"ndcg100={r['ndcg_100']:.4f}", flush=True)

with open(MODELING_DIR / 'logs' / 'epoch_scan_results_ext.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to modeling/logs/epoch_scan_results_ext.json')
