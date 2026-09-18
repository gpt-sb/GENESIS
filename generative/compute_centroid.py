#!/usr/bin/env python3
"""
Phase 3, Step 3 — Compute VP Centroid from Contrastive Model.

Computes the 128-d L2-normalized centroid of Very Potent (class 3,
pChEMBL >= 7.0) training compounds in the contrastive projection space.

Design rationale (VP-only centroid):
  - Acts as a directional beacon, not a data filter
  - Ordinal gradient naturally exists in the learned latent space
  - ~24,991 VP compounds provide statistically robust centroid
  - Mixed-potency centroid would dilute the RL reward signal

Usage:
    python generative/compute_centroid.py

Output:
    generative/processed/vp_centroid.json
"""
import os
import sys
import json
import pickle
import logging

import numpy as np
import torch

GENESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELING_DIR = os.path.join(GENESIS_DIR, 'modeling')
GEN_DIR = os.path.join(GENESIS_DIR, 'generative', 'processed')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def main():
    # Load contrastive model — only the ProjectionHead (768→128-d) matters here;
    # the PotencyRegressor output is discarded (model returns z, _).
    sys.path.insert(0, MODELING_DIR)
    from model import ContrastiveModel

    ckpt_path = os.path.join(MODELING_DIR, 'checkpoints', 'best.pt')
    log.info('Loading contrastive model from %s', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = ContrastiveModel()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load compound index
    with open(os.path.join(MODELING_DIR, 'compound_index.pkl'), 'rb') as f:
        compound_index = pickle.load(f)

    # Load embeddings
    emb_ids = np.load(
        os.path.join(GENESIS_DIR, 'embeddings', 'compound_ids.npy'), allow_pickle=True)
    n_compounds = len(emb_ids)
    emb_mmap = np.memmap(
        os.path.join(GENESIS_DIR, 'embeddings', 'compound_embeddings.mmap'),
        dtype='float16', mode='r', shape=(n_compounds, 768))

    # Restrict to training set — prevents reward leakage.
    # If val/test VPs contributed to the centroid, the RL reward signal would
    # encode held-out information, inflating downstream validation metrics.
    with open(os.path.join(MODELING_DIR, 'splits.json')) as f:
        splits = json.load(f)
    train_ids = set(splits['train']['positive_ids'] + splits['train']['negative_ids'])

    # Identify VP compounds (potency_class_num == 3, pChEMBL ≥ 7.0) in training set.
    # VP-only (not mixed-potency) because the ordinal gradient already exists in
    # contrastive space — including Active/Potent compounds would dilute the centroid
    # away from the "hot end" and weaken the RL reward signal.
    vp_cids = []
    for cid, meta in compound_index.items():
        if cid in train_ids and meta.get('potency_class_num', -1) == 3:
            vp_cids.append(cid)

    log.info('VP training compounds: %d', len(vp_cids))

    if len(vp_cids) == 0:
        log.error('No VP compounds found — cannot compute centroid')
        return

    # Project VP compounds through contrastive model
    all_projections = []
    batch_size = 512

    for start in range(0, len(vp_cids), batch_size):
        end = min(start + batch_size, len(vp_cids))
        batch_embs = []
        for cid in vp_cids[start:end]:
            emb_idx = compound_index[cid]['emb_idx']
            batch_embs.append(emb_mmap[emb_idx].astype(np.float32))

        embs_t = torch.tensor(np.stack(batch_embs))
        with torch.no_grad():
            z, _ = model(embs_t)
        all_projections.append(z.numpy())

    projections = np.concatenate(all_projections, axis=0)
    log.info('Projected %d VP compounds → shape %s', len(projections), projections.shape)

    # Compute L2-normalised centroid.  Individual projections are already unit-
    # length (ProjectionHead applies F.normalize), but the *mean* of unit vectors
    # has magnitude < 1 proportional to cluster spread.  Re-normalising gives a
    # directional centroid on the hypersphere, not a shrunken interior point.
    centroid = projections.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    log.info('Centroid norm: %.6f (should be ~1.0)', np.linalg.norm(centroid))
    log.info('Centroid dim: %d', len(centroid))

    # Quality gate — cosine similarities of each VP compound to the centroid.
    # High mean (>0.9) + low std (<0.05) confirms a tight, coherent VP cluster.
    # Low mean would signal that the contrastive model failed to separate VP
    # from other classes — a cue to retrain Phase 2 before proceeding to RL.
    sims = projections @ centroid
    log.info('VP-centroid similarity: mean=%.4f, std=%.4f, min=%.4f, max=%.4f',
             sims.mean(), sims.std(), sims.min(), sims.max())

    # Save as JSON (not pickle) — human-readable, version-controllable, and
    # avoids arbitrary code execution risk.  Embedded metadata (sim stats,
    # compound count, checkpoint path) provides provenance for reproducibility.
    # The key 'oscc_vp_centroid' is referenced by all 6 downstream generative
    # consumers (VAE, cVAE, Diffusion, RL, CGLF, CRAVE).
    os.makedirs(GEN_DIR, exist_ok=True)
    output = {
        'oscc_vp_centroid': centroid.tolist(),
        'centroid_dim': int(len(centroid)),
        'n_vp_compounds': int(len(vp_cids)),
        'centroid_norm': float(np.linalg.norm(centroid)),
        'vp_sim_stats': {
            'mean': float(sims.mean()),
            'std': float(sims.std()),
            'min': float(sims.min()),
            'max': float(sims.max()),
        },
        'source_checkpoint': ckpt_path,
    }
    out_path = os.path.join(GEN_DIR, 'vp_centroid.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    log.info('VP centroid → %s', out_path)


if __name__ == '__main__':
    main()
