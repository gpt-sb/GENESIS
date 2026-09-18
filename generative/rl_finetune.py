#!/usr/bin/env python3
"""
Phase 3, Step 8 — Reinforcement Learning Fine-Tuning.

REINFORCE (Williams, 1992) policy gradient optimization of the VAE decoder.
The reward function R(x) is non-differentiable — RDKit molecular parsing,
QED/SA computations, and the frozen MolDeBERTa+ContrastiveModel pipeline
have no gradients. REINFORCE bypasses this via the log-derivative trick:
R(x) * ∇θ log πθ(x|z), using the decoder's differentiable log-probs
weighted by the black-box reward to produce valid policy gradients.

Multi-objective reward combining:
  - OSCC potency (centroid similarity in 128-d space) — 40%
  - QED (drug-likeness) — 20%
  - SA score (synthetic accessibility) — 20%
  - Validity — 20%

Key fix baked in (from UPDATE_1):
  - Computes REAL potency reward via MolDeBERTa → ContrastiveModel → cosine sim
    (original had hardcoded 0.0 potency)

Usage:
    python generative/rl_finetune.py --base-model vae --epochs 50 --device mps
"""
import os
import sys
import json
import time
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from rdkit import Chem
from rdkit.Chem import QED, Descriptors

GENESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(GENESIS_DIR, 'generative'))

from dataset import decode_tokens
from vae import SELFIES_VAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

GEN_DIR = os.path.join(GENESIS_DIR, 'generative')
MODELING_DIR = os.path.join(GENESIS_DIR, 'modeling')
CHECKPOINT_DIR = os.path.join(GEN_DIR, 'checkpoints')
LOG_DIR = os.path.join(GEN_DIR, 'logs')


# ── SA Score ─────────────────────────────────────────────────────────

def get_sa_score(mol):
    """Get synthetic accessibility score."""
    try:
        from rdkit.Chem import RDConfig
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer
        return sascorer.calculateScore(mol)
    except Exception:
        n_rings = Descriptors.RingCount(mol)
        n_stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        mw = Descriptors.MolWt(mol)
        return min(1.0 + 0.5 * n_rings + 0.3 * len(n_stereo) + mw / 500.0, 10.0)


# ── Reward Function ─────────────────────────────────────────────────

class MultiObjectiveReward:
    """Multi-objective reward for RL fine-tuning.

    Components:
      - potency: cosine similarity to VP centroid (from contrastive model)
      - qed: drug-likeness [0, 1]
      - sa: synthetic accessibility [1, 10] → normalized to [0, 1]
      - validity: whether SMILES parses to valid molecule
    """

    def __init__(self, centroid, weights=None, kl_penalty=0.5, device='cpu'):
        self.centroid = centroid
        self.weights = weights or {'potency': 0.4, 'qed': 0.2, 'sa': 0.2, 'validity': 0.2}
        self.kl_penalty = kl_penalty
        self.device = device

        self._potency_model = None
        self._tokenizer = None
        self._embedder = None
        self._load_potency_scorer()

    def _load_potency_scorer(self):
        """Load MolDeBERTa + ContrastiveModel for potency reward.

        Both models are frozen (requires_grad=False) — they serve as
        reward oracles, NOT trainable components. Gradients must not
        flow through the reward function; only the decoder policy is
        updated via REINFORCE.
        """
        try:
            sys.path.insert(0, MODELING_DIR)
            from model import ContrastiveModel

            ckpt_path = os.path.join(MODELING_DIR, 'checkpoints', 'best.pt')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model = ContrastiveModel()
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval().to(self.device)
            self._potency_model = model

            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(
                'SaeedLab/MolDeBERTa-base-123M-mtr', use_fast=False)
            self._embedder = AutoModel.from_pretrained(
                'SaeedLab/MolDeBERTa-base-123M-mtr')
            self._embedder.eval().to(self.device)
            for p in self._embedder.parameters():
                p.requires_grad = False
            for p in self._potency_model.parameters():
                p.requires_grad = False
            log.info('Potency scorer loaded (MolDeBERTa + ContrastiveModel)')
        except Exception as e:
            log.warning('Failed to load potency scorer: %s — potency reward disabled', e)

    def _compute_potency_batch(self, smiles_list):
        """Compute cosine similarity to VP centroid for a batch of SMILES.

        Two-tier defensive pattern: (1) batch MolDeBERTa inference for
        GPU efficiency, then (2) individual per-molecule fallback if the
        batch call fails. RL-generated SMILES are arbitrary — malformed
        strings can crash the tokeniser. The fallback prevents one bad
        molecule from invalidating the entire batch's potency scores.
        Failed molecules receive potency = 0.0 (graceful degradation).
        """
        if self._potency_model is None or self._embedder is None:
            return [0.0] * len(smiles_list)

        centroid_t = torch.tensor(self.centroid, device=self.device, dtype=torch.float32)
        centroid_t = centroid_t / (centroid_t.norm() + 1e-12)
        potency_scores = [0.0] * len(smiles_list)

        valid_indices = []
        valid_smiles = []
        for i, smi in enumerate(smiles_list):
            if smi is not None and Chem.MolFromSmiles(smi) is not None:
                valid_indices.append(i)
                valid_smiles.append(smi)

        if not valid_smiles:
            return potency_scores

        with torch.no_grad():
            try:
                inputs = self._tokenizer(valid_smiles, return_tensors='pt', padding=True,
                                         truncation=True, max_length=128).to(self.device)
                outputs = self._embedder(**inputs)
                embs = outputs.last_hidden_state[:, 0, :]
                z, _ = self._potency_model(embs)
                sims = F.cosine_similarity(z, centroid_t.unsqueeze(0), dim=-1)
                sims = sims.clamp(0, 1).cpu().tolist()
                for idx, sim_val in zip(valid_indices, sims):
                    potency_scores[idx] = sim_val
            except Exception:
                for idx, smi in zip(valid_indices, valid_smiles):
                    try:
                        inp = self._tokenizer([smi], return_tensors='pt', padding=True,
                                              truncation=True, max_length=128).to(self.device)
                        out = self._embedder(**inp)
                        emb = out.last_hidden_state[:, 0, :]
                        z, _ = self._potency_model(emb)
                        sim = F.cosine_similarity(z, centroid_t.unsqueeze(0), dim=-1)
                        potency_scores[idx] = sim.clamp(0, 1).item()
                    except Exception:
                        potency_scores[idx] = 0.0

        return potency_scores

    def compute(self, smiles_list, log_probs=None, prior_log_probs=None):
        """Compute reward for a batch of generated SMILES.

        Reward is an ADDITIVE weighted sum (not multiplicative). This
        allows graceful trade-offs: a highly potent molecule with moderate
        SA still gets encouraged. Multiplicative would zero-out the
        reward if any single component fails — too harsh for early-phase
        discovery where potency is the primary requirement and synthetic
        routes can be optimised later.  See §11.8 for the formal design
        justification and concrete scoring examples.
        """
        n = len(smiles_list)
        rewards = torch.zeros(n)
        info = {k: [] for k in ['potency', 'qed', 'sa', 'validity']}

        potency_scores = self._compute_potency_batch(smiles_list)

        for i, smi in enumerate(smiles_list):
            if smi is None:
                for k in info:
                    info[k].append(0.0)
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                info['validity'].append(0.0)
                info['potency'].append(0.0)
                info['qed'].append(0.0)
                info['sa'].append(0.0)
                continue

            info['validity'].append(1.0)

            try:
                qed_val = QED.qed(mol)
            except Exception:
                qed_val = 0.0
            info['qed'].append(qed_val)

            sa_raw = get_sa_score(mol)
            sa_norm = max(0, (10.0 - sa_raw) / 9.0)
            info['sa'].append(sa_norm)

            pot_val = potency_scores[i]
            info['potency'].append(pot_val)

            rewards[i] = (self.weights['validity'] * 1.0 +
                          self.weights['qed'] * qed_val +
                          self.weights['sa'] * sa_norm +
                          self.weights['potency'] * pot_val)

        if log_probs is not None and prior_log_probs is not None:
            kl = (log_probs - prior_log_probs).mean(dim=-1)
            rewards = rewards.to(kl.device) - self.kl_penalty * kl.detach()

        return rewards, info


# ── RL Training Functions ────────────────────────────────────────────

def generate_with_log_probs(model, n_samples, device, temperature=1.0, top_k=10):
    """Generate molecules and collect log probabilities for REINFORCE.

    Returns per-token log π_θ(x_t | x_{<t}, z) needed by the
    log-derivative trick: REINFORCE multiplies each log-prob by the
    episode reward to form the policy gradient.  Top-k=10 constrains
    sampling to the 10 most likely tokens at each step — prevents
    degenerate token choices while maintaining exploration (k=1 would
    be greedy/deterministic; k=vocab_size would be unconstrained).
    z ~ N(0,I) from the prior (encoder is NOT used), providing fresh
    random seeds that prevent mode collapse by construction.
    """
    model.eval()
    z = torch.randn(n_samples, model.latent_dim, device=device)

    h = model.decoder.latent_to_hidden(z)
    h = h.view(n_samples, model.decoder.n_layers, model.decoder.hidden_dim)
    h = h.permute(1, 0, 2).contiguous()

    input_token = torch.ones(n_samples, 1, dtype=torch.long, device=device)
    all_tokens = [input_token.squeeze(1)]
    all_log_probs = []

    for step in range(model.decoder.max_len - 1):
        emb = model.decoder.embedding(input_token)
        output, h = model.decoder.gru(emb, h)
        logits = model.decoder.output_proj(output.squeeze(1))

        scaled_logits = logits / temperature
        if top_k > 0:
            top_k_vals, top_k_idx = scaled_logits.topk(top_k, dim=-1)
            probs = F.softmax(top_k_vals, dim=-1)
            sampled_idx = torch.multinomial(probs, 1)
            tokens = top_k_idx.gather(-1, sampled_idx).squeeze(-1)
            log_prob = F.log_softmax(scaled_logits, dim=-1).gather(
                -1, tokens.unsqueeze(-1)).squeeze(-1)
        else:
            probs = F.softmax(scaled_logits, dim=-1)
            tokens = torch.multinomial(probs, 1).squeeze(-1)
            log_prob = F.log_softmax(scaled_logits, dim=-1).gather(
                -1, tokens.unsqueeze(-1)).squeeze(-1)

        all_tokens.append(tokens)
        all_log_probs.append(log_prob)
        input_token = tokens.unsqueeze(1)

    tokens_tensor = torch.stack(all_tokens, dim=1)
    log_probs_tensor = torch.stack(all_log_probs, dim=1)

    return tokens_tensor, log_probs_tensor, z


def tokens_to_smiles(tokens_batch, inv_vocab):
    """Convert batch of token IDs to SMILES strings."""
    import selfies as sf

    smiles_list = []
    for i in range(tokens_batch.size(0)):
        selfies_str = decode_tokens(tokens_batch[i].cpu().numpy(), inv_vocab)
        try:
            smi = sf.decoder(selfies_str)
            smiles_list.append(smi)
        except Exception:
            smiles_list.append(None)
    return smiles_list


def reinforce_step(model, optimizer, reward_fn, inv_vocab, device,
                   batch_size=64, temperature=1.0, baseline_ema=0.0,
                   baseline_decay=0.9):
    """One REINFORCE optimization step.

    Implements: loss = -Σ (R(x) - baseline) · Σ_t log π_θ(x_t|x_{<t},z)
    The EMA baseline (decay=0.9) converts absolute rewards into relative
    signals — "better/worse than recent average" — which drastically
    reduces REINFORCE variance.  Without it, all rewards are positive
    [0,1] so every molecule's probability would increase.
    Pad mask ensures only real token decisions receive credit, not
    padding positions (variable-length SELFIES sequences).
    """
    model.train()

    tokens, log_probs, z = generate_with_log_probs(
        model, batch_size, device, temperature=temperature, top_k=10)

    smiles_list = tokens_to_smiles(tokens, inv_vocab)
    rewards, info = reward_fn.compute(smiles_list, log_probs=log_probs)
    rewards = rewards.to(device)

    mean_reward = rewards.mean().item()
    baseline = baseline_ema
    advantages = rewards - baseline

    pad_mask = (tokens[:, 1:] != 0).float()
    masked_log_probs = log_probs * pad_mask
    loss = -(advantages.unsqueeze(1) * masked_log_probs).sum(1).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    new_baseline = baseline * baseline_decay + mean_reward * (1 - baseline_decay)

    return mean_reward, new_baseline, info, loss.item()


def main():
    parser = argparse.ArgumentParser(description='RL Fine-Tuning')
    parser.add_argument('--base-model', choices=['vae', 'cvae'], default='vae')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps-per-epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--kl-penalty', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── Load base model ──────────────────────────────────────────────
    ckpt_name = f'best_{args.base_model}.pt'
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    log.info('Loading base model from %s', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vae_args = ckpt['args']

    with open(os.path.join(GEN_DIR, 'processed', 'vocab.json')) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    model = SELFIES_VAE(
        vocab_size=vocab_size,
        emb_dim=vae_args.get('emb_dim', 128),
        hidden_dim=vae_args.get('hidden_dim', 512),
        latent_dim=vae_args.get('latent_dim', 128),
        n_layers=vae_args.get('n_layers', 2),
        max_len=128,
        conditional=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    log.info('Base model loaded (epoch %d)', ckpt['epoch'])

    # Freeze encoder — only fine-tune decoder for generation.
    # The encoder defines the VAE's latent geometry; if RL tuned it,
    # the latent space could collapse as gradients optimise for reward,
    # breaking the Gaussian prior assumption.  Freezing constrains RL to:
    # "given the same latent codes, decode them into higher-reward molecules."
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Load centroid
    centroid_path = os.path.join(GEN_DIR, 'processed', 'vp_centroid.json')
    with open(centroid_path) as f:
        centroid = np.array(json.load(f)['oscc_vp_centroid'], dtype=np.float32)

    # Load inverse vocab
    with open(os.path.join(GEN_DIR, 'processed', 'inv_vocab.json')) as f:
        inv_vocab = json.load(f)

    # Setup reward function
    reward_fn = MultiObjectiveReward(centroid, kl_penalty=args.kl_penalty,
                                     device=str(device))

    # lr=1e-5 is 30× below the VAE training rate (3e-4).  REINFORCE
    # gradients are inherently high-variance (reward × log-prob).  The
    # cautious rate acts as an implicit trust region, limiting per-step
    # policy shift and preserving the pre-trained chemical grammar.
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # ── RL Training ──────────────────────────────────────────────────
    history = []
    baseline = 0.0
    best_mean_reward = -float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_rewards = []
        epoch_losses = []
        epoch_validity = []

        for step in range(args.steps_per_epoch):
            reward, baseline, info, loss = reinforce_step(
                model, optimizer, reward_fn, inv_vocab, device,
                batch_size=args.batch_size, temperature=args.temperature,
                baseline_ema=baseline)
            epoch_rewards.append(reward)
            epoch_losses.append(loss)
            epoch_validity.append(np.mean(info['validity']))

        mean_reward = np.mean(epoch_rewards)
        mean_loss = np.mean(epoch_losses)
        mean_validity = np.mean(epoch_validity)
        elapsed = time.time() - t0

        log.info('Epoch %3d/%d | reward=%.4f | loss=%.4f | validity=%.1f%% | '
                 'qed=%.3f | sa=%.3f | baseline=%.4f [%.1fs]',
                 epoch, args.epochs, mean_reward, mean_loss,
                 100 * mean_validity,
                 np.mean([np.mean(info['qed'])]),
                 np.mean([np.mean(info['sa'])]),
                 baseline, elapsed)

        history.append({
            'epoch': epoch, 'reward': mean_reward, 'loss': mean_loss,
            'validity': mean_validity, 'baseline': baseline, 'elapsed_s': elapsed
        })

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean_reward': mean_reward,
                'args': vars(args),
            }, os.path.join(CHECKPOINT_DIR, f'best_rl_{args.base_model}.pt'))
            log.info('  ★ New best → best_rl_%s.pt (reward=%.4f)', args.base_model, mean_reward)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info('Early stopping at epoch %d', epoch)
                break

    # Save history
    with open(os.path.join(LOG_DIR, f'history_rl_{args.base_model}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Final generation ──────────────────────────────────────────────
    log.info('=== Final Generation (1000 molecules) ===')
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f'best_rl_{args.base_model}.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    import selfies as sf
    model.eval()
    all_smiles = []
    for _ in range(10):
        tokens, _, _ = generate_with_log_probs(model, 100, device,
                                                temperature=args.temperature, top_k=10)
        smiles = tokens_to_smiles(tokens, inv_vocab)
        all_smiles.extend([s for s in smiles if s is not None])

    log.info('Generated %d valid SMILES', len(all_smiles))

    gen_path = os.path.join(GEN_DIR, 'results', f'generated_rl_{args.base_model}.txt')
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
    with open(gen_path, 'w') as f:
        for s in all_smiles:
            f.write(s + '\n')
    log.info('Generated SMILES → %s', gen_path)

    try:
        from evaluate_gen import evaluate_molecules
        results = evaluate_molecules(all_smiles, method_name=f'rl_{args.base_model}',
                                     device=str(device), compute_potency=True)
    except Exception as e:
        log.warning('Evaluation failed: %s', e)

    log.info('RL fine-tuning complete. Best reward=%.4f', best_mean_reward)


if __name__ == '__main__':
    main()
