#!/usr/bin/env python3
"""
Phase 3, Steps 4-5 — SELFIES-VAE and Conditional VAE.

Architectures:
  1. VAE:  GRU encoder → (μ,σ) → 128-d latent → GRU decoder (autoregressive)
  2. cVAE: Same + FiLM conditioning on potency class & VP centroid similarity

Both share the same encoder-decoder backbone and training loop.

Usage:
    python generative/vae.py --mode vae --epochs 100 --device mps
    python generative/vae.py --mode cvae --epochs 100 --device mps
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

GENESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(GENESIS_DIR, 'generative'))

from dataset import SelfiesDataset, get_dataloaders, decode_tokens

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

GEN_DIR = os.path.join(GENESIS_DIR, 'generative')
CHECKPOINT_DIR = os.path.join(GEN_DIR, 'checkpoints')
LOG_DIR = os.path.join(GEN_DIR, 'logs')


# ── FiLM Conditioning Module ────────────────────────────────────────

class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation for conditioning on potency/centroid.

    FiLM (Perez et al., 2018) applies per-neuron scale (γ) and shift (β)
    rather than concatenating the condition vector to the input.  This
    preserves hidden dimensionality and is more expressive for low-d
    conditions (5-d) modulating high-d states (1024-d) — concatenation
    would barely perturb the representation.

    γ learns 'which structural features matter given this potency target'
    β learns 'what baseline activity signature all neurons should express'
    """

    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h, cond):
        """h: (batch, hidden_dim), cond: (batch, cond_dim)"""
        return self.gamma(cond) * h + self.beta(cond)


# ── VAE Encoder ──────────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    """GRU encoder: token sequence → (μ, log_var) in 128-d latent space.

    Separate nn.Module from VAEDecoder by design — enables selective
    freezing in downstream RL fine-tuning (rl_finetune.py freezes the
    encoder while tuning the decoder).  No parameters are shared.
    """

    def __init__(self, vocab_size, emb_dim=128, hidden_dim=512, latent_dim=128,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """x: (batch, seq_len) token IDs → μ, log_var"""
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        _, h = self.gru(emb)     # h: (n_layers, batch, hidden_dim)
        h_last = h[-1]           # (batch, hidden_dim)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar


# ── VAE Decoder ──────────────────────────────────────────────────────

class VAEDecoder(nn.Module):
    """GRU decoder: z → token sequence (teacher-forced during training).

    Initial hidden state is computed from z via latent_to_hidden (128→1024),
    then reshaped to (2 layers, batch, 512).  For cVAE, FiLM modulates
    this initial state *before* reshaping, so the conditioning signal
    pervades the entire decoding process from the first GRU step.
    """

    def __init__(self, vocab_size, emb_dim=128, hidden_dim=512, latent_dim=128,
                 n_layers=2, dropout=0.1, max_len=128, conditional=False, cond_dim=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_len = max_len
        self.conditional = conditional

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * n_layers)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        if conditional:
            self.film = FiLMConditioner(cond_dim, hidden_dim * n_layers)

    def forward(self, z, target_tokens=None, cond=None):
        """
        z: (batch, latent_dim) — latent vector
        target_tokens: (batch, seq_len) — teacher-forcing targets (training)
        cond: (batch, cond_dim) — conditioning features (cVAE only)
        Returns: logits (batch, seq_len, vocab_size)
        """
        batch_size = z.size(0)
        h = self.latent_to_hidden(z)  # (batch, hidden_dim * n_layers)

        if self.conditional and cond is not None:
            h = self.film(h, cond)

        h = h.view(batch_size, self.n_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()  # (n_layers, batch, hidden_dim)

        if target_tokens is not None:
            # Teacher forcing
            emb = self.embedding(target_tokens[:, :-1])  # exclude last token
            output, _ = self.gru(emb, h)
            logits = self.output_proj(output)
            return logits
        else:
            # Autoregressive generation
            return self._generate(h, batch_size)

    def _generate(self, h, batch_size):
        """Autoregressive generation from initial hidden state."""
        input_token = torch.ones(batch_size, 1, dtype=torch.long, device=h.device)
        logits_list = []

        for _ in range(self.max_len - 1):
            emb = self.embedding(input_token)
            output, h = self.gru(emb, h)
            logit = self.output_proj(output)
            logits_list.append(logit)
            input_token = logit.argmax(dim=-1)

        return torch.cat(logits_list, dim=1)


# ── VAE Model ────────────────────────────────────────────────────────

class SELFIES_VAE(nn.Module):
    """Full VAE / conditional VAE for SELFIES generation."""

    def __init__(self, vocab_size, emb_dim=128, hidden_dim=512, latent_dim=128,
                 n_layers=2, dropout=0.1, max_len=128, conditional=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.conditional = conditional

        # 5-d conditioning vector: [potency_class_onehot(4), centroid_sim(1)]
        cond_dim = 5 if conditional else 0

        self.encoder = VAEEncoder(vocab_size, emb_dim, hidden_dim, latent_dim,
                                  n_layers, dropout)
        self.decoder = VAEDecoder(vocab_size, emb_dim, hidden_dim, latent_dim,
                                  n_layers, dropout, max_len, conditional, cond_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tokens, cond=None):
        """
        tokens: (batch, seq_len) — input/target token IDs
        cond: (batch, cond_dim) — conditioning (cVAE)
        Returns: logits, mu, logvar
        """
        mu, logvar = self.encoder(tokens)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, target_tokens=tokens, cond=cond)
        return logits, mu, logvar

    def generate(self, n_samples, device, cond=None, temperature=1.0, top_k=0):
        """Generate molecules by sampling from prior.

        Two independent temperature knobs:
          - temperature here scales the prior z: z = N(0,I) × T_prior.
            Controls *where* in latent space to explore (global structure).
          - temperature in _sample_top_k scales logits before softmax.
            Controls *how* the decoder converts hidden states to tokens
            (local token-level randomness).
        """
        z = torch.randn(n_samples, self.latent_dim, device=device) * temperature

        if self.conditional and cond is not None:
            cond = cond.to(device)

        self.eval()
        with torch.no_grad():
            logits = self.decoder(z, cond=cond)

        if top_k > 0:
            tokens = self._sample_top_k(logits, top_k, temperature)
        else:
            tokens = logits.argmax(dim=-1)

        return tokens.cpu()

    def _sample_top_k(self, logits, k, temperature):
        """Top-k sampling from logits."""
        logits = logits / temperature
        top_k_vals, top_k_idx = logits.topk(k, dim=-1)
        probs = F.softmax(top_k_vals, dim=-1)
        sampled = torch.multinomial(probs.view(-1, k), 1).view(logits.size(0), -1)
        tokens = top_k_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return tokens

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss Functions ───────────────────────────────────────────────────

def vae_loss(logits, targets, mu, logvar, kl_weight=1.0, pad_idx=0):
    """VAE ELBO loss: reconstruction + KL divergence.

    kl_weight follows a cyclical annealing schedule (Fu et al., 2019):
    repeating 0→1 ramps across 4 cycles.  This combats posterior collapse
    by giving the encoder periodic 'freedom' (β≈0) to learn informative
    latent codes before KL regularisation re-engages.
    """
    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets[:, 1:].reshape(-1),
        ignore_index=pad_idx,
        reduction='mean'
    )
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


def kl_annealing_weight(epoch, warmup_epochs=10, strategy='cyclical', n_cycles=4, max_epochs=100):
    """KL weight annealing schedule."""
    if strategy == 'linear':
        return min(1.0, epoch / warmup_epochs)
    elif strategy == 'cyclical':
        cycle_len = max_epochs / n_cycles
        pos_in_cycle = epoch % cycle_len
        return min(1.0, pos_in_cycle / (cycle_len * 0.5))
    elif strategy == 'monotonic':
        return min(1.0, epoch / warmup_epochs)
    return 1.0


# ── Conditioning Helper ──────────────────────────────────────────────

def build_conditioning(batch, centroid=None):
    """Build 5-d conditioning vector from batch data.

    [potency_class_0, potency_class_1, potency_class_2, potency_class_3, centroid_sim]

    Deliberately 5-d, not 133-d (UPDATE_1 fix): including the raw 128-d
    centroid vector would constitute information leakage — the FiLM network
    could trivially decode the VP destination without using the latent code z.
    The 5-d signal provides *direction* (scalar similarity) without
    *destination* (raw coordinates), forcing genuine conditional generation.
    """
    potency = batch['potency_class']
    potency_oh = F.one_hot(potency.clamp(0, 3), num_classes=4).float()

    if centroid is not None:
        proj = batch['proj_128']
        centroid_t = torch.tensor(centroid, dtype=torch.float32, device=proj.device)
        sim = F.cosine_similarity(proj, centroid_t.unsqueeze(0), dim=-1, eps=1e-8)
        sim = sim.unsqueeze(-1)
    else:
        sim = torch.zeros(potency.size(0), 1, device=potency.device)

    return torch.cat([potency_oh, sim], dim=-1)


# ── Training Loop ────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, kl_weight, centroid=None):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    for batch in loader:
        tokens = batch['tokens'].to(device)
        cond = None
        if model.conditional:
            cond = build_conditioning(batch, centroid).to(device)

        logits, mu, logvar = model(tokens, cond=cond)
        loss, recon, kl = vae_loss(logits, tokens, mu, logvar,
                                   kl_weight=kl_weight, pad_idx=0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
    }


@torch.no_grad()
def validate(model, loader, device, kl_weight, centroid=None):
    """Validate one epoch."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    for batch in loader:
        tokens = batch['tokens'].to(device)
        cond = None
        if model.conditional:
            cond = build_conditioning(batch, centroid).to(device)

        logits, mu, logvar = model(tokens, cond=cond)
        loss, recon, kl = vae_loss(logits, tokens, mu, logvar,
                                   kl_weight=kl_weight, pad_idx=0)

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
    }


def generate_and_decode(model, n_samples, device, inv_vocab, centroid=None,
                        target_potency=3, temperature=1.0, top_k=10):
    """Generate molecules and decode to SMILES."""
    import selfies as sf

    cond = None
    if model.conditional and centroid is not None:
        potency_oh = torch.zeros(n_samples, 4)
        potency_oh[:, target_potency] = 1.0
        sim = torch.ones(n_samples, 1) * 0.9
        cond = torch.cat([potency_oh, sim], dim=-1).to(device)

    tokens = model.generate(n_samples, device, cond=cond,
                            temperature=temperature, top_k=top_k)

    smiles_list = []
    for i in range(tokens.size(0)):
        selfies_str = decode_tokens(tokens[i].numpy(), inv_vocab)
        try:
            smi = sf.decoder(selfies_str)
            if smi:
                smiles_list.append(smi)
        except Exception:
            pass

    return smiles_list


def main():
    parser = argparse.ArgumentParser(description='SELFIES-VAE / cVAE Training')
    parser.add_argument('--mode', choices=['vae', 'cvae'], default='vae')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--emb-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--kl-strategy', choices=['linear', 'cyclical', 'monotonic'],
                        default='cyclical')
    parser.add_argument('--kl-warmup', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gen-every', type=int, default=10, help='Generate samples every N epochs')
    parser.add_argument('--n-gen', type=int, default=100, help='N samples to generate for monitoring')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    conditional = args.mode == 'cvae'
    log.info('Mode: %s (conditional=%s)', args.mode, conditional)

    # Load centroid for cVAE conditioning
    centroid = None
    if conditional:
        centroid_path = os.path.join(GEN_DIR, 'processed', 'vp_centroid.json')
        with open(centroid_path) as f:
            centroid = np.array(json.load(f)['oscc_vp_centroid'], dtype=np.float32)
        log.info('Loaded VP centroid (dim=%d)', len(centroid))

    # Load data
    log.info('Loading data ...')
    load_proj = conditional
    train_dl, val_dl, test_dl = get_dataloaders(
        batch_size=args.batch_size, load_projections=load_proj, device=args.device)

    vocab_size = train_dl.dataset.vocab_size
    max_len = train_dl.dataset.max_len
    log.info('Vocab size: %d, max_len: %d', vocab_size, max_len)
    log.info('Train: %d, Val: %d, Test: %d',
             len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))

    # Load inverse vocab for decoding
    inv_vocab_path = os.path.join(GEN_DIR, 'processed', 'inv_vocab.json')
    with open(inv_vocab_path) as f:
        inv_vocab = json.load(f)

    # Build model
    model = SELFIES_VAE(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_len=max_len,
        conditional=conditional,
    ).to(device)
    log.info('Model parameters: %s', f'{model.param_count():,}')

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # Training loop
    history = []
    best_val_recon = float('inf')
    patience_counter = 0
    ckpt_name = f'best_{args.mode}.pt'

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        kl_weight = kl_annealing_weight(epoch, args.kl_warmup, args.kl_strategy,
                                        max_epochs=args.epochs)

        train_metrics = train_epoch(model, train_dl, optimizer, device, kl_weight, centroid)
        val_metrics = validate(model, val_dl, device, kl_weight, centroid)

        scheduler.step(val_metrics['recon'])
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        log.info('Epoch %3d/%d | train_loss=%.4f recon=%.4f kl=%.4f | '
                 'val_loss=%.4f recon=%.4f kl=%.4f | kl_w=%.3f lr=%.1e [%.1fs]',
                 epoch, args.epochs,
                 train_metrics['loss'], train_metrics['recon'], train_metrics['kl'],
                 val_metrics['loss'], val_metrics['recon'], val_metrics['kl'],
                 kl_weight, lr, elapsed)

        entry = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'lr': lr,
            'elapsed_s': elapsed,
        }
        for k, v in train_metrics.items():
            entry[f'train_{k}'] = v
        for k, v in val_metrics.items():
            entry[f'val_{k}'] = v
        history.append(entry)

        # Checkpoint — use recon loss (not total loss) to avoid cyclical KL artefacts.
        # With cyclical annealing, total loss drops artificially at cycle starts (β→0)
        # and spikes at cycle peaks (β→1).  Reconstruction loss is a stable proxy for
        # generative quality regardless of KL schedule position.
        if val_metrics['recon'] < best_val_recon:
            best_val_recon = val_metrics['recon']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_recon': val_metrics['recon'],
                'val_loss': val_metrics['loss'],
                'args': vars(args),
            }, os.path.join(CHECKPOINT_DIR, ckpt_name))
            log.info('  ★ New best → %s (val_recon=%.4f)', ckpt_name, val_metrics['recon'])
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info('Early stopping at epoch %d', epoch)
                break

        # Periodic generation for monitoring — inline quality gate.
        # Sudden uniqueness drop → posterior collapse (decoder ignores z).
        # Sudden validity drop → SELFIES parse instability.
        # Diagnostic only — not used for early stopping or checkpointing.
        if epoch % args.gen_every == 0:
            smiles = generate_and_decode(model, args.n_gen, device, inv_vocab,
                                         centroid=centroid, temperature=1.0, top_k=10)
            valid_count = len(smiles)
            unique_count = len(set(smiles))
            log.info('  Generation: %d/%d valid, %d unique',
                     valid_count, args.n_gen, unique_count)

    # Save history
    history_path = os.path.join(LOG_DIR, f'history_{args.mode}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    log.info('History → %s', history_path)

    # Final generation & evaluation
    log.info('=== Final Generation (1000 molecules) ===')
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, ckpt_name), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    smiles = generate_and_decode(model, 1000, device, inv_vocab,
                                 centroid=centroid, temperature=1.0, top_k=10)
    log.info('Generated %d valid SMILES from 1000 samples', len(smiles))

    # Save generated SMILES
    gen_path = os.path.join(GEN_DIR, 'results', f'generated_{args.mode}.txt')
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
    with open(gen_path, 'w') as f:
        for s in smiles:
            f.write(s + '\n')
    log.info('Generated SMILES → %s', gen_path)

    # Run evaluation
    try:
        from evaluate_gen import evaluate_molecules
        results = evaluate_molecules(smiles, method_name=args.mode, device=str(device),
                                     compute_potency=True)
        log.info('Evaluation complete: validity=%.1f%%, uniqueness=%.1f%%, novelty=%.1f%%',
                 100 * results['metrics']['validity'],
                 100 * results['metrics']['uniqueness'],
                 100 * results['metrics']['novelty'])
    except Exception as e:
        log.warning('Evaluation failed: %s', e)

    log.info('Training complete. Best val_recon=%.4f at epoch %d',
             best_val_recon, ckpt['epoch'])


if __name__ == '__main__':
    main()
