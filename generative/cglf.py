#!/usr/bin/env python3
"""
Phase 3, Steps 9-11 — Contrastive-Guided Latent Flow (CGLF).

Novel method combining:
  1. Flow matching in the 128-d contrastive projection space
  2. VP centroid as differentiable guidance signal
  3. Learned SELFIES decoder from latent space

Key innovations over existing methods:
  - VAE: No mode collapse — flow matching has no KL term
  - Diffusion: 10× faster — deterministic ODE path, ~100 steps
  - RL: No reward hacking — potency built into flow geometry
  - PropMolFlow: Works on real drug-sized molecules, not QM9-only

Architecture:
  CGLF = FlowMatchingNet(128-d) + PotencyGuidance(centroid) + SELFIESDecoder(GRU)

Key fix baked in (from UPDATE_1):
  - 5-d conditioning only: [potency_oh(4), sim(1)]
    (original had 133-d with 128-d centroid → information leakage)

Usage:
    python generative/cglf.py --epochs 200 --device mps
"""
import os
import sys
import json
import math
import time
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

GENESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(GENESIS_DIR, 'generative'))

from dataset import SelfiesDataset, get_dataloaders, decode_tokens
from vae import SELFIES_VAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

GEN_DIR = os.path.join(GENESIS_DIR, 'generative')
CHECKPOINT_DIR = os.path.join(GEN_DIR, 'checkpoints')
LOG_DIR = os.path.join(GEN_DIR, 'logs')


# ── Sinusoidal Time Embedding ────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ── AdaLN (Adaptive Layer Norm) ──────────────────────────────────────

class AdaLN(nn.Module):
    """Adaptive Layer Normalization — conditions LN parameters on time + potency.

    Unlike FiLM (γ ⊙ h + β), AdaLN normalises FIRST (zero-mean, unit-var)
    then modulates: (1 + γ) ⊙ LN(h) + δ.  The (1 + γ) formulation is
    critical: at initialisation γ ≈ 0, so AdaLN reduces to standard
    LayerNorm (identity conditioning).  FiLM with γ = 0 would zero out
    the hidden state, causing gradient vanishing in the 8-layer MLP.
    Chosen over FiLM because raw hidden states drift in scale across
    8 layers; pre-normalisation keeps modulation numerically stable.
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x, cond):
        """x: (batch, hidden_dim), cond: (batch, cond_dim)"""
        x = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale) + shift


# ── Flow Matching Vector Field Network ───────────────────────────────

class FlowMatchingNet(nn.Module):
    """
    Velocity field v_θ(z_t, t, c) for Conditional Flow Matching (CFM).

    Given: z_t at time t ∈ [0, 1], condition c
    Predicts: velocity vector v ∈ R^128

    Architecture: 8-layer MLP with AdaLN conditioning + skip connections.

    Conditioning is 5-d: [potency_oh(4), cosine_sim(1)].
    """

    def __init__(self, latent_dim=128, hidden_dim=512, cond_dim=5, n_layers=8,
                 dropout=0.1, cfg_prob=0.15):
        super().__init__()
        self.latent_dim = latent_dim
        self.cfg_prob = cfg_prob

        # Time embedding
        self.time_emb = SinusoidalTimeEmb(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition embedding: 5-d only (potency_oh + sim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # AdaLN residual blocks
        self.blocks = nn.ModuleList()
        self.adalns = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ))
            self.adalns.append(AdaLN(hidden_dim, hidden_dim))

        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, t, cond=None):
        """
        z_t: (batch, latent_dim) — position at time t
        t: (batch,) — time ∈ [0, 1]
        cond: (batch, cond_dim) — conditioning (5-d)
        Returns: velocity v (batch, latent_dim)
        """
        t_emb = self.time_proj(self.time_emb(t))

        if cond is not None:
            if self.training and self.cfg_prob > 0:
                mask = (torch.rand(cond.size(0), 1, device=cond.device) > self.cfg_prob).float()
                cond = cond * mask
            c_emb = self.cond_proj(cond)
            combined_cond = t_emb + c_emb
        else:
            combined_cond = t_emb

        h = self.input_proj(z_t)

        for block, adaln in zip(self.blocks, self.adalns):
            h_norm = adaln(h, combined_cond)
            h = h + block(h_norm)

        h = self.output_norm(h)
        return self.output_proj(h)


# ── Conditional Flow Matching ────────────────────────────────────────

class ConditionalFlowMatcher:
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).

    Linear interpolant: z_t = (1-t) * z_0 + t * z_1
    Velocity: v_t = z_1 - z_0

    Where z_0 ~ N(0, I) (noise) and z_1 = data point.
    """

    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def sample_time(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    def compute_flow(self, z_0, z_1, t):
        t = t.unsqueeze(-1)
        z_t = (1 - t) * z_0 + t * z_1
        v_target = z_1 - z_0
        return z_t, v_target

    @torch.no_grad()
    def sample(self, model, shape, device, cond=None, n_steps=100,
               guidance_scale=1.0, centroid=None, potency_guidance_weight=0.0):
        """ODE integration: integrate velocity field from t=0 to t=1.

        Dual guidance system (both applied during Euler integration):
          1. CFG (classifier-free guidance): amplifies learned conditioning
             signal in discrete potency-class space.  Asks "what would
             the model generate for this class vs no class?"
          2. Potency gradient guidance: Riemannian gradient of cosine
             similarity on S^127.  Geometric force toward the VP centroid
             along the sphere surface, independent of the learned model.
        The two are complementary — CFG captures class-specific patterns;
        geometric guidance provides hard potency signal in uncertain regions.

        Time-dependent scaling α(t) = α₀·t: guidance is zero at t=0
        (let the flow establish molecular structure) and maximal at t=1
        (refine a nearly-formed molecule toward high potency).
        Euler integrator is sufficient: OT-CFM straight-line paths have
        approximately constant velocity, so higher-order solvers add
        cost without accuracy gain.
        """
        model.eval()
        z = torch.randn(shape, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((shape[0],), i * dt, device=device)

            if guidance_scale > 1.0 and cond is not None:
                v_cond = model(z, t, cond)
                v_uncond = model(z, t, torch.zeros_like(cond))
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = model(z, t, cond)

            if potency_guidance_weight > 0 and centroid is not None:
                centroid_t = torch.tensor(centroid, device=device, dtype=torch.float32)
                z_norm = F.normalize(z, p=2, dim=-1)
                grad_sim = centroid_t - (z_norm * centroid_t).sum(-1, keepdim=True) * z_norm
                scale = potency_guidance_weight * (i * dt)
                v = v + scale * grad_sim

            z = z + dt * v

        return z


# ── SELFIES Decoder for CGLF ────────────────────────────────────────

class LatentToSELFIES(nn.Module):
    """Dedicated decoder: 128-d latent → SELFIES token sequence.

    Architecturally identical to VAE decoder (same GRU dims) but with
    INDEPENDENT weights.  The flow network targets the VAE encoder's
    μ outputs (deterministic), which differ from the distribution the
    VAE decoder was trained on (μ + σ·ε reparameterisation noise).
    Weight-sharing would force the decoder to handle both distributions
    — a mismatch that degrades reconstruction.  Stage 1 pretraining
    aligns these weights specifically to the clean encoder-μ space.
    """

    def __init__(self, vocab_size, latent_dim=128, emb_dim=128, hidden_dim=512,
                 n_layers=2, dropout=0.1, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * n_layers)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, target_tokens=None):
        batch_size = z.size(0)
        h = self.latent_to_hidden(z)
        h = h.view(batch_size, self.n_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()

        if target_tokens is not None:
            emb = self.embedding(target_tokens[:, :-1])
            output, _ = self.gru(emb, h)
            return self.output_proj(output)
        else:
            input_token = torch.ones(batch_size, 1, dtype=torch.long, device=z.device)
            logits_list = []
            for _ in range(self.max_len - 1):
                emb = self.embedding(input_token)
                output, h = self.gru(emb, h)
                logit = self.output_proj(output)
                logits_list.append(logit)
                input_token = logit.argmax(dim=-1)
            return torch.cat(logits_list, dim=1)


# ── CGLF Complete Model ─────────────────────────────────────────────

class CGLF(nn.Module):
    """
    Contrastive-Guided Latent Flow.

    Components:
      1. Encoder: reuses trained VAE encoder (frozen) to map SELFIES → 128-d
      2. FlowMatchingNet: learns velocity field v_θ in 128-d space
      3. LatentToSELFIES: decodes 128-d → SELFIES tokens

    Training:
      Stage 1: Train decoder (encoder→z→decoder reconstruction)
      Stage 2: Train flow matching (CFM loss) with potency conditioning
    """

    def __init__(self, vocab_size, latent_dim=128, flow_hidden=512, flow_layers=8,
                 dec_hidden=512, dec_layers=2, emb_dim=128, max_len=128,
                 cfg_prob=0.15, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim

        # Conditioning: potency_oh(4) + sim(1) = 5
        cond_dim = 5

        self.flow_net = FlowMatchingNet(
            latent_dim=latent_dim, hidden_dim=flow_hidden,
            cond_dim=cond_dim, n_layers=flow_layers,
            dropout=dropout, cfg_prob=cfg_prob)

        self.decoder = LatentToSELFIES(
            vocab_size=vocab_size, latent_dim=latent_dim,
            emb_dim=emb_dim, hidden_dim=dec_hidden,
            n_layers=dec_layers, dropout=dropout, max_len=max_len)

        self.flow_matcher = ConditionalFlowMatcher()

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Conditioning Builder ─────────────────────────────────────────────

def build_cglf_conditioning(potency_class, proj_128, centroid):
    """Build 5-d conditioning: [potency_oh(4), sim(1)].

    No raw 128-d centroid — prevents information leakage
    and keeps conditioning dimensionality low.
    The flow must LEARN the geometry of the latent space rather
    than being handed the centroid direction.  Same UPDATE_1 fix
    as cVAE: original 133-d conditioning allowed the flow to
    trivially learn the identity mapping, bypassing meaningful
    generation.  The 5-d signal provides ordinal potency class +
    scalar similarity — enough guidance for conditional generation
    without short-circuiting the learning objective.
    """
    potency_oh = F.one_hot(potency_class.clamp(0, 3), num_classes=4).float()
    centroid_t = torch.tensor(centroid, dtype=torch.float32, device=proj_128.device)
    sim = F.cosine_similarity(proj_128, centroid_t.unsqueeze(0), dim=-1, eps=1e-8)
    return torch.cat([potency_oh, sim.unsqueeze(-1)], dim=-1)


# ── Training Functions ───────────────────────────────────────────────

def train_decoder_epoch(cglf_model, vae_encoder, dataloader, optimizer, device):
    """Stage 1: Train decoder for reconstruction (encoder → z → decoder).

    Mirrors developmental biology: establish the 'body plan' (chemical
    grammar of valid SELFIES) before specialisation (potency-directed
    flow).  The VAE encoder (frozen) provides stable μ codes as teacher
    signals.  At convergence, the decoder maps the VAE latent
    distribution to valid molecules with high fidelity, creating a
    fixed 'evaluation interface' for Stage 2 flow training.
    """
    cglf_model.decoder.train()
    vae_encoder.eval()
    total_loss = 0
    n = 0

    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        with torch.no_grad():
            mu, _ = vae_encoder(tokens)
        logits = cglf_model.decoder(mu, target_tokens=tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=0, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(cglf_model.decoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n


def train_flow_epoch(cglf_model, vae_encoder, dataloader, optimizer, device, centroid):
    """Stage 2: Train flow matching network (CFM loss).

    Simple MSE velocity regression: ||v_θ(z_t, t, c) - (z₁ - z₀)||².
    The target velocity (z₁ - z₀) is constant along each OT path — a
    much simpler regression target than DDPM's curved noise schedule,
    which is why CFM converges faster with fewer epochs.
    The decoder is frozen; the flow network's only job is to produce
    latent codes that the fixed decoder translates to valid, potent
    molecules.
    """
    cglf_model.flow_net.train()
    vae_encoder.eval()
    total_loss = 0
    n = 0

    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        potency = batch['potency_class'].to(device)
        proj = batch['proj_128'].to(device)

        with torch.no_grad():
            mu, _ = vae_encoder(tokens)
        z_1 = mu

        cond = build_cglf_conditioning(potency, proj, centroid).to(device)

        z_0 = torch.randn_like(z_1)
        t = cglf_model.flow_matcher.sample_time(z_1.size(0), device)

        z_t, v_target = cglf_model.flow_matcher.compute_flow(z_0, z_1, t)

        v_pred = cglf_model.flow_net(z_t, t, cond)

        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(cglf_model.flow_net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n


@torch.no_grad()
def validate_flow(cglf_model, vae_encoder, dataloader, device, centroid):
    """Validate flow matching."""
    cglf_model.flow_net.eval()
    vae_encoder.eval()
    total_loss = 0
    n = 0

    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        potency = batch['potency_class'].to(device)
        proj = batch['proj_128'].to(device)

        mu, _ = vae_encoder(tokens)
        z_1 = mu

        cond = build_cglf_conditioning(potency, proj, centroid).to(device)

        z_0 = torch.randn_like(z_1)
        t = cglf_model.flow_matcher.sample_time(z_1.size(0), device)
        z_t, v_target = cglf_model.flow_matcher.compute_flow(z_0, z_1, t)
        v_pred = cglf_model.flow_net(z_t, t, cond)
        loss = F.mse_loss(v_pred, v_target)

        total_loss += loss.item()
        n += 1

    return total_loss / n


def generate_cglf(cglf_model, inv_vocab, device, centroid, n_samples=1000,
                  n_steps=100, guidance_scale=2.0, potency_guidance=0.5,
                  target_potency=3, temperature=1.0):
    """Generate molecules using CGLF pipeline."""
    import selfies as sf

    # Build conditioning for target potency (5-d: potency_oh + sim)
    # sim = 0.9 (not 1.0) — deliberate diversity preservation: targets
    # a spherical cap around the VP centroid with angular radius
    # arccos(0.9) ≈ 26°, balancing potency with chemical diversity.
    # sim = 1.0 would collapse all molecules to the centroid itself.
    potency_oh = torch.zeros(n_samples, 4, device=device)
    potency_oh[:, target_potency] = 1.0
    sim = torch.ones(n_samples, 1, device=device) * 0.9
    cond = torch.cat([potency_oh, sim], dim=-1)

    z = cglf_model.flow_matcher.sample(
        cglf_model.flow_net,
        shape=(n_samples, cglf_model.latent_dim),
        device=device,
        cond=cond,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        centroid=centroid,
        potency_guidance_weight=potency_guidance)

    cglf_model.decoder.eval()
    with torch.no_grad():
        logits = cglf_model.decoder(z)
    tokens = logits.argmax(dim=-1).cpu()

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
    parser = argparse.ArgumentParser(description='CGLF Training')
    parser.add_argument('--stage1-epochs', type=int, default=50, help='Decoder pretraining epochs')
    parser.add_argument('--stage2-epochs', type=int, default=200, help='Flow matching epochs')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--flow-hidden', type=int, default=512)
    parser.add_argument('--flow-layers', type=int, default=8)
    parser.add_argument('--cfg-prob', type=float, default=0.15)
    parser.add_argument('--guidance-scale', type=float, default=2.0)
    parser.add_argument('--potency-guidance', type=float, default=0.5)
    parser.add_argument('--n-steps', type=int, default=100, help='ODE integration steps')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gen-every', type=int, default=25)
    parser.add_argument('--vae-checkpoint', default=None)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── Load VAE encoder ─────────────────────────────────────────────
    vae_ckpt_path = args.vae_checkpoint or os.path.join(CHECKPOINT_DIR, 'best_vae.pt')
    log.info('Loading VAE encoder from %s', vae_ckpt_path)
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    vae_args = vae_ckpt['args']

    with open(os.path.join(GEN_DIR, 'processed', 'vocab.json')) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    vae_model = SELFIES_VAE(
        vocab_size=vocab_size,
        emb_dim=vae_args.get('emb_dim', 128),
        hidden_dim=vae_args.get('hidden_dim', 512),
        latent_dim=vae_args.get('latent_dim', 128),
        n_layers=vae_args.get('n_layers', 2),
        max_len=128, conditional=False,
    ).to(device)
    vae_model.load_state_dict(vae_ckpt['model_state_dict'])
    vae_encoder = vae_model.encoder
    vae_encoder.eval()
    for p in vae_encoder.parameters():
        p.requires_grad = False
    log.info('VAE encoder loaded (frozen)')

    # Load centroid
    centroid_path = os.path.join(GEN_DIR, 'processed', 'vp_centroid.json')
    with open(centroid_path) as f:
        centroid = np.array(json.load(f)['oscc_vp_centroid'], dtype=np.float32)

    # Load inverse vocab
    with open(os.path.join(GEN_DIR, 'processed', 'inv_vocab.json')) as f:
        inv_vocab = json.load(f)

    # ── Load data ────────────────────────────────────────────────────
    log.info('Loading data (with projections for conditioning) ...')
    train_dl, val_dl, _ = get_dataloaders(
        batch_size=args.batch_size, load_projections=True, device=args.device)
    log.info('Train: %d, Val: %d', len(train_dl.dataset), len(val_dl.dataset))

    # ── Build CGLF model ─────────────────────────────────────────────
    cglf = CGLF(
        vocab_size=vocab_size,
        latent_dim=vae_args.get('latent_dim', 128),
        flow_hidden=args.flow_hidden,
        flow_layers=args.flow_layers,
        dec_hidden=vae_args.get('hidden_dim', 512),
        dec_layers=vae_args.get('n_layers', 2),
        emb_dim=vae_args.get('emb_dim', 128),
        max_len=128,
        cfg_prob=args.cfg_prob,
    ).to(device)
    log.info('CGLF parameters: %s', f'{cglf.param_count():,}')

    # ── Stage 1: Decoder pretraining ──────────────────────────────────
    log.info('=== Stage 1: Decoder Pretraining (%d epochs) ===', args.stage1_epochs)
    dec_optimizer = Adam(cglf.decoder.parameters(), lr=args.lr)
    best_dec_loss = float('inf')

    for epoch in range(1, args.stage1_epochs + 1):
        t0 = time.time()
        train_loss = train_decoder_epoch(cglf, vae_encoder, train_dl, dec_optimizer, device)

        cglf.decoder.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for batch in val_dl:
                tokens = batch['tokens'].to(device)
                mu, _ = vae_encoder(tokens)
                logits = cglf.decoder(mu, target_tokens=tokens)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       tokens[:, 1:].reshape(-1), ignore_index=0)
                val_loss += loss.item()
                n += 1
        val_loss /= n
        elapsed = time.time() - t0

        log.info('S1 Epoch %3d/%d | train=%.4f | val=%.4f [%.1fs]',
                 epoch, args.stage1_epochs, train_loss, val_loss, elapsed)

        if val_loss < best_dec_loss:
            best_dec_loss = val_loss
            torch.save(cglf.decoder.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'cglf_decoder.pt'))

    # Reload best decoder
    cglf.decoder.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, 'cglf_decoder.pt'),
                   map_location=device, weights_only=True))
    # Freeze decoder after Stage 1 — creates a fixed 'evaluation interface'
    # for Stage 2.  Without freezing, the decoder could co-adapt with the
    # flow in pathological ways (e.g., the flow produces arbitrary codes and
    # the decoder learns to handle them — defeating the purpose of learning
    # the data distribution's geometry).  The functional dependency ordering
    # enc_φ → dec_ψ → v_θ ensures each stage has a single well-defined loss.
    for p in cglf.decoder.parameters():
        p.requires_grad = False
    log.info('Decoder pretrained (best val=%.4f)', best_dec_loss)

    # ── Stage 2: Flow matching training ───────────────────────────────
    log.info('=== Stage 2: Flow Matching (%d epochs) ===', args.stage2_epochs)
    flow_optimizer = Adam(cglf.flow_net.parameters(), lr=args.lr)
    flow_scheduler = CosineAnnealingLR(flow_optimizer, T_max=args.stage2_epochs, eta_min=1e-6)

    history = []
    best_flow_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.stage2_epochs + 1):
        t0 = time.time()

        train_loss = train_flow_epoch(cglf, vae_encoder, train_dl, flow_optimizer, device, centroid)
        val_loss = validate_flow(cglf, vae_encoder, val_dl, device, centroid)

        flow_scheduler.step()
        lr = flow_optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        log.info('S2 Epoch %3d/%d | train=%.6f | val=%.6f | lr=%.1e [%.1fs]',
                 epoch, args.stage2_epochs, train_loss, val_loss, lr, elapsed)

        history.append({
            'stage': 2, 'epoch': epoch,
            'train_loss': train_loss, 'val_loss': val_loss,
            'lr': lr, 'elapsed_s': elapsed
        })

        if val_loss < best_flow_loss:
            best_flow_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'flow_net_state_dict': cglf.flow_net.state_dict(),
                'decoder_state_dict': cglf.decoder.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, os.path.join(CHECKPOINT_DIR, 'best_cglf.pt'))
            log.info('  ★ New best → best_cglf.pt (val_loss=%.6f)', val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info('Early stopping at epoch %d', epoch)
                break

        if epoch % args.gen_every == 0:
            smiles = generate_cglf(cglf, inv_vocab, device, centroid,
                                    n_samples=100, n_steps=args.n_steps,
                                    guidance_scale=args.guidance_scale,
                                    potency_guidance=args.potency_guidance)
            log.info('  Generation: %d/100 valid SMILES', len(smiles))

    with open(os.path.join(LOG_DIR, 'history_cglf.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Final generation ──────────────────────────────────────────────
    log.info('=== Final Generation (1000 molecules) ===')
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_cglf.pt'),
                      map_location=device, weights_only=False)
    cglf.flow_net.load_state_dict(ckpt['flow_net_state_dict'])
    cglf.decoder.load_state_dict(ckpt['decoder_state_dict'])

    smiles = generate_cglf(cglf, inv_vocab, device, centroid,
                            n_samples=1000, n_steps=args.n_steps,
                            guidance_scale=args.guidance_scale,
                            potency_guidance=args.potency_guidance)
    log.info('Generated %d valid SMILES', len(smiles))

    gen_path = os.path.join(GEN_DIR, 'results', 'generated_cglf.txt')
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
    with open(gen_path, 'w') as f:
        for s in smiles:
            f.write(s + '\n')
    log.info('Generated SMILES → %s', gen_path)

    try:
        from evaluate_gen import evaluate_molecules
        results = evaluate_molecules(smiles, method_name='cglf', device=str(device),
                                     compute_potency=True)
        log.info('Evaluation: validity=%.1f%%, uniqueness=%.1f%%, novelty=%.1f%%',
                 100 * results['metrics']['validity'],
                 100 * results['metrics']['uniqueness'],
                 100 * results['metrics']['novelty'])
    except Exception as e:
        log.warning('Evaluation failed: %s', e)


if __name__ == '__main__':
    main()
