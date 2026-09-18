#!/usr/bin/env python3
"""
Phase 3, Steps 6-7 — Latent Diffusion Model for Molecular Generation.

**CRITICAL NEGATIVE FINDING**: This is the worst-performing method in the
GENESIS pipeline. Mode collapse on 2/3 random seeds produces uniqueness
as low as 10.1% and even negative potency. See Algorithm_Design.md §10.13
for full root-cause analysis. The failure is data-regime-dependent: latent
diffusion requires 10–100× more training data than GENESIS's 34K compounds.

Operates in the VAE's 128-d latent space:
  1. Train VAE first (vae.py) to get encoder/decoder
  2. Encode training set → 128-d latent vectors (μ only, deterministic)
  3. Train denoising diffusion model on latent vectors
  4. Sample: noise → diffusion denoise → VAE decode → SELFIES → SMILES

Conditioning via classifier-free guidance (CFG, w=2.0) on a 5-d vector:
  [potency_class one-hot (4-d), cosine_sim_to_VP_centroid (1-d)]

Key fixes baked in (from UPDATE_1):
  - Cosine noise schedule (was linear → poor sample quality)
  - Float32 enforcement on MPS (was float64 → NaN overflow)

Usage:
    python generative/diffusion.py --epochs 200 --device mps
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
from vae import SELFIES_VAE, build_conditioning

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

GEN_DIR = os.path.join(GENESIS_DIR, 'generative')
CHECKPOINT_DIR = os.path.join(GEN_DIR, 'checkpoints')
LOG_DIR = os.path.join(GEN_DIR, 'logs')


# ── Sinusoidal Timestep Embedding ────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: (batch,) float in [0, 1]"""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ── Denoising Network ───────────────────────────────────────────────

class DenoiseNet(nn.Module):
    """MLP denoiser for 128-d latent diffusion.

    Predicts noise ε given noisy latent z_t, timestep t, and optional condition c.
    Architecture: 6-layer MLP with skip connections (~1.6M params).

    Input fusion is additive: h = input_proj(z_t) + time_proj(SinEmb(t)) + cond_proj(c).
    All three signals contribute equally to the shared 512-d space without
    cross-attention or FiLM gating.  This keeps parameter count low but
    limits the model's ability to learn input-dependent interactions between
    noise level and conditioning — a likely contributor to mode collapse
    (see §10.13.2).  For comparison, CGLF's AdaLN (§12.3) provides
    timestep-adaptive normalisation that is more expressive.
    """

    def __init__(self, latent_dim=128, hidden_dim=512, cond_dim=5, n_layers=6,
                 dropout=0.1, cfg_prob=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.cfg_prob = cfg_prob

        self.time_emb = SinusoidalTimeEmb(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, t, cond=None):
        """
        z_t: (batch, latent_dim) — noisy latent
        t: (batch,) — timestep in [0, 1]
        cond: (batch, cond_dim) — conditioning (optional)
        Returns: predicted noise ε (batch, latent_dim)
        """
        h = self.input_proj(z_t)

        t_emb = self.time_emb(t)
        t_emb = self.time_proj(t_emb)
        h = h + t_emb

        if cond is not None:
            if self.training and self.cfg_prob > 0:
                mask = torch.rand(cond.size(0), 1, device=cond.device) > self.cfg_prob
                cond = cond * mask.float()
            c_emb = self.cond_proj(cond)
            h = h + c_emb

        for block in self.blocks:
            h = h + block(h)

        h = self.output_norm(h)
        return self.output_proj(h)


# ── Diffusion Schedule ───────────────────────────────────────────────

class GaussianDiffusion:
    """DDPM with cosine noise schedule (Nichol & Dhariwal, 2021).

    Uses cosine schedule for smoother āα_cumprod curve → better
    sample quality & diversity compared to linear schedule.
    The linear schedule destroys signal too fast early and too slowly
    late, creating a U-shaped difficulty curve for the denoiser.
    The cosine schedule distributes information loss more evenly —
    roughly equal amounts of signal removed at each timestep.

    All tensors forced to float32 for MPS compatibility.  Apple's MPS
    backend can silently promote to float64 or use float16 intermediates,
    causing NaN overflow in posterior variance terms as small as 1e-8.
    """

    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.n_steps = n_steps
        self.device = device

        # Cosine noise schedule
        s = 0.008
        steps = torch.arange(n_steps + 1, dtype=torch.float32, device=device)
        f_t = torch.cos(((steps / n_steps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=0.999).float()
        alphas_cumprod = alphas_cumprod[1:].float()
        alphas = 1.0 - betas
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device, dtype=torch.float32),
                                         alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, z_0, t, noise=None):
        """Forward diffusion: add noise to z_0 at timestep t.

        z_t = sqrt(āα_t) * z_0 + sqrt(1 - āα_t) * ε
        This is a closed-form shortcut — we can jump directly to any
        noise level t without iterating through all intermediate steps.
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        # Enforce float32 for MPS
        z_0 = z_0.float()
        noise = noise.float()
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1).float()
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).float()
        return sqrt_alpha * z_0 + sqrt_one_minus_alpha * noise, noise

    @torch.no_grad()
    def p_sample(self, model, z_t, t, cond=None, guidance_scale=1.0):
        """Reverse diffusion: denoise one step.

        When guidance_scale > 1.0, uses Classifier-Free Guidance (CFG):
        ε̂ = ε_uncond + w * (ε_cond - ε_uncond)
        Intuitively: 'generate a molecule, but make it w× more like what
        I asked for than what I'd produce without guidance.'
        With w=2.0, the conditioning effect is doubled.
        """
        z_t = z_t.float()
        t_tensor = torch.full((z_t.size(0),), t, device=z_t.device, dtype=torch.float32)
        t_normalized = t_tensor / self.n_steps

        if guidance_scale > 1.0 and cond is not None:
            eps_cond = model(z_t, t_normalized, cond)
            eps_uncond = model(z_t, t_normalized, torch.zeros_like(cond))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(z_t, t_normalized, cond)

        beta = self.betas[t].float()
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].float()
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].float()

        z_prev = sqrt_recip_alpha * (z_t - beta / sqrt_one_minus_alpha * eps)

        if t > 0:
            noise = torch.randn_like(z_t)
            z_prev = z_prev + torch.sqrt(self.posterior_variance[t].float()) * noise

        return z_prev

    @torch.no_grad()
    def sample(self, model, shape, cond=None, guidance_scale=1.0):
        """Full reverse diffusion: noise → clean latent."""
        z = torch.randn(shape, device=self.device, dtype=torch.float32)

        for t in reversed(range(self.n_steps)):
            z = self.p_sample(model, z, t, cond, guidance_scale)

        return z


# ── Latent Dataset ───────────────────────────────────────────────────

class LatentDataset(Dataset):
    """Dataset of VAE-encoded latent vectors."""

    def __init__(self, latents, potency_classes, pchembl_vals, proj_128=None):
        self.latents = latents
        self.potency_classes = potency_classes
        self.pchembl = pchembl_vals
        self.proj_128 = proj_128

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        item = {
            'z': torch.tensor(self.latents[idx], dtype=torch.float32),
            'potency_class': torch.tensor(self.potency_classes[idx], dtype=torch.long),
            'pchembl': torch.tensor(self.pchembl[idx], dtype=torch.float32),
        }
        if self.proj_128 is not None:
            item['proj_128'] = torch.tensor(self.proj_128[idx], dtype=torch.float32)
        return item


def encode_dataset(vae_model, dataloader, device):
    """Encode all training data through VAE encoder to get latent vectors.

    WARNING: Uses μ (mean) only, without reparameterisation.  The resulting
    latent 'distribution' is a sparse cloud of ~34K deterministic points in
    128-d space — not a smooth continuous density.  This sparsity is a root
    cause of the diffusion model's mode collapse (§10.13.2): the denoiser
    must navigate between isolated points rather than a dense manifold.
    """
    vae_model.eval()
    all_mu = []
    all_potency = []
    all_pchembl = []

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            mu, logvar = vae_model.encoder(tokens)
            all_mu.append(mu.cpu().numpy())
            all_potency.append(batch['potency_class'].numpy())
            all_pchembl.append(batch['pchembl'].numpy())

    return (np.concatenate(all_mu), np.concatenate(all_potency),
            np.concatenate(all_pchembl))


def build_cond_from_latent_batch(batch, centroid=None):
    """Build 5-d conditioning vector for latent diffusion.

    Dimensions:
      [0:4] — potency class one-hot (VNP=0, NP=1, P=2, VP=3)
      [4]   — cosine similarity to VP centroid in 128-d contrastive space
    At generation time, cond = [0,0,0,1, 0.9] targets Very Potent molecules
    with 90% centroid similarity.
    """
    potency = batch['potency_class']
    potency_oh = F.one_hot(potency.clamp(0, 3), num_classes=4).float()

    if centroid is not None and 'proj_128' in batch:
        proj = batch['proj_128']
        centroid_t = torch.tensor(centroid, dtype=torch.float32, device=proj.device)
        sim = F.cosine_similarity(proj, centroid_t.unsqueeze(0), dim=-1, eps=1e-8)
        sim = sim.unsqueeze(-1)
    else:
        sim = torch.zeros(potency.size(0), 1, device=potency.device)

    return torch.cat([potency_oh, sim], dim=-1)


# ── Training Loop ────────────────────────────────────────────────────

def train_diffusion_epoch(model, diffusion, loader, optimizer, device, centroid=None):
    """Train one epoch of diffusion denoising."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        z_0 = batch['z'].to(device).float()
        cond = build_cond_from_latent_batch(batch, centroid).to(device)

        t = torch.randint(0, diffusion.n_steps, (z_0.size(0),), device=device)
        t_normalized = t.float() / diffusion.n_steps

        noise = torch.randn_like(z_0)
        z_t, _ = diffusion.q_sample(z_0, t, noise)

        eps_pred = model(z_t, t_normalized, cond)
        loss = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate_diffusion(model, diffusion, loader, device, centroid=None):
    """Validate diffusion model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        z_0 = batch['z'].to(device).float()
        cond = build_cond_from_latent_batch(batch, centroid).to(device)

        t = torch.randint(0, diffusion.n_steps, (z_0.size(0),), device=device)
        t_normalized = t.float() / diffusion.n_steps

        noise = torch.randn_like(z_0)
        z_t, _ = diffusion.q_sample(z_0, t, noise)

        eps_pred = model(z_t, t_normalized, cond)
        loss = F.mse_loss(eps_pred, noise)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def generate_molecules(diffusion, denoise_model, vae_model, device, inv_vocab,
                       n_samples=1000, guidance_scale=2.0, target_potency=3,
                       centroid=None, n_steps=200):
    """Generate molecules: diffusion sample → VAE decode → SELFIES → SMILES.

    Uses reduced sampling steps (default 200 vs 1000 training steps) for
    5× speedup.  The quality gap between 200 and 1000 steps is modest
    because the cosine schedule concentrates most information in the
    middle timesteps; the coarser step grid skips mostly-redundant
    high-noise and low-noise refinements.
    """
    import selfies as sf

    potency_oh = torch.zeros(n_samples, 4, device=device)
    potency_oh[:, target_potency] = 1.0
    sim = torch.ones(n_samples, 1, device=device) * 0.9
    cond = torch.cat([potency_oh, sim], dim=-1)

    fast_diffusion = GaussianDiffusion(n_steps=n_steps, device=device)

    z = fast_diffusion.sample(denoise_model, (n_samples, denoise_model.latent_dim),
                              cond=cond, guidance_scale=guidance_scale)

    vae_model.eval()
    with torch.no_grad():
        logits = vae_model.decoder(z)
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
    parser = argparse.ArgumentParser(description='Latent Diffusion Training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n-steps', type=int, default=1000, help='Diffusion timesteps')
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--cfg-prob', type=float, default=0.1, help='CFG dropout prob')
    parser.add_argument('--guidance-scale', type=float, default=2.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gen-every', type=int, default=25)
    parser.add_argument('--vae-checkpoint', default=None, help='Path to VAE checkpoint')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── Load VAE ─────────────────────────────────────────────────────
    vae_ckpt_path = args.vae_checkpoint or os.path.join(CHECKPOINT_DIR, 'best_vae.pt')
    log.info('Loading VAE from %s', vae_ckpt_path)
    vae_ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    vae_args = vae_ckpt['args']

    with open(os.path.join(GEN_DIR, 'processed', 'vocab.json')) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    max_len = 128

    vae_model = SELFIES_VAE(
        vocab_size=vocab_size,
        emb_dim=vae_args.get('emb_dim', 128),
        hidden_dim=vae_args.get('hidden_dim', 512),
        latent_dim=vae_args.get('latent_dim', 128),
        n_layers=vae_args.get('n_layers', 2),
        max_len=max_len,
        conditional=False,
    ).to(device)
    vae_model.load_state_dict(vae_ckpt['model_state_dict'])
    vae_model.eval()
    log.info('VAE loaded (epoch %d, val_loss=%.4f)', vae_ckpt['epoch'], vae_ckpt['val_loss'])

    # ── Load centroid ────────────────────────────────────────────────
    centroid_path = os.path.join(GEN_DIR, 'processed', 'vp_centroid.json')
    with open(centroid_path) as f:
        centroid = np.array(json.load(f)['oscc_vp_centroid'], dtype=np.float32)

    # ── Encode training data ──────────────────────────────────────────
    log.info('Encoding training data through VAE ...')
    train_dl, val_dl, _ = get_dataloaders(batch_size=args.batch_size)

    train_latents, train_potency, train_pchembl = encode_dataset(vae_model, train_dl, device)
    val_latents, val_potency, val_pchembl = encode_dataset(vae_model, val_dl, device)
    log.info('Encoded: train=%d, val=%d (latent_dim=%d)',
             len(train_latents), len(val_latents), train_latents.shape[1])

    train_lat_ds = LatentDataset(train_latents, train_potency, train_pchembl)
    val_lat_ds = LatentDataset(val_latents, val_potency, val_pchembl)
    train_lat_dl = DataLoader(train_lat_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_lat_dl = DataLoader(val_lat_ds, batch_size=args.batch_size, shuffle=False)

    # ── Build diffusion model ─────────────────────────────────────────
    denoise_model = DenoiseNet(
        latent_dim=vae_args.get('latent_dim', 128),
        hidden_dim=args.hidden_dim,
        cond_dim=5,
        n_layers=args.n_layers,
        cfg_prob=args.cfg_prob,
    ).to(device)
    n_params = sum(p.numel() for p in denoise_model.parameters() if p.requires_grad)
    log.info('DenoiseNet: %s params', f'{n_params:,}')

    diffusion = GaussianDiffusion(n_steps=args.n_steps, device=device)
    optimizer = Adam(denoise_model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    with open(os.path.join(GEN_DIR, 'processed', 'inv_vocab.json')) as f:
        inv_vocab = json.load(f)

    # ── Training loop ─────────────────────────────────────────────────
    history = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_diffusion_epoch(denoise_model, diffusion, train_lat_dl,
                                           optimizer, device, centroid)
        val_loss = validate_diffusion(denoise_model, diffusion, val_lat_dl,
                                      device, centroid)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        log.info('Epoch %3d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.1e [%.1fs]',
                 epoch, args.epochs, train_loss, val_loss, lr, elapsed)

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'lr': lr, 'elapsed_s': elapsed
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': denoise_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, os.path.join(CHECKPOINT_DIR, 'best_diffusion.pt'))
            log.info('  ★ New best → best_diffusion.pt (val_loss=%.6f)', val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info('Early stopping at epoch %d', epoch)
                break

        if epoch % args.gen_every == 0:
            smiles = generate_molecules(diffusion, denoise_model, vae_model, device,
                                         inv_vocab, n_samples=100,
                                         guidance_scale=args.guidance_scale,
                                         centroid=centroid)
            log.info('  Generation: %d/100 valid SMILES', len(smiles))

    with open(os.path.join(LOG_DIR, 'history_diffusion.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Final generation ──────────────────────────────────────────────
    log.info('=== Final Generation (1000 molecules) ===')
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_diffusion.pt'),
                      map_location=device, weights_only=False)
    denoise_model.load_state_dict(ckpt['model_state_dict'])

    smiles = generate_molecules(diffusion, denoise_model, vae_model, device,
                                 inv_vocab, n_samples=1000,
                                 guidance_scale=args.guidance_scale,
                                 centroid=centroid, n_steps=args.n_steps)
    log.info('Generated %d valid SMILES', len(smiles))

    gen_path = os.path.join(GEN_DIR, 'results', 'generated_diffusion.txt')
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
    with open(gen_path, 'w') as f:
        for s in smiles:
            f.write(s + '\n')
    log.info('Generated SMILES → %s', gen_path)

    try:
        from evaluate_gen import evaluate_molecules
        results = evaluate_molecules(smiles, method_name='diffusion', device=str(device),
                                     compute_potency=True)
        log.info('Evaluation: validity=%.1f%%, uniqueness=%.1f%%, novelty=%.1f%%',
                 100 * results['metrics']['validity'],
                 100 * results['metrics']['uniqueness'],
                 100 * results['metrics']['novelty'])
    except Exception as e:
        log.warning('Evaluation failed: %s', e)

    log.info('Diffusion training complete. Best val_loss=%.6f', best_val_loss)


if __name__ == '__main__':
    main()
