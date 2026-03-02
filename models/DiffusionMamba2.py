"""
DiffusionMamba2 — Diffusion Policy Actor with Mamba2Simple temporal head.
=========================================================================
Mirrors the architecture of RLMamba2Racer but replaces the Gaussian action
distribution with DDPM-based iterative denoising.

Observation pipeline:
    VisionEncoder(CNN) → odom_expand → feature_projection → Mamba2Simple
    → obs_features (temporal context)

Action generation:
    obs_features → ConditionalDenoisingMLP → K-step DDPM reverse process → action
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AuxModels import VisionEncoder, ResidualBlock
from utils.diffusion_utils import *
from mamba_ssm.modules.mamba2_simple import Mamba2Simple


# ===========================================================================
# DiffusionMamba2 — full actor
# ===========================================================================

class DiffusionMamba2(nn.Module):
    """
    Diffusion Policy actor with Mamba2Simple temporal backbone.

    Observation encoder follows the same pattern as ``RLMamba2Racer``:
        VisionEncoder → odom_expand → feature_projection
        → Mamba2Simple (temporal) → pre_mamba_norm / norm_layer
    The final observation embedding is fed into a DDPM denoising MLP that
    iteratively refines pure noise into an action.

    ``create_observation_buffer`` / rolling ``obs_buffer`` match the RLMamba2Racer
    API so the training loop can treat both models identically.
    """
    def __init__(
        self,
        state_dim=4,
        action_dim=2,
        encoder=None,
        num_diffusion_steps=50,
        obs_feature_dim=256,           # matches d_model for Mamba2
        time_emb_dim=32,
        hidden_dims=(768, 768, 768),
        beta_schedule="cosine",
        # Mamba2 config (mirrors RLMamba2Racer defaults)
        d_model=256,
        d_state=16,
        d_conv=4,
        d_head=16,
        expand=2,
        memory_length=48,
        odom_expand=64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.obs_feature_dim = obs_feature_dim
        self.d_model = d_model
        self.memory_length = memory_length

        # --- Vision encoder (1-D CNN for LiDAR) ---
        if encoder is None:
            encoder = VisionEncoder(num_scan_beams=1080)
        self.vision_encoder = encoder
        conv_output_size = self.vision_encoder.output_size

        # --- State expansion ---
        self.odom_expand = nn.Linear(state_dim, odom_expand)

        # --- Feature projection (CNN+odom → d_model) ---
        feature_input_size = conv_output_size + odom_expand
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_input_size, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, d_model),
            nn.ReLU(),
        )

        # --- Mamba2 temporal backbone ---
        self.pre_mamba_norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2Simple(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            headdim=d_head,
            expand=expand,
        )
        self.norm_layer = nn.LayerNorm(d_model)

        # --- Conditional denoising MLP (ε_θ) ---
        self.denoise_net = ConditionalDenoisingMLP(
            action_dim=action_dim,
            obs_feature_dim=obs_feature_dim,
            time_emb_dim=time_emb_dim,
            hidden_dims=hidden_dims,
        )

        # --- DDPM noise schedule ---
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_diffusion_steps)
        else:
            betas = linear_beta_schedule(num_diffusion_steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------
    # Observation buffer (same API as RLMamba2Racer)
    # ------------------------------------------------------------------

    @torch.jit.export
    def create_observation_buffer(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.memory_length, self.d_model, device=device)

    # ------------------------------------------------------------------
    # Observation encoding  (VisionEncoder → Mamba2)
    # ------------------------------------------------------------------

    def encode_observation(self, scan_tensor, state_tensor, obs_buffer,
                           max_speed_estimate=20.0):
        """
        Encode raw observations through CNN + Mamba2 temporal backbone.

        Args:
            scan_tensor:  (B, 1, num_beams)
            state_tensor: (B, state_dim)
            obs_buffer:   (B, memory_length, d_model) – rolling feature buffer

        Returns:
            obs_features: (B, obs_feature_dim)
            obs_buffer:   updated buffer (shifted left, new frame appended)
        """
        batch_size = scan_tensor.shape[0]
        device = scan_tensor.device

        if obs_buffer is None:
            obs_buffer = self.create_observation_buffer(batch_size, device)
        if obs_buffer.device != device:
            obs_buffer = obs_buffer.to(device)

        # Sanitise inputs
        scan_tensor = torch.nan_to_num(scan_tensor, posinf=20.0, neginf=0.0)
        state_tensor = torch.nan_to_num(state_tensor, 0.0)

        # CNN → odom expand → concat → project
        vision_features = self.vision_encoder(scan_tensor)
        vision_features = torch.nan_to_num(vision_features, nan=0.0)
        state_norm = state_tensor / max_speed_estimate
        state_feat = self.odom_expand(state_norm)
        combined = torch.cat([vision_features, state_feat], dim=-1)
        projected = self.feature_projection(combined)
        projected = torch.nan_to_num(projected, nan=0.0)

        # Shift buffer left, append newest frame
        obs_buffer = torch.cat([
            obs_buffer[:, 1:, :],
            projected.unsqueeze(1),
        ], dim=1)

        # Mamba2 temporal pass
        obs_normed = self.pre_mamba_norm(obs_buffer)
        obs_normed = torch.clamp(obs_normed, -10.0, 10.0)
        mamba_out = self.mamba(obs_normed)
        mamba_out = torch.nan_to_num(mamba_out, nan=0.0)
        obs_features = self.norm_layer(mamba_out[:, -1, :])
        obs_features = torch.nan_to_num(obs_features, nan=0.0)

        return obs_features, obs_buffer

    # ------------------------------------------------------------------
    # Forward / reverse diffusion helpers
    # ------------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: q(a^k | a^0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_noise(self, noisy_action, obs_features, t):
        """ε_θ(a^k, k, o)"""
        return self.denoise_net(noisy_action, obs_features, t)

    @torch.no_grad()
    def p_mean_variance(self, x_t, obs_features, t):
        pred_noise = self.predict_noise(x_t, obs_features, t)
        sqrt_recip = extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        mean = sqrt_recip * (x_t - beta_t / sqrt_one_minus * pred_noise)
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(self, x_t, obs_features, t):
        mean, var, _ = self.p_mean_variance(x_t, obs_features, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_action(self, obs_features, deterministic=False):
        B = obs_features.shape[0]
        device = obs_features.device
        x = torch.randn(B, self.action_dim, device=device)
        for k in reversed(range(self.num_diffusion_steps)):
            t = torch.full((B,), k, device=device, dtype=torch.long)
            if deterministic:
                mean, _, _ = self.p_mean_variance(x, obs_features, t)
                x = mean
            else:
                x = self.p_sample(x, obs_features, t)
        return x

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def compute_diffusion_loss(self, actions, obs_features):
        """L_diff = E_{k,ε}[||ε − ε_θ(a^k, k, o)||²]"""
        B = actions.shape[0]
        device = actions.device
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)
        noise = torch.randn_like(actions)
        noisy = self.q_sample(actions, t, noise=noise)
        pred = self.predict_noise(noisy, obs_features, t)
        return F.mse_loss(pred, noise)

    def compute_denoising_log_prob(self, a_prev, a_curr, obs_features, t):
        """log p_θ(a^{k-1} | a^k, o)  — Gaussian with predicted mean, fixed var."""
        pred_noise = self.predict_noise(a_curr, obs_features, t)
        sqrt_recip = extract(self.sqrt_recip_alphas, t, a_curr.shape)
        beta_t = extract(self.betas, t, a_curr.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, a_curr.shape)
        mean = sqrt_recip * (a_curr - beta_t / sqrt_one_minus * pred_noise)
        var = extract(self.posterior_variance, t, a_curr.shape).clamp(min=1e-6)
        log_prob = -0.5 * ((a_prev - mean) ** 2 / var + torch.log(var)
                           + math.log(2 * math.pi))
        return log_prob.sum(dim=-1)

    def compute_chain_log_prob(self, denoising_chain, obs_features):
        """Σ_{k=1}^{K} log p_θ(a^{k-1}|a^k, s)"""
        B = obs_features.shape[0]
        device = obs_features.device
        total = torch.zeros(B, device=device)
        K = len(denoising_chain) - 1
        for step_idx in range(K):
            k = K - step_idx
            t = torch.full((B,), k, device=device, dtype=torch.long)
            total += self.compute_denoising_log_prob(
                denoising_chain[step_idx + 1],
                denoising_chain[step_idx],
                obs_features, t)
        return total

    # ------------------------------------------------------------------
    # Sample with chain (for PPO data collection)
    # ------------------------------------------------------------------

    def sample_action_with_chain(self, obs_features, deterministic=False):
        B = obs_features.shape[0]
        device = obs_features.device
        x = torch.randn(B, self.action_dim, device=device)
        chain = [x.clone()]
        for k in reversed(range(self.num_diffusion_steps)):
            t = torch.full((B,), k, device=device, dtype=torch.long)
            if deterministic:
                mean, _, _ = self.p_mean_variance(x, obs_features, t)
                x = mean
            else:
                x = self.p_sample(x, obs_features, t)
            chain.append(x.clone())
        log_prob = self.compute_chain_log_prob(chain, obs_features)
        return chain[-1], chain, log_prob

    # ------------------------------------------------------------------
    # Convenience forward (matches RLMamba2Racer signature pattern)
    # ------------------------------------------------------------------

    def forward(self, scan_tensor, state_tensor, obs_buffer,
                max_speed_estimate=20, deterministic=False):
        """
        Full forward pass: encode observation → denoise → action.

        Returns:
            action:     (B, action_dim)
            log_prob:   (B,)
            obs_buffer: updated observation buffer
        """
        obs_features, obs_buffer = self.encode_observation(
            scan_tensor, state_tensor, obs_buffer, max_speed_estimate
        )
        action, chain, log_prob = self.sample_action_with_chain(
            obs_features, deterministic=deterministic
        )
        return action, log_prob, obs_buffer
