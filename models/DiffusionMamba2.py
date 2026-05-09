"""
DiffusionMamba2 — Diffusion Policy Actor with Mamba2 temporal head.
=========================================================================
Mirrors the architecture of RLMamba2Racer but replaces the Gaussian action
distribution with DDPM-based iterative denoising.

Observation pipeline:
    VisionEncoder(CNN) → odom_expand → feature_projection → Mamba2.step()
    → obs_features (temporal context via recurrent SSM state)

Action generation:
    obs_features → ConditionalDenoisingMLP → K-step DDPM reverse process → action
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AuxModels import VisionEncoder, ResidualBlock
from utils.diffusion_utils import *
from mamba_ssm.modules.mamba2 import Mamba2


# ===========================================================================
# DiffusionMamba2 — full actor
# ===========================================================================

class DiffusionMamba2(nn.Module):
    """
    Diffusion Policy actor with Mamba2 recurrent temporal backbone.

    Observation encoder:
        VisionEncoder → odom_expand → feature_projection
        → Mamba2.step() (recurrent SSM state) → norm_layer
    The final observation embedding is fed into a DDPM denoising MLP that
    iteratively refines pure noise into an action.

    Temporal memory is carried through Mamba2's compact SSM state
    ``(conv_state, ssm_state)`` rather than a sliding feature buffer,
    giving O(1) per-step cost with infinite temporal horizon.
    """
    def __init__(
        self,
        state_dim=4,
        action_dim=2,
        encoder=None,
        num_diffusion_steps=50,
        inference_steps=0,          # if >0, use DDIM sampling with this many steps
        obs_feature_dim=256,           # matches d_model for Mamba2
        time_emb_dim=32,
        hidden_dims=(768, 768, 768),
        beta_schedule="cosine",
        # Mamba2 config
        d_model=256,
        d_state=16,
        d_conv=4,
        d_head=16,
        expand=2,
        odom_expand=64,
        # Action normalisation range  (raw env units → [-1, 1])
        action_low=(-0.34, 0.0),      # (min_steering, min_speed)
        action_high=(0.34, 20.0),     # (max_steering, max_speed)
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.inference_steps = inference_steps  # 0 → fall back to num_diffusion_steps
        self.obs_feature_dim = obs_feature_dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        # Internal recurrent state managed during inference
        self._conv_state = None
        self._ssm_state = None

        # --- Action normalisation  (raw ↔ [-1, 1]) ---
        act_lo = torch.tensor(action_low,  dtype=torch.float32)
        act_hi = torch.tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_lo", act_lo)
        self.register_buffer("action_hi", act_hi)

        # --- State normalisation ranges (must match critic) ---
        default_ranges = [
            [-5.0, 20.0],   # linear_vel_x
            [-5.0,  5.0],   # linear_vel_y
            [-15.0, 15.0],  # ang_vel_z
        ]
        state_lo = torch.tensor([r[0] for r in default_ranges[:state_dim]], dtype=torch.float32)
        state_hi = torch.tensor([r[1] for r in default_ranges[:state_dim]], dtype=torch.float32)
        self.register_buffer("state_lo", state_lo)
        self.register_buffer("state_hi", state_hi)

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
            nn.Linear(feature_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_model),
            nn.ReLU(),
        )

        # --- Mamba2 temporal backbone (recurrent SSM) ---
        self.pre_mamba_norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
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
    # Recurrent state management  (Mamba2 SSM state)
    # ------------------------------------------------------------------

    def allocate_state(self, batch_size: int, device: torch.device):
        """Allocate zero-initialised Mamba2 recurrent state.

        Returns:
            (conv_state, ssm_state) tuple used by ``Mamba2.step()``.
        """
        conv, ssm = self.mamba.allocate_inference_cache(batch_size, max_seqlen=1)
        return conv.contiguous(), ssm.contiguous()

    def reset_temporal_state(self, agent_idxs=None):
        """Reset internal recurrent state.

        Args:
            agent_idxs: optional array of agent indices to reset.
                        If None, resets all agents.
        """
        if self._conv_state is None:
            return
        if agent_idxs is None:
            self._conv_state.zero_()
            self._ssm_state.zero_()
        else:
            agent_idxs = torch.as_tensor(agent_idxs, device=self._conv_state.device, dtype=torch.long).reshape(-1)
            agent_idxs = agent_idxs[(agent_idxs >= 0) & (agent_idxs < self._conv_state.shape[0])]
            if agent_idxs.numel() == 0:
                return
            self._conv_state[agent_idxs] = 0.0
            self._ssm_state[agent_idxs] = 0.0

    # ------------------------------------------------------------------
    # Observation encoding  (VisionEncoder → Mamba2.step)
    # ------------------------------------------------------------------

    def encode_observation(self, scan_tensor, state_tensor,
                           conv_state=None, ssm_state=None,
                           max_speed_estimate=20.0):
        """
        Encode raw observations through CNN + Mamba2 recurrent step.

        When *conv_state* / *ssm_state* are **not** supplied the network
        manages internal recurrent state and returns only ``obs_features``.
        When explicit states are passed (e.g. during TBTT training) the
        updated states are returned so the caller can detach them.

        Args:
            scan_tensor:  (B, 1, num_beams)
            state_tensor: (B, state_dim)
            conv_state:   Mamba2 conv state or None
            ssm_state:    Mamba2 SSM state or None

        Returns:
            obs_features                          when using internal state
            (obs_features, conv_state, ssm_state) when external state supplied
        """
        batch_size = scan_tensor.shape[0]
        device = scan_tensor.device
        _using_internal = conv_state is None

        if _using_internal:
            if self._conv_state is None or self._conv_state.shape[0] != batch_size:
                self._conv_state, self._ssm_state = self.allocate_state(batch_size, device)
            conv_state = self._conv_state
            ssm_state = self._ssm_state
        if conv_state.device != device:
            conv_state = conv_state.to(device).contiguous()
            ssm_state = ssm_state.to(device).contiguous()
        scan_tensor = torch.nan_to_num(scan_tensor, posinf=20.0, neginf=0.0)
        state_tensor = torch.nan_to_num(state_tensor, 0.0)

        # Normalize LiDAR scans to [0, 1] (match critic preprocessing)
        scan_tensor = scan_tensor.clamp(0.0, 10.0) / 10.0

        # CNN → odom expand → concat → project
        vision_features = self.vision_encoder(scan_tensor)
        vision_features = torch.nan_to_num(vision_features, nan=0.0)
        state_norm = 2.0 * (state_tensor - self.state_lo) / (self.state_hi - self.state_lo + 1e-8) - 1.0
        state_norm = state_norm.clamp(-1.0, 1.0)
        state_feat = self.odom_expand(state_norm)
        combined = torch.cat([vision_features, state_feat], dim=-1)
        projected = self.feature_projection(combined)
        projected = torch.nan_to_num(projected, nan=0.0)

        # Pre-norm and clamp before Mamba2 step
        projected_normed = self.pre_mamba_norm(projected)
        projected_normed = torch.clamp(projected_normed, -10.0, 10.0)
        mamba_dtype = next(self.mamba.parameters()).dtype
        projected_normed = projected_normed.to(dtype=mamba_dtype)
        conv_state = conv_state.to(dtype=mamba_dtype)
        ssm_state = ssm_state.to(dtype=mamba_dtype)

        # Mamba2 recurrent step: O(1) per timestep, infinite horizon
        # step() expects (B, 1, d_model) input
        with torch.amp.autocast(device_type=device.type, enabled=False):
            mamba_out, conv_state, ssm_state = self.mamba.step(
                projected_normed.unsqueeze(1), conv_state, ssm_state
            )
        mamba_out = torch.nan_to_num(mamba_out, nan=0.0)
        obs_features = self.norm_layer(mamba_out.squeeze(1))
        obs_features = torch.nan_to_num(obs_features, nan=0.0)

        if _using_internal:
            self._conv_state = conv_state
            self._ssm_state = ssm_state
            return obs_features
        return obs_features, conv_state, ssm_state

    # ------------------------------------------------------------------
    # Forward / reverse diffusion helpers
    # ------------------------------------------------------------------

    def normalize_action(self, action):
        """Map raw action from [action_lo, action_hi] → [-1, 1]."""
        return 2.0 * (action - self.action_lo) / (self.action_hi - self.action_lo + 1e-8) - 1.0

    def denormalize_action(self, action_norm):
        """Map normalised action from [-1, 1] → [action_lo, action_hi], clamped."""
        raw = (action_norm + 1.0) * 0.5 * (self.action_hi - self.action_lo) + self.action_lo
        return raw.clamp(self.action_lo.unsqueeze(0), self.action_hi.unsqueeze(0))

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

        # Predict x_0 from the noise prediction and clamp to [-1, 1]
        # This prevents cascading amplification of prediction errors through
        # the reverse chain (critical with aggressive cosine schedule / few steps).
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x_0_pred = (x_t - sqrt_one_minus_t * pred_noise) / sqrt_alphas_cumprod_t.clamp(min=1e-8)
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        # Re-derive posterior mean from the clamped x_0 prediction
        coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x_0_pred + coef2 * x_t

        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(self, x_t, obs_features, t):
        mean, var, _ = self.p_mean_variance(x_t, obs_features, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.sqrt(var) * noise

    def _ddim_timestep_schedule(self, num_steps=None):
        """Return evenly-spaced timestep indices for DDIM sub-sampling."""
        K = self.num_diffusion_steps
        S = num_steps or self.inference_steps or K
        S = min(S, K)
        if S == K:
            return list(range(K - 1, -1, -1))
        # Span the full reverse process. For S=1 this intentionally returns
        # [K - 1], so one-step DDIM predicts x0 from the noisiest state.
        return torch.linspace(K - 1, 0, S).round().long().tolist()

    @torch.no_grad()
    def ddim_sample_action(self, obs_features, num_steps=None, eta=0.0):
        """
        Fast DDIM sampling.  Uses the *same* trained noise schedule but
        skips intermediate timesteps for much fewer MLP forwards.

        Args:
            obs_features:  (B, obs_feature_dim)
            num_steps:     Override for number of DDIM steps (default: self.inference_steps)
            eta:           Stochasticity (0 = deterministic DDIM, 1 ≈ DDPM)
        """
        B = obs_features.shape[0]
        device = obs_features.device
        schedule = self._ddim_timestep_schedule(num_steps)

        x = torch.randn(B, self.action_dim, device=device)
        for i, t_curr in enumerate(schedule):
            t = torch.full((B,), t_curr, device=device, dtype=torch.long)
            pred_noise = self.predict_noise(x, obs_features, t)

            alpha_t = extract(self.alphas_cumprod, t, x.shape)
            x_0_pred = ((x - torch.sqrt(1.0 - alpha_t) * pred_noise)
                        / torch.sqrt(alpha_t).clamp(min=1e-8))
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)

            if i + 1 < len(schedule):
                t_next = schedule[i + 1]
                alpha_next = self.alphas_cumprod[t_next]
            else:
                alpha_next = torch.tensor(1.0, device=device)  # α_0 = 1

            # DDIM update
            if eta > 0 and t_curr > 0:
                sigma = (eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t).clamp(min=1e-8)
                         * (1 - alpha_t / alpha_next.clamp(min=1e-8))))
                dir_xt = torch.sqrt((1 - alpha_next - sigma ** 2).clamp(min=0)) * pred_noise
                x = torch.sqrt(alpha_next) * x_0_pred + dir_xt + sigma * torch.randn_like(x)
            else:
                dir_xt = torch.sqrt((1 - alpha_next).clamp(min=0)) * pred_noise
                x = torch.sqrt(alpha_next) * x_0_pred + dir_xt

        return self.denormalize_action(x.clamp(-1.0, 1.0))

    @torch.no_grad()
    def ddim_sample_action_with_chain(self, obs_features, num_steps=None, eta=0.5):
        """
        DDIM sampling that stores the denoising chain and computes inline
        log-probs — used for D2PPO RL rollouts.

        With eta > 0 each DDIM step is a Gaussian transition:
            x_prev = mean_ddim(x_curr, obs) + sigma * z,   z ~ N(0, I)
        so log p(x_prev | x_curr, obs) is analytically available.

        Returns:
            action:       (B, action_dim) de-normalised
            chain:        list of S+1 tensors [(B, action_dim)] in normalised space
                          chain[0] = pure noise (x_T), chain[S] = final action (x_0)
            log_prob:     (B,) sum of per-step log-probs
            ddim_schedule: list of S timestep indices (descending), needed by learning
        """
        B = obs_features.shape[0]
        device = obs_features.device
        schedule = self._ddim_timestep_schedule(num_steps)
        S = len(schedule)

        x = torch.randn(B, self.action_dim, device=device)
        chain = [x.clone()]
        log_prob_accum = torch.zeros(B, device=device)

        for i, t_curr in enumerate(schedule):
            t = torch.full((B,), t_curr, device=device, dtype=torch.long)
            pred_noise = self.predict_noise(x, obs_features, t)

            alpha_t = extract(self.alphas_cumprod, t, x.shape)
            x_0_pred = ((x - torch.sqrt(1.0 - alpha_t) * pred_noise)
                        / torch.sqrt(alpha_t).clamp(min=1e-8))
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)

            if i + 1 < S:
                t_next = schedule[i + 1]
                alpha_next = self.alphas_cumprod[t_next]
            else:
                alpha_next = torch.tensor(1.0, device=device)

            if eta > 0 and t_curr > 0:
                sigma = (eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha_t).clamp(min=1e-8)
                    * (1 - alpha_t / alpha_next.clamp(min=1e-8))
                ))
                dir_xt = torch.sqrt((1 - alpha_next - sigma ** 2).clamp(min=0)) * pred_noise
                mean_ddim = torch.sqrt(alpha_next) * x_0_pred + dir_xt
                noise_z = torch.randn_like(x)
                x_prev = mean_ddim + sigma * noise_z

                # Gaussian log-prob of this transition
                var = (sigma ** 2).clamp(min=1e-6)
                step_lp = -0.5 * (
                    (x_prev - mean_ddim) ** 2 / var
                    + torch.log(var)
                    + math.log(2 * math.pi)
                )
                log_prob_accum = log_prob_accum + step_lp.sum(dim=-1)
            else:
                # Deterministic step — no stochasticity, infinite log-prob
                # (delta distribution).  For PPO we skip these steps.
                dir_xt = torch.sqrt((1 - alpha_next).clamp(min=0)) * pred_noise
                x_prev = torch.sqrt(alpha_next) * x_0_pred + dir_xt

            x = x_prev
            chain.append(x.clone())

        return (self.denormalize_action(chain[-1].clamp(-1.0, 1.0)),
                chain, log_prob_accum, schedule)

    def compute_ddim_log_prob(self, x_prev, x_curr, obs_features, t_curr_idx, t_next_idx, eta=0.5):
        """
        Compute log p_theta(x_prev | x_curr, obs) for a single DDIM step
        transitioning from timestep t_curr to t_next (t_next < t_curr).

        Used during PPO learning to recompute log-probs under the current policy.

        Args:
            x_prev:      (B, action_dim)  — the denoised sample at t_next
            x_curr:      (B, action_dim)  — the noisier sample at t_curr
            obs_features:(B, obs_feature_dim)
            t_curr_idx:  int or (B,) tensor — current DDIM timestep index
            t_next_idx:  int or None — next DDIM timestep index (None means final step → alpha=1)
            eta:         float — DDIM stochasticity parameter
        """
        B = x_curr.shape[0]
        device = x_curr.device
        if isinstance(t_curr_idx, int):
            t = torch.full((B,), t_curr_idx, device=device, dtype=torch.long)
        else:
            t = t_curr_idx

        pred_noise = self.predict_noise(x_curr, obs_features, t)

        alpha_t = extract(self.alphas_cumprod, t, x_curr.shape)
        x_0_pred = ((x_curr - torch.sqrt(1.0 - alpha_t) * pred_noise)
                    / torch.sqrt(alpha_t).clamp(min=1e-8))
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        if t_next_idx is not None:
            alpha_next = self.alphas_cumprod[t_next_idx]
        else:
            alpha_next = torch.tensor(1.0, device=device)

        sigma = (eta * torch.sqrt(
            (1 - alpha_next) / (1 - alpha_t).clamp(min=1e-8)
            * (1 - alpha_t / alpha_next.clamp(min=1e-8))
        ))
        dir_xt = torch.sqrt((1 - alpha_next - sigma ** 2).clamp(min=0)) * pred_noise
        mean_ddim = torch.sqrt(alpha_next) * x_0_pred + dir_xt

        var = (sigma ** 2).clamp(min=1e-6)
        log_prob = -0.5 * (
            (x_prev - mean_ddim) ** 2 / var
            + torch.log(var)
            + math.log(2 * math.pi)
        )
        return log_prob.sum(dim=-1)

    def compute_ddim_log_prob_with(self, x_prev, x_curr, obs_features,
                                    t_curr_idx, t_next_idx, denoise_net, eta=0.5):
        """Like compute_ddim_log_prob but uses a supplied denoise_net
        (frozen old-policy snapshot for PPO importance ratios)."""
        B = x_curr.shape[0]
        device = x_curr.device
        if isinstance(t_curr_idx, int):
            t = torch.full((B,), t_curr_idx, device=device, dtype=torch.long)
        else:
            t = t_curr_idx

        pred_noise = denoise_net(x_curr, obs_features, t)

        alpha_t = extract(self.alphas_cumprod, t, x_curr.shape)
        x_0_pred = ((x_curr - torch.sqrt(1.0 - alpha_t) * pred_noise)
                    / torch.sqrt(alpha_t).clamp(min=1e-8))
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        if t_next_idx is not None:
            alpha_next = self.alphas_cumprod[t_next_idx]
        else:
            alpha_next = torch.tensor(1.0, device=device)

        sigma = (eta * torch.sqrt(
            (1 - alpha_next) / (1 - alpha_t).clamp(min=1e-8)
            * (1 - alpha_t / alpha_next.clamp(min=1e-8))
        ))
        dir_xt = torch.sqrt((1 - alpha_next - sigma ** 2).clamp(min=0)) * pred_noise
        mean_ddim = torch.sqrt(alpha_next) * x_0_pred + dir_xt

        var = (sigma ** 2).clamp(min=1e-6)
        log_prob = -0.5 * (
            (x_prev - mean_ddim) ** 2 / var
            + torch.log(var)
            + math.log(2 * math.pi)
        )
        return log_prob.sum(dim=-1)

    @torch.no_grad()
    def sample_action(self, obs_features, deterministic=False):
        """Sample action via stochastic DDPM reverse process.

        The per-step noise is integral to reverse diffusion — it is NOT
        RL exploration.  Mean-only sampling collapses to near-zero output.
        When ``deterministic`` is True we still use the stochastic reverse
        chain (which gives high-quality diverse actions) rather than the
        broken mean-only path.
        """
        B = obs_features.shape[0]
        device = obs_features.device
        x = torch.randn(B, self.action_dim, device=device)
        for k in reversed(range(self.num_diffusion_steps)):
            t = torch.full((B,), k, device=device, dtype=torch.long)
            x = self.p_sample(x, obs_features, t)
        return self.denormalize_action(x.clamp(-1.0, 1.0))   # [-1,1] → raw action space


    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def compute_diffusion_loss(self, actions, obs_features):
        """L_diff = E_{k,ε[||ε − ε_θ(a^k, k, o)||²]"""
        actions = self.normalize_action(actions)
        B = actions.shape[0]
        device = actions.device
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)
        noise = torch.randn_like(actions)
        noisy = self.q_sample(actions, t, noise=noise)
        pred = self.predict_noise(noisy, obs_features, t)
        return F.mse_loss(pred, noise)

    def compute_denoising_log_prob(self, a_prev, a_curr, obs_features, t):
        pred_noise = self.predict_noise(a_curr, obs_features, t)

        # Predict x_0 from noise prediction and clamp (matches p_mean_variance)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, a_curr.shape)
        sqrt_one_minus_t = extract(self.sqrt_one_minus_alphas_cumprod, t, a_curr.shape)
        x_0_pred = (a_curr - sqrt_one_minus_t * pred_noise) / sqrt_alphas_cumprod_t.clamp(min=1e-8)
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        # Re-derive posterior mean from clamped x_0
        coef1 = extract(self.posterior_mean_coef1, t, a_curr.shape)
        coef2 = extract(self.posterior_mean_coef2, t, a_curr.shape)
        mean = coef1 * x_0_pred + coef2 * a_curr

        var = extract(self.posterior_variance, t, a_curr.shape).clamp(min=1e-6)
        log_prob = -0.5 * ((a_prev - mean) ** 2 / var + torch.log(var)
                           + math.log(2 * math.pi))
        return log_prob.sum(dim=-1)

    def compute_denoising_log_prob_with(self, a_prev, a_curr, obs_features, t,
                                         denoise_net):
        """Like compute_denoising_log_prob but uses a supplied denoise_net
        (e.g. the frozen snapshot for old-policy log-probs in PPO)."""
        pred_noise = denoise_net(a_curr, obs_features, t)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, a_curr.shape)
        sqrt_one_minus_t = extract(self.sqrt_one_minus_alphas_cumprod, t, a_curr.shape)
        x_0_pred = (a_curr - sqrt_one_minus_t * pred_noise) / sqrt_alphas_cumprod_t.clamp(min=1e-8)
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        coef1 = extract(self.posterior_mean_coef1, t, a_curr.shape)
        coef2 = extract(self.posterior_mean_coef2, t, a_curr.shape)
        mean = coef1 * x_0_pred + coef2 * a_curr

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
        """
        Sample action via full DDPM reverse process, computing chain log-prob
        *inline* to avoid a redundant second pass through the denoising MLP.
        """
        B = obs_features.shape[0]
        device = obs_features.device
        x = torch.randn(B, self.action_dim, device=device)
        chain = [x.clone()]
        log_prob_accum = torch.zeros(B, device=device)

        for k in reversed(range(self.num_diffusion_steps)):
            t = torch.full((B,), k, device=device, dtype=torch.long)
            # p_mean_variance already calls predict_noise once —
            # reuse mean & var for both the sample and the log-prob.
            mean, var, _ = self.p_mean_variance(x, obs_features, t)

            if deterministic or k == 0:
                x_prev = mean
            else:
                noise = torch.randn_like(x)
                x_prev = mean + torch.sqrt(var) * noise

            # Accumulate log p(x_{k-1} | x_k, obs) inline
            if k > 0:
                var_c = var.clamp(min=1e-6)
                step_lp = -0.5 * (
                    (x_prev - mean) ** 2 / var_c
                    + torch.log(var_c)
                    + math.log(2 * math.pi)
                )
                log_prob_accum = log_prob_accum + step_lp.sum(dim=-1)

            x = x_prev
            chain.append(x.clone())

        # chain lives in normalised [-1,1] space; clamp and return raw action
        return self.denormalize_action(chain[-1].clamp(-1.0, 1.0)), chain, log_prob_accum


    # ------------------------------------------------------------------
    # Convenience forward (matches RLMamba2Racer signature pattern)
    # ------------------------------------------------------------------

    def forward(self, scan_tensor, state_tensor,
                conv_state=None, ssm_state=None,
                max_speed_estimate=20, deterministic=False):
        """
        Full forward pass: encode observation → denoise → action.

        Returns:
            action:      (B, action_dim)
            log_prob:    (B,)
            conv_state:  updated Mamba2 conv state
            ssm_state:   updated Mamba2 SSM state
        """
        obs_features, conv_state, ssm_state = self.encode_observation(
            scan_tensor, state_tensor, conv_state, ssm_state, max_speed_estimate
        )
        if deterministic:
            action = self.sample_action(obs_features, deterministic=True)
            log_prob = torch.zeros(action.shape[0], device=action.device)
        else:
            action, _, log_prob = self.sample_action_with_chain(
                obs_features, deterministic=False
            )
        return action, log_prob, conv_state, ssm_state
