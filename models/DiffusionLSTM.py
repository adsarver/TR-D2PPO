"""
DiffusionLSTM — Diffusion Policy Actor with LSTM temporal head.
================================================================
Mirrors the architecture of HybridLSTM / ExampleNetwork but replaces the
Gaussian action distribution with DDPM-based iterative denoising.

Observation pipeline:
    VisionEncoder(CNN) → state normalisation → feature_projection → LSTM
    → obs_features (temporal context)

Action generation:
    obs_features → ConditionalDenoisingMLP → K-step DDPM reverse process → action
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AuxModels import *
from utils.diffusion_utils import *

class DiffusionLSTM(nn.Module):
    """
    Diffusion Policy actor with LSTM temporal backbone.

    Observation encoder follows the same pattern as ``ExampleNetwork`` /
    ``HybridLSTM``:
        VisionEncoder → state normalise → feature_projection → LSTM
        → obs_features (temporal context)

    ``create_observation_buffer`` + ``get_init_hidden`` match the LSTM model
    API so the training loop can treat both models identically.

    Parameters
    ----------
    memory_length : int
        Number of historical feature vectors kept in the rolling buffer.
    memory_stride : int
        Only append a new frame to the buffer every *memory_stride* steps;
        between strides the latest slot is overwritten so the LSTM always
        sees fresh perception while history stays stable.
    """
    def __init__(
        self,
        state_dim=4,
        action_dim=2,
        encoder=None,
        num_diffusion_steps=50,
        inference_steps=0,          # DDIM steps for fast inference (0 = use full DDPM)
        time_emb_dim=32,
        hidden_dims=(768, 768, 768),
        beta_schedule="cosine",
        # LSTM config (mirrors ExampleNetwork defaults)
        lstm_hidden_size=256,
        lstm_num_layers=2,
        memory_length=5,
        memory_stride=5,
        odom_expand=64,
        proj_hidden=768,            # Feature projection intermediate dim
        # Action normalisation range  (raw env units → [-1, 1])
        action_low=(-0.34, 0.0),     # (min_steering, min_speed)
        action_high=(0.34, 10.0),     # (max_steering, max_speed)
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.inference_steps = inference_steps  # 0 → fall back to num_diffusion_steps
        self.obs_feature_dim = lstm_hidden_size   # LSTM output == obs embedding dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.memory_length = memory_length
        self.memory_stride = memory_stride
        self.step_counter = 0

        # --- Vision encoder (1-D CNN for LiDAR) ---
        if encoder is None:
            encoder = VisionEncoder(num_scan_beams=1080)
        self.vision_encoder = encoder
        conv_output_size = self.vision_encoder.output_size

        # --- State normalisation (baked-in ranges like ExampleNetwork) ---
        default_ranges = [
            [-5.0, 20.0],   # linear_vel_x
            [-5.0,  5.0],   # linear_vel_y
            [-15.0, 15.0],  # ang_vel_z
            [-10.0, 10.0],  # linear_accel_x
        ]
        state_lo = torch.tensor([r[0] for r in default_ranges[:state_dim]], dtype=torch.float32)
        state_hi = torch.tensor([r[1] for r in default_ranges[:state_dim]], dtype=torch.float32)
        self.register_buffer("state_lo", state_lo)
        self.register_buffer("state_hi", state_hi)

        # --- Action normalisation  (raw ↔ [-1, 1]) ---
        # DDPM assumes x_0 lives in a standard-ish range.  We map
        # [action_low, action_high] ↔ [-1, 1] before any diffusion op
        # and invert after sampling.
        act_lo = torch.tensor(action_low,  dtype=torch.float32)
        act_hi = torch.tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_lo", act_lo)
        self.register_buffer("action_hi", act_hi)

        # --- State expansion ---
        self.odom_expand_layer = nn.Linear(state_dim, odom_expand)

        # --- Feature projection (CNN+odom → lstm_hidden_size) ---
        feature_input_size = conv_output_size + odom_expand
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_input_size, proj_hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden, lstm_hidden_size),
            nn.LeakyReLU(),
        )

        # --- LSTM temporal backbone ---
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1 if lstm_num_layers > 1 else 0.0,
        )

        self.post_lstm_norm = nn.LayerNorm(lstm_hidden_size)

        # --- Conditional denoising MLP (ε_θ) ---
        self.denoise_net = ConditionalDenoisingMLP(
            action_dim=action_dim,
            obs_feature_dim=lstm_hidden_size,
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
    # Observation / hidden-state buffers (same API as ExampleNetwork)
    # ------------------------------------------------------------------

    def create_observation_buffer(self, batch_size, device):
        self.step_counter = 0
        return torch.zeros(batch_size, self.memory_length,
                           self.lstm_hidden_size, device=device)

    def get_init_hidden(self, batch_size, device, transpose=False):
        h0 = torch.zeros(self.lstm_num_layers, batch_size,
                          self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size,
                          self.lstm_hidden_size, device=device)
        if transpose:
            h0 = h0.transpose(0, 1).contiguous()
            c0 = c0.transpose(0, 1).contiguous()
        return (h0, c0)

    # ------------------------------------------------------------------
    # Observation encoding  (VisionEncoder → LSTM)
    # ------------------------------------------------------------------

    def encode_observation(self, scan_tensor, state_tensor, obs_buffer,
                           hidden_h=None, hidden_c=None):
        """
        Encode raw observations through CNN + LSTM temporal backbone.

        Args:
            scan_tensor:  (B, 1, num_beams)
            state_tensor: (B, state_dim)
            obs_buffer:   (B, memory_length, lstm_hidden_size)
            hidden_h:     (B, num_layers, lstm_hidden_size) or None
            hidden_c:     (B, num_layers, lstm_hidden_size) or None

        Returns:
            obs_features: (B, lstm_hidden_size)
            obs_buffer:   updated buffer
            hidden_h:     (B, num_layers, lstm_hidden_size)
            hidden_c:     (B, num_layers, lstm_hidden_size)
        """
        batch_size = scan_tensor.shape[0]
        device = scan_tensor.device
        
        if obs_buffer is None:
            obs_buffer = self.create_observation_buffer(batch_size, device)
        if hidden_h is None or hidden_c is None:
            hidden = self.get_init_hidden(batch_size, device)
        else:
            hidden = (hidden_h.transpose(0, 1).contiguous(),
                      hidden_c.transpose(0, 1).contiguous())

        # Scan normalisation
        scan_norm = scan_tensor.clamp(0.0, 10.0) / 10.0

        # CNN features
        vision_features = self.vision_encoder(scan_norm)

        # State normalisation (ExampleNetwork-style baked ranges)
        state_norm = 2.0 * (state_tensor - self.state_lo) / (self.state_hi - self.state_lo + 1e-8) - 1.0
        state_norm = state_norm.clamp(-1.0, 1.0)
        state_feat = self.odom_expand_layer(state_norm)

        combined = torch.cat([vision_features, state_feat], dim=-1)
        current_feature = self.feature_projection(combined)

        # Rolling buffer update (stride-aware, like ExampleNetwork)
        self.step_counter += 1
        if self.step_counter % self.memory_stride == 0:
            obs_buffer = torch.cat([
                obs_buffer[:, 1:, :],
                current_feature.unsqueeze(1),
            ], dim=1)
        else:
            obs_buffer = obs_buffer.clone()
            obs_buffer[:, -1, :] = current_feature

        # LSTM temporal pass
        lstm_out, hidden_new = self.lstm(obs_buffer, hidden)
        obs_features = self.post_lstm_norm(lstm_out[:, -1, :])

        # Transpose hidden back to (B, num_layers, H)
        hidden_h_out = hidden_new[0].transpose(0, 1).contiguous()
        hidden_c_out = hidden_new[1].transpose(0, 1).contiguous()

        return obs_features, obs_buffer, hidden_h_out, hidden_c_out

    def normalize_action(self, action):
        """Map raw action from [action_lo, action_hi] → [-1, 1]."""
        return 2.0 * (action - self.action_lo) / (self.action_hi - self.action_lo + 1e-8) - 1.0

    def denormalize_action(self, action_norm):
        """Map normalised action from [-1, 1] → [action_lo, action_hi]."""
        return (action_norm + 1.0) * 0.5 * (self.action_hi - self.action_lo) + self.action_lo

    # ------------------------------------------------------------------
    # Forward / reverse diffusion helpers
    # ------------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_noise(self, noisy_action, obs_features, t):
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
        # Evenly spaced, always including step 0
        step_size = K / S
        return [int(round((S - 1 - i) * step_size)) for i in range(S)]

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
    def sample_action(self, obs_features, deterministic=False):
        """Sample action using DDIM (fast) if inference_steps > 0, else full DDPM."""
        if self.inference_steps > 0:
            return self.ddim_sample_action(
                obs_features, eta=0.0 if deterministic else 0.3
            )
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
        return self.denormalize_action(x.clamp(-1.0, 1.0))   # [-1,1] → raw action space

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def compute_diffusion_loss(self, actions, obs_features):
        B = actions.shape[0]
        device = actions.device
        actions_norm = self.normalize_action(actions)   # raw → [-1,1]
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)
        noise = torch.randn_like(actions_norm)
        noisy = self.q_sample(actions_norm, t, noise=noise)
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

    def compute_chain_log_prob(self, denoising_chain, obs_features):
        B = obs_features.shape[0]
        device = obs_features.device
        total = torch.zeros(B, device=device)
        K = len(denoising_chain) - 1
        
        for step_idx in range(K):
            k = K - 1 - step_idx
            t = torch.full((B,), k, device=device, dtype=torch.long)
            total += self.compute_denoising_log_prob(
                denoising_chain[step_idx + 1],
                denoising_chain[step_idx],
                obs_features, t)
        return total


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
    # Convenience forward (matches ExampleNetwork / HybridLSTM signature)
    # ------------------------------------------------------------------

    def forward(self, scan_tensor, state_tensor, obs_buffer,
                hidden_h=None, hidden_c=None, deterministic=False):
        """
        Full forward pass: encode observation → LSTM → denoise → action.

        Returns:
            action:     (B, action_dim)
            log_prob:   (B,)
            obs_buffer: updated observation buffer
            hidden_h:   (B, num_layers, lstm_hidden_size)
            hidden_c:   (B, num_layers, lstm_hidden_size)
        """
        obs_features, obs_buffer, hidden_h, hidden_c = self.encode_observation(
            scan_tensor, state_tensor, obs_buffer, hidden_h, hidden_c
        )
        action, chain, log_prob = self.sample_action_with_chain(
            obs_features, deterministic=deterministic
        )
        return action, log_prob, obs_buffer, hidden_h, hidden_c
