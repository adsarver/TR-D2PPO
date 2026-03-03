import torch 
import torch.nn as nn
import math
from models.AuxModels import ResidualBlock


# ===========================================================================
# Noise Schedules
# ===========================================================================

def cosine_beta_schedule(num_timesteps, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal, 2021)."""
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule (original DDPM)."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def extract(a, t, x_shape):
    """Extract coefficients at timesteps *t*, reshaped for broadcasting."""
    # Ensure t is a 1-D long tensor of shape (batch,)
    if t.dim() != 1:
        t = t.reshape(-1)
    batch_size = t.shape[0]

    if a.dim() == 1:
        out = a[t]
    else:
        idx = t.view(batch_size, *([1] * (a.dim() - 1))).expand(batch_size, *a.shape[1:])
        out = a.gather(0, idx)

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep *t*."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalDenoisingMLP(nn.Module):
    """
    Noise-prediction network ``ε_θ(a^k, o, k)`` conditioned on observation
    features and diffusion timestep.

    Also exposes *dispersive-loss hooks* — forward hooks that collect
    intermediate features used to compute InfoNCE / Hinge dispersion
    penalties during pre-training (D²PPO Stage 1).

    Architecture
    ------------
    [noisy_action ⊕ obs_features ⊕ time_emb] → MLP of ``ResidualBlock``
    layers → predicted noise ``ε``.
    """

    def __init__(self, action_dim=2, obs_feature_dim=256,
                 time_emb_dim=32, hidden_dims=(768, 768, 768)):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        input_dim = action_dim + obs_feature_dim + time_emb_dim
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(ResidualBlock(in_d, h))
            in_d = h
        self.mlp = nn.Sequential(*layers)
        self.out_proj = nn.Linear(in_d, action_dim)

        # --- Dispersive hook storage ---
        self._dispersive_hooks = []
        self._intermediate_features = []

    # -- Dispersive hooks -------------------------------------------------

    def register_dispersive_hooks(self, layer_indices=None):
        """Attach forward hooks to selected ``ResidualBlock`` layers to capture
        intermediate activations for the dispersive loss.

        Args:
            layer_indices: Which residual blocks to hook.
                ``None``  → hook **all** blocks.
                ``"early"`` / ``"mid"`` / ``"late"`` → convenience shortcuts.
                ``int`` or ``list[int]`` → explicit block indices.
        """
        self.remove_dispersive_hooks()
        res_blocks = [m for m in self.mlp if isinstance(m, ResidualBlock)]
        n = len(res_blocks)
        if n == 0:
            return

        # Resolve convenience strings
        if layer_indices is None:
            indices = list(range(n))
        elif layer_indices == "early":
            indices = [0]
        elif layer_indices == "mid":
            indices = [n // 2]
        elif layer_indices == "late":
            indices = [n - 1]
        elif isinstance(layer_indices, int):
            indices = [layer_indices]
        else:
            indices = list(layer_indices)

        for idx in indices:
            if 0 <= idx < n:
                handle = res_blocks[idx].register_forward_hook(self._hook_fn)
                self._dispersive_hooks.append(handle)

    def _hook_fn(self, module, input, output):
        self._intermediate_features.append(output)

    def remove_dispersive_hooks(self):
        for h in self._dispersive_hooks:
            h.remove()
        self._dispersive_hooks.clear()
        self._intermediate_features.clear()

    def get_intermediate_features(self):
        feats = list(self._intermediate_features)
        self._intermediate_features.clear()
        return feats

    # -- Forward ----------------------------------------------------------

    def forward(self, noisy_action, obs_features, t):
        t_emb = self.time_mlp(t)
        x = torch.cat([noisy_action, obs_features, t_emb], dim=-1)
        x = self.mlp(x)
        return self.out_proj(x)