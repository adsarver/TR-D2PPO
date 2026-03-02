import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalDenoisingMLP(nn.Module):
    """
    Noise prediction network ε_θ(a^k, k, o).
    [768, 768, 768] MLP with ResidualBlock connections + sinusoidal time
    embeddings.  Forward hooks expose intermediate features for dispersive loss.
    """
    def __init__(self, action_dim, obs_feature_dim,
                 time_emb_dim=32, hidden_dims=(768, 768, 768)):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        input_dim = action_dim + obs_feature_dim + time_emb_dim
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(ResidualBlock(in_dim, h_dim))
            in_dim = h_dim
        self.mlp_blocks = nn.ModuleList(layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], action_dim),
        )

        self._intermediate_features = {}
        self._hooks = []

    # --- dispersive-loss hooks ---

    def register_dispersive_hooks(self, layer_indices=None):
        self.remove_dispersive_hooks()
        n = len(self.mlp_blocks)
        if layer_indices is None:
            layer_indices = list(range(n))
        elif layer_indices == "early":
            layer_indices = [0]
        elif layer_indices == "mid":
            layer_indices = [n // 2]
        elif layer_indices == "late":
            layer_indices = [n - 1]
        elif isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        for idx in layer_indices:
            def hook_fn(module, input, output, idx=idx):
                self._intermediate_features[idx] = output
            h = self.mlp_blocks[idx].register_forward_hook(hook_fn)
            self._hooks.append(h)

    def remove_dispersive_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._intermediate_features = {}

    def get_intermediate_features(self):
        return self._intermediate_features

    def forward(self, noisy_action, obs_features, timestep):
        t_emb = self.time_mlp(timestep)
        x = torch.cat([noisy_action, obs_features, t_emb], dim=-1)
        self._intermediate_features = {}
        for block in self.mlp_blocks:
            x = block(x)
        return self.output_proj(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = nn.Linear(in_features, out_features)
        
        self.norm2 = nn.LayerNorm(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        
        self.projection = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        # Check for NaN in input
        if torch.isnan(x).any():
            print(f"NaN input to ResidualBlock!")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Branch 1
        out = self.norm1(x)
        out = self.linear1(out)
        out = F.relu(out)
        
        # Branch 2
        out = self.norm2(out)
        out = self.linear2(out)
        
        # Residual connection
        result = F.relu(out + self.projection(x))
        
        # Safety check
        if torch.isnan(result).any():
            print(f"NaN output from ResidualBlock!")
            result = torch.nan_to_num(result, nan=0.0)
        
        return result
    
    
class VisionEncoder2d(nn.Module):
    def __init__(self, image_size=64):
        super(VisionEncoder2d, self).__init__()
                
        # Input shape: (agents, 1, image_size, image_size)
        # Based off of TinyLidarNet from: https://arxiv.org/pdf/2410.07447
        # Adapted for flexible input sizes with adaptive pooling
        # Added dropout for cross-track generalization
        
        # Calculate adaptive kernel sizes based on image size
        # For image_size=64: k1=5, s1=2 -> 30x30
        # For image_size=128: k1=7, s1=2 -> 61x61
        k1 = max(3, min(7, image_size // 12))
        k2 = max(3, min(5, image_size // 16))
        
        self.conv_layers = nn.Sequential(
            # First conv: Reduce spatial dimensions by ~2x
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=k1, stride=2, padding=k1//2),
            nn.GroupNorm(1, 24),
            nn.ReLU(),
            
            # Second conv: Reduce by ~2x
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=k2, stride=2, padding=k2//2),
            nn.GroupNorm(1, 36),
            nn.ReLU(),
            
            # Third conv: Reduce by ~2x
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 48),
            nn.ReLU(),
            
            # Fourth conv: Increase channels, same spatial size
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            
            # Fifth conv: Final feature extraction
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            
            # Adaptive pooling to fixed size for consistent output
            # This ensures output is always (64, 4, 4) regardless of input size
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        # Fixed output size: 64 channels * 4 * 4 = 1024
        self.output_size = 64 * 4 * 4

    def forward(self, scan_tensor):
        return self.conv_layers(scan_tensor)
    
class VisionEncoder(nn.Module):
    def __init__(self, num_scan_beams=1080):
        super(VisionEncoder, self).__init__()
        
        # Input shape: (batch_size, 1, num_scan_beams)
        # Based off of TinyLidarNet from: https://arxiv.org/pdf/2410.07447
        # Added dropout for cross-track generalization
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=10, stride=4),
            nn.GroupNorm(1, 24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=24, out_channels=36, kernel_size=8, stride=4),
            nn.GroupNorm(1, 36),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=36, out_channels=48, kernel_size=4, stride=2),
            nn.GroupNorm(1, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten()
        )
        
        # Calculate the output size of the conv layers
        dummy_input = torch.randn(1, 1, num_scan_beams)
        self.output_size = self._get_conv_output_size(dummy_input)

    def _get_conv_output_size(self, x):
        x = self.conv_layers(x)
        return int(np.prod(x.size()[1:]))

    def forward(self, scan_tensor):
        # Handle batched inputs from replay buffer: [batch, agents, 1, scan] -> [batch*agents, 1, scan]
        original_shape = scan_tensor.shape
        if len(original_shape) == 4:  # [batch, agents, 1, scan]
            batch_size, num_agents = original_shape[0], original_shape[1]
            scan_tensor = scan_tensor.reshape(batch_size * num_agents, original_shape[2], original_shape[3])
            features = self.conv_layers(scan_tensor)
            # Reshape back: [batch*agents, features] -> [batch, agents, features]
            features = features.reshape(batch_size, num_agents, -1)
            return features
        else:
            return self.conv_layers(scan_tensor)