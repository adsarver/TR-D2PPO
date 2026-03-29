import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.AuxModels import VisionEncoder
from mamba_ssm.modules.mamba2_simple import Mamba2Simple

class CriticNetwork(nn.Module):
    """
    LSTM-enhanced value network with downsampled temporal memory.
    Mirrors the actor's temporal backbone (DiffusionLSTM) so the critic
    has comparable representational capacity for long-horizon returns.

    During rollout the caller manages ``obs_buffer`` and ``hidden`` state
    exactly like the actor — see ``D2PPOAgent.get_action_and_value``.
    During ``learn()`` the critic receives cached ``critic_features``
    from the rollout so it never needs to re-encode with stale hidden state.
    """
    def __init__(
        self, 
        state_dim=4, 
        encoder=None,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        memory_length=48,
        memory_stride=100,
        odom_expand=64,
        proj_hidden=256,
        num_scan_beams=1080,
    ):
        super(CriticNetwork, self).__init__()
        
        # Vision encoder (CNN for LIDAR)
        self.conv_layers = encoder if encoder is not None else VisionEncoder(num_scan_beams=num_scan_beams)
        conv_output_size = self.conv_layers.output_size
        
        # Memory configuration
        self.memory_length = memory_length
        self.memory_stride = memory_stride
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.step_counter = 0
        
        # Combine CNN features with state vector
        self.feature_input_size = conv_output_size + odom_expand
        
        # Odom handling — baked-in normalisation ranges (same as actor)
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

        self.odom_expand_layer = nn.Linear(state_dim, odom_expand)
        
        # Feature projection (CNN+odom → lstm input)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_input_size, proj_hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden, lstm_hidden_size),
            nn.LeakyReLU(),
        )
        
        # LSTM temporal backbone
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1 if lstm_num_layers > 1 else 0.0,
        )
        self.post_lstm_norm = nn.LayerNorm(lstm_hidden_size)
        
        # Value head (maps LSTM output to scalar state value)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(lstm_hidden_size, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
    
    # ------------------------------------------------------------------
    # Buffer / hidden-state management (same API as DiffusionLSTM)
    # ------------------------------------------------------------------

    def create_observation_buffer(self, batch_size, device):
        """Create a buffer to store recent observations for LSTM input."""
        self.step_counter = 0
        return torch.zeros(batch_size, self.memory_length, self.lstm_hidden_size, device=device)

    def get_init_hidden(self, batch_size, device, transpose=False):
        """Return zero-initialised LSTM hidden state."""
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        if transpose:
            h0 = h0.transpose(0, 1).contiguous()  # → (B, num_layers, H)
            c0 = c0.transpose(0, 1).contiguous()
        return (h0, c0)

    # ------------------------------------------------------------------
    # Observation encoding (CNN → projection → LSTM → features)
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
            critic_features: (B, lstm_hidden_size)
            obs_buffer:      updated buffer
            hidden_h:        (B, num_layers, lstm_hidden_size)
            hidden_c:        (B, num_layers, lstm_hidden_size)
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
        vision_features = self.conv_layers(scan_norm)

        # State normalisation (baked ranges, same as actor)
        state_norm = 2.0 * (state_tensor - self.state_lo) / (self.state_hi - self.state_lo + 1e-8) - 1.0
        state_norm = state_norm.clamp(-1.0, 1.0)
        state_feat = self.odom_expand_layer(state_norm)

        combined = torch.cat([vision_features, state_feat], dim=-1)
        current_feature = self.feature_projection(combined)

        # Rolling buffer update (stride-aware, like DiffusionLSTM)
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
        critic_features = self.post_lstm_norm(lstm_out[:, -1, :])

        # Transpose hidden back to (B, num_layers, H)
        hidden_h_out = hidden_new[0].transpose(0, 1).contiguous()
        hidden_c_out = hidden_new[1].transpose(0, 1).contiguous()

        return critic_features, obs_buffer, hidden_h_out, hidden_c_out

    # ------------------------------------------------------------------
    # Forward: used during learn() with cached features
    # ------------------------------------------------------------------

    def forward_from_features(self, critic_features):
        """
        Value prediction from pre-computed (cached) LSTM features.
        Used during ``learn()`` to avoid re-encoding with stale hidden state.
        """
        return self.fc_layers(critic_features)

    def forward(self, scan_tensor, state_tensor, obs_buffer=None,
                hidden_h=None, hidden_c=None):
        """
        Full forward pass: encode observations then predict value.
        Used during rollout when temporal state is properly maintained.

        Returns:
            value: (B, 1) state value estimate
            obs_buffer: updated observation buffer
            hidden_h: updated LSTM hidden h
            hidden_c: updated LSTM hidden c
        """
        critic_features, obs_buffer, hidden_h, hidden_c = self.encode_observation(
            scan_tensor, state_tensor, obs_buffer, hidden_h, hidden_c
        )
        value = self.fc_layers(critic_features)
        return value, obs_buffer, hidden_h, hidden_c


class Mamba2CriticNetwork(nn.Module):
    """
    Mamba2-enhanced value network.
    Mirrors the actor's temporal backbone (DiffusionMamba2) so the critic
    has comparable representational capacity for long-horizon returns.
    """
    def __init__(
        self, 
        state_dim=4, 
        encoder=None,
        d_model=256,
        d_state=16,
        d_conv=4,
        d_head=16,
        expand=2,
        memory_length=48,
        odom_expand=64,
        num_scan_beams=1080,
    ):
        super(Mamba2CriticNetwork, self).__init__()
        
        # Vision encoder (CNN for LIDAR)
        self.conv_layers = encoder if encoder is not None else VisionEncoder(num_scan_beams=num_scan_beams)
        conv_output_size = self.conv_layers.output_size
        
        # Memory configuration
        self.memory_length = memory_length
        self.d_model = d_model
        
        # Odom handling — baked-in normalisation ranges
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

        self.odom_expand_layer = nn.Linear(state_dim, odom_expand)
        
        # Feature projection (CNN+odom → d_model)
        feature_input_size = conv_output_size + odom_expand
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_input_size, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, d_model),
            nn.ReLU(),
        )

        # Mamba2 temporal backbone
        self.pre_mamba_norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2Simple(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            headdim=d_head,
            expand=expand,
        )
        self.norm_layer = nn.LayerNorm(d_model)
        
        # Value head
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def create_observation_buffer(self, batch_size, device):
        """Create a buffer to store recent observations for Mamba input."""
        return torch.zeros(batch_size, self.memory_length, self.d_model, device=device)

    def encode_observation(self, scan_tensor, state_tensor, obs_buffer):
        """
        Encode raw observations through CNN + Mamba2 temporal backbone.
        """
        batch_size = scan_tensor.shape[0]
        device = scan_tensor.device

        if obs_buffer is None:
            obs_buffer = self.create_observation_buffer(batch_size, device)
        if obs_buffer.device != device:
            obs_buffer = obs_buffer.to(device)

        # Scan normalisation
        scan_norm = scan_tensor.clamp(0.0, 10.0) / 10.0

        # CNN features
        vision_features = self.conv_layers(scan_norm)

        # State normalisation
        state_norm = 2.0 * (state_tensor - self.state_lo) / (self.state_hi - self.state_lo + 1e-8) - 1.0
        state_norm = state_norm.clamp(-1.0, 1.0)
        state_feat = self.odom_expand_layer(state_norm)

        combined = torch.cat([vision_features, state_feat], dim=-1)
        projected = self.feature_projection(combined)

        # Rolling buffer update
        obs_buffer = torch.cat([
            obs_buffer[:, 1:, :],
            projected.unsqueeze(1),
        ], dim=1)

        # Mamba2 temporal pass
        obs_normed = self.pre_mamba_norm(obs_buffer)
        mamba_out = self.mamba(obs_normed)
        critic_features = self.norm_layer(mamba_out[:, -1, :])

        return critic_features, obs_buffer

    def forward_from_features(self, critic_features):
        """
        Value prediction from pre-computed (cached) Mamba features.
        Used during ``learn()`` to avoid re-encoding.
        """
        return self.fc_layers(critic_features)

    def forward(self, scan_tensor, state_tensor, obs_buffer=None):
        """
        Full forward pass: encode observations then predict value.
        """
        critic_features, obs_buffer = self.encode_observation(
            scan_tensor, state_tensor, obs_buffer
        )
        value = self.fc_layers(critic_features)
        return value, obs_buffer