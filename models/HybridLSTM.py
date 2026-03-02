
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.AuxModels import VisionEncoder

class ExampleNetwork(nn.Module):
    """
    LSTM-enhanced policy network with downsampled temporal memory.
    Maintains a buffer of observations sampled every N steps for efficient long-term memory.
    
    Args:
        memory_length: Number of historical observations to keep (e.g., 5)
        memory_stride: Sample every Nth observation (e.g., 5 = keep every 5th step)
        Total temporal window = memory_length * memory_stride steps
        Example: 5 memories at stride 5 = 25 steps = 0.5 seconds at 50Hz
    """
    def __init__(
        self, 
        state_dim=3, 
        action_dim=2,
        encoder=VisionEncoder(),
        lstm_hidden_size=128,
        lstm_num_layers=1,
        memory_length=5,  # Keep 5 observations
        memory_stride=5   # Sample every 5 steps, total window of 25 steps
        ):
        super(ExampleNetwork, self).__init__()
        
        # Vision encoder (CNN for LIDAR)
        self.conv_layers = encoder
        conv_output_size = self.conv_layers.output_size
        
        # Memory configuration
        self.memory_length = memory_length  # Number of observations in sequence
        self.memory_stride = memory_stride  # Steps between samples
        self.step_counter = 0  # Track when to sample
        
        # --- State normalization (baked into the network for easy deployment) ---
        # Each row is [min, max] for the corresponding state dimension.
        # Values outside these ranges are clamped to [-1, 1].
        # Default ranges (state_dim=4): vel_x, vel_y, ang_vel_z, accel_x
        default_ranges = [
            [-5.0, 20.0],   # linear_vel_x   (m/s)
            [-5.0,  5.0],   # linear_vel_y   (m/s)
            [-15.0, 15.0],  # ang_vel_z      (rad/s)
            [-10.0, 10.0],  # linear_accel_x (m/s²)
        ]
        state_lo = torch.tensor([r[0] for r in default_ranges[:state_dim]], dtype=torch.float32)
        state_hi = torch.tensor([r[1] for r in default_ranges[:state_dim]], dtype=torch.float32)
        self.register_buffer('state_lo', state_lo)
        self.register_buffer('state_hi', state_hi)
        
        # Combine CNN features with state vector
        feature_input_size = conv_output_size + state_dim
        
        # Project features to LSTM input size with gradual compression + dropout
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_input_size, 768),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, lstm_hidden_size),
            nn.LeakyReLU()
        )
        
        # LSTM for temporal modeling
        # 2-layer LSTM captures richer temporal dynamics for overtaking
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1 if lstm_num_layers > 1 else 0.0
        )
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
    
        # Compact policy head with LayerNorm for cross-map generalization
        self.fc_layers = nn.Sequential(
            nn.LayerNorm(lstm_hidden_size),
            nn.Linear(lstm_hidden_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )

        # Head for the mean (mu) of the action distribution
        self.mean_head = nn.Linear(32, action_dim)
        
        # log_std head (learnable parameter for exploration)
        self.log_std_head = nn.Linear(32, action_dim)
    
    def get_init_hidden(self, batch_size, device, transpose=False):
        """Initialize hidden and cell states for LSTM."""
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        if transpose:
            h0 = h0.transpose(0, 1).contiguous()
            c0 = c0.transpose(0, 1).contiguous()
        return (h0, c0)
    
    def create_observation_buffer(self, batch_size, device):
        """
        Create a buffer to store recent observations for LSTM input.
        Returns a buffer of shape (batch_size, memory_length, feature_size)
        """
        self.step_counter = 0  # Reset stride counter with new buffer
        return torch.zeros(batch_size, self.memory_length, self.lstm_hidden_size).to(device)

    def forward(self, scan_tensor, state_tensor, obs_buffer, hidden_h, hidden_c):
        """
        Args:
            scan_tensor: (batch, 1, lidar_beams) - Current observation
            state_tensor: (batch, state_dim) - Current state
            obs_buffer: (batch, memory_length, lstm_hidden_size) - Historical features buffer
            hidden: Optional LSTM hidden state tuple (h, c)
        Returns:
            loc, scale: Action distribution parameters
            obs_buffer: Updated observation buffer (shifted + new observation)
            hidden: Updated LSTM hidden state (for next step)
        """
        batch_size = scan_tensor.shape[0]
        device = scan_tensor.device
        # Update batch_size after potential reshaping
        batch_size = scan_tensor.shape[0]
        
        # Initialize buffers if not provided
        if obs_buffer is None:
            obs_buffer = self.create_observation_buffer(batch_size, device)
        if hidden_h is None or hidden_c is None:
            hidden = self.get_init_hidden(batch_size, device)
        else:
            # Transpose hidden from [batch, num_layers, hidden] to [num_layers, batch, hidden]
            hidden = (hidden_h.transpose(0, 1).contiguous(), hidden_c.transpose(0, 1).contiguous())
        
        # CNN feature extraction and concatenation
        scan_normalized = scan_tensor.clamp(0.0, 30.0) / 30.0 # TODO: Changes to 10m

        vision_features = self.conv_layers(scan_normalized)
        
        # Normalize state to [-1, 1] using registered physical ranges
        state_normalized = 2.0 * (state_tensor - self.state_lo) / (self.state_hi - self.state_lo) - 1.0
        state_normalized = state_normalized.clamp(-1.0, 1.0)
        
        combined_features = torch.cat((vision_features, state_normalized), dim=1)
        
        # Project to LSTM input size
        current_feature = self.feature_projection(combined_features)
        
        # Update observation buffer (shift left, add new observation at end)
        self.step_counter += 1
        if self.step_counter % self.memory_stride == 0:
            # Stride step: shift history left and append current feature
            obs_buffer_updated = torch.cat([
                obs_buffer[:, 1:, :],  # Drop oldest observation
                current_feature.unsqueeze(1)  # Add newest observation
            ], dim=1)
        else:
            # Between strides: overwrite the last slot with current feature
            # so the LSTM always sees fresh perception, but history stays stable
            obs_buffer_updated = obs_buffer.clone()
            obs_buffer_updated[:, -1, :] = current_feature
        
        # LSTM forward pass on observation sequence
        lstm_out, hidden_new = self.lstm(obs_buffer_updated, hidden)
        lstm_final = lstm_out[:, -1, :]
        
        # Policy head
        x = self.fc_layers(lstm_final)
        loc = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -2.0, 2.0)  # Prevent scale collapse: min=exp(-2)≈0.135 (more exploration)
        scale = torch.exp(log_std)
        scale = torch.clamp(scale, min=0.1, max=10.0)  # Higher floor for pretrained model exploration
        
        # Transpose hidden states from [num_layers, batch, hidden] to [batch, num_layers, hidden]
        hidden_transposed = (hidden_new[0].transpose(0, 1).contiguous(), hidden_new[1].transpose(0, 1).contiguous())
        
        return loc, scale, obs_buffer_updated, hidden_transposed[0], hidden_transposed[1]