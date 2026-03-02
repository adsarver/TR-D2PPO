
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.AuxModels import VisionEncoder
from mamba_ssm.modules.mamba2_simple import Mamba2Simple

class OldMamba2Network(nn.Module):
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
        state_dim=4, 
        action_dim=2,
        encoder=VisionEncoder(),
        d_model=256,
        d_state=16,
        d_conv=4,
        d_head=16,
        expand=2,
        num_layers=4,
        memory_length=48    # Sequence length
        ):
        super(OldMamba2Network, self).__init__()
        
        # Vision encoder (CNN for LIDAR)
        self.conv_layers = encoder
        conv_output_size = self.conv_layers.output_size
        
        # Memory configuration
        self.memory_length = memory_length  # Number of observations in sequence
        self.d_model = d_model
        
        # Combine CNN features with state vector
        self.feature_input_size = conv_output_size + state_dim
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_input_size, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, d_model),
            nn.ReLU()
        )
        
        # Mamba for temporal modeling
        self.mamba = Mamba2Simple(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                headdim=d_head,
                expand=expand
            )
        
        
        # Policy head (maps LSTM output to action distribution) - Adjusted for larger hidden size
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, 496),
            nn.ReLU(),
            nn.Linear(496, 382),
            nn.ReLU(),
            nn.Linear(382, 296),
            nn.ReLU(),
            nn.Linear(296, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Head for the mean (mu) of the action distribution
        self.mean_head = nn.Linear(32, action_dim)
        
        # log_std head (learnable parameter for exploration)
        self.log_std_head = nn.Linear(32, action_dim)
    
    @torch.jit.export
    def create_observation_buffer(self, batch_size:int, device:torch.device):
        """
        Create a buffer to store recent observations for LSTM input.
        Returns a buffer of shape (batch_size, memory_length, feature_size)
        """
        # Placeholder - will be filled with actual encoded features
        return torch.zeros(batch_size, self.memory_length, self.d_model).to(device)

    def forward(self, scan_tensor, state_tensor, obs_buffer):
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
        
        # Initialize buffers if not provided
        if obs_buffer is None:
            obs_buffer = self.create_observation_buffer(batch_size, device)
            
        # CNN feature extraction and concatenation
        vision_features = self.conv_layers(scan_tensor)
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        
        combined_features = self.feature_projection(combined_features)
        
        # Update buffer: shift left, add new sample
        obs_buffer = torch.cat([
            obs_buffer[:, 1:, :],  # Drop oldest
            combined_features.unsqueeze(1)  # Add newest
        ], dim=1)
                
        # LSTM forward pass on observation sequence
        transformer_out = self.mamba(obs_buffer)
        transformer_final = transformer_out[:, -1, :]
        
        # Policy head
        x = self.fc_layers(transformer_final)
        loc = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -5.0, 2.0)  # Prevent scale collapse: min=exp(-5)≈0.007
        scale = torch.exp(log_std)
        scale = torch.clamp(scale, min=0.01, max=10.0)  # Hard floor on scale
                
        return loc, scale, obs_buffer