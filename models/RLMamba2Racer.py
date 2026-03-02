import torch
import torch.nn as nn
from models.AuxModels import VisionEncoder
from mamba_ssm.modules.mamba2_simple import Mamba2Simple

class RLMamba2Racer(nn.Module):
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
        memory_length=48,    # Sequence length
        odom_expand = 64
        ):
        super(RLMamba2Racer, self).__init__()
        
        # Vision encoder (CNN for LIDAR)
        self.conv_layers = encoder
        conv_output_size = self.conv_layers.output_size
        
        # Memory configuration
        self.memory_length = memory_length  # Number of observations in sequence
        self.d_model = d_model
        
        # Combine CNN features with state vector
        self.feature_input_size = conv_output_size + odom_expand
        
        # Odom handling
        self.odom_expand = nn.Linear(state_dim, odom_expand)
        self.norm_layer = nn.LayerNorm(self.d_model)
        
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
        
        # Pre-Mamba normalization (critical for preventing NaN in Mamba's SSM)
        self.pre_mamba_norm = nn.LayerNorm(d_model)
        
        # Policy head with residual connections (maps LSTM output to action distribution)
        self.fc_layers = nn.Sequential(
            # First residual block: 256 -> 496 -> 496
            ResidualBlock(d_model, 496),
            # Second residual block: 496 -> 382 -> 382  
            ResidualBlock(496, 382),
            # Third residual block: 382 -> 296 -> 296
            ResidualBlock(382, 296),
            # Fourth residual block: 296 -> 192 -> 192
            ResidualBlock(296, 192),
            # Fifth residual block: 192 -> 128 -> 128
            ResidualBlock(192, 128),
            # Sixth residual block: 128 -> 64 -> 64
            ResidualBlock(128, 64),
            # Final projection to 32
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Head for the mean (mu) of the action distribution
        self.mean_head = nn.Linear(32, action_dim)
        
        # State-independent log_std (simple learnable parameter)
        # Much simpler than a full network head, less likely to destabilize pretrained model
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    @torch.jit.export
    def create_observation_buffer(self, batch_size:int, device:torch.device):
        """
        Create a buffer to store recent observations for LSTM input.
        Returns a buffer of shape (batch_size, memory_length, feature_size)
        """
        # Use zeros - will be filled with real features immediately
        return torch.zeros(batch_size, self.memory_length, self.d_model, device=device)

    def forward(self, scan_tensor, state_tensor, obs_buffer, max_speed_estimate=20):
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
        original_shape = scan_tensor.shape
        
        if torch.isinf(scan_tensor).any() or torch.isnan(scan_tensor).any():
            print(f"CRITICAL: Input scan contains INF or NAN")
            scan_tensor = torch.nan_to_num(scan_tensor, posinf=20.0, neginf=0.0)
        
        if torch.isinf(state_tensor).any() or torch.isnan(state_tensor).any():
            print(f"CRITICAL: Input state contains INF or NAN")
            state_tensor = torch.nan_to_num(state_tensor, 0.0)
        
        # Initialize buffers if not provided
        if obs_buffer is None:
            obs_buffer = self.create_observation_buffer(batch_size, device)
        if obs_buffer.device != device:
            obs_buffer = obs_buffer.to(device)
            
        # CNN feature extraction and concatenation
        vision_features = self.conv_layers(scan_tensor)
        
        if torch.isnan(vision_features).any():
            print(f"NaN after CNN! Replacing with zeros")
            vision_features = torch.nan_to_num(vision_features, nan=0.0)
        
        state_tensor = state_tensor / max_speed_estimate  # Normalize by max speed estimate
        state_tensor = self.odom_expand(state_tensor)
        
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        
        # Feature projection
        combined_features = self.feature_projection(combined_features)
        
        if torch.isnan(combined_features).any():
            print(f"NaN after feature_projection! Replacing with zeros")
            combined_features = torch.nan_to_num(combined_features, nan=0.0)
        
        # Update buffer: shift left, add new sample
        obs_buffer = torch.cat([
            obs_buffer[:, 1:, :],  # Drop oldest
            combined_features.unsqueeze(1)  # Add newest
        ], dim=1)
        
        # Normalize buffer before Mamba (critical for SSM stability)
        obs_buffer_normed = self.pre_mamba_norm(obs_buffer)
        
        # Clamp to prevent extreme values going into Mamba
        obs_buffer_normed = torch.clamp(obs_buffer_normed, -10.0, 10.0)
                
        # Mamba forward pass on normalized sequence
        mamba_out = self.mamba(obs_buffer_normed)
        
        if torch.isnan(mamba_out).any():
            print(f"NaN after Mamba! Replacing with zeros")
            mamba_out = torch.nan_to_num(mamba_out, nan=0.0)
        
        mamba_final = mamba_out[:, -1, :]
        
        # Normalize Mamba output
        mamba_final = self.norm_layer(mamba_final)
        
        if torch.isnan(mamba_final).any():
            print(f"NaN after norm_layer! Replacing with zeros")
            mamba_final = torch.nan_to_num(mamba_final, nan=0.0)
        
        # Policy head
        x = self.fc_layers(mamba_final)
        
        if torch.isnan(x).any():
            print(f"NaN after fc_layers! Replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)
        
        loc = self.mean_head(x)
        
        # Clamp loc to prevent extreme values
        loc = torch.clamp(loc, -10.0, 10.0)
        
        # Use state-independent log_std, broadcast to batch size
        log_std = self.log_std.expand_as(loc)
        scale = torch.exp(log_std.clamp(-2.0, 2.0))  # Clamp for stability (scale in [0.135, 7.4]) 
                
        return loc, scale, obs_buffer
    
class ShallowLSTM(nn.Module):
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
        encoder=None,
        lstm_hidden_size=128,
        lstm_num_layers=1,
        memory_length=5,  # Keep 5 observations
        memory_stride=5   # Sample every 5 steps, total window of 25 steps
        ):
        super(ShallowLSTM, self).__init__()
        
        # Vision encoder (CNN for LIDAR)
        self.conv_layers = encoder
        conv_output_size = self.conv_layers.output_size
        
        # Memory configuration
        self.memory_length = memory_length  # Number of observations in sequence
        self.memory_stride = memory_stride  # Steps between samples
        self.step_counter = 0  # Track when to sample
        
        # Combine CNN features with state vector
        feature_input_size = conv_output_size + state_dim
        
        # Project features to LSTM input size with gradual compression + dropout
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization for cross-track generalization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, lstm_hidden_size),
            nn.ReLU()
        )
        
        # LSTM for temporal modeling
        # Input will be sequence of memory_length observations
        # Using nn.LSTM with flatten_parameters disabled for vmap compatibility
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
    
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
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
        # Placeholder - will be filled with actual encoded features
        return torch.zeros(batch_size, self.memory_length, self.lstm_hidden_size).to(device)
