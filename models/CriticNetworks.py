class CriticNetwork(nn.Module):
    """
    LSTM-enhanced value network with downsampled temporal memory.
    Maintains a buffer of observations sampled every N steps for efficient long-term memory.
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
        num_layers=4,
        memory_length=48,    # Sequence length
        odom_expand=64
        ):
        super(CriticNetwork, self).__init__()
        
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
        self.norm_layer = nn.LayerNorm(self.feature_input_size)
        
        # Feature projection
        # self.feature_projection = nn.Sequential(
        #     nn.Linear(self.feature_input_size, 768),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(768, d_model),
        #     nn.ReLU()
        # )
        
        # Mamba for temporal modeling
        # self.mamba = Mamba2Simple(
        #         d_model=d_model,
        #         d_state=d_state,
        #         d_conv=d_conv,
        #         headdim=d_head,
        #         expand=expand
        # )
        
        # Value head (maps LSTM output to state value) - Adjusted for larger hidden size
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_input_size, self.feature_input_size // 2),
            nn.ReLU(),
            nn.Linear(self.feature_input_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def create_observation_buffer(self, batch_size, device):
        """Create a buffer to store recent observations for LSTM input."""
        return torch.zeros(batch_size, self.memory_length, self.d_model).to(device)

    def forward(self, scan_tensor, state_tensor, obs_buffer=None, max_speed_estimate=20):
        """
        Args:
            scan_tensor: (batch, 1, lidar_beams) or (T, B, 1, lidar_beams) - Current observation
            state_tensor: (batch, state_dim) or (T, B, state_dim) - Current state
            obs_buffer: (batch, memory_length, lstm_hidden_size) - Historical features buffer
            hidden: Optional LSTM hidden state tuple (h, c)
        Returns:
            value: State value estimate
            obs_buffer: Updated observation buffer
            hidden: Updated LSTM hidden state
        """
        
        batch_size = scan_tensor.shape[0]
        device = scan_tensor.device
        original_shape = scan_tensor.shape
        if len(original_shape) == 4:  # [batch, agents, 1, scan]
            batch_size, num_agents = original_shape[0], original_shape[1]
            # Flatten batch and agents dimensions
            scan_tensor = scan_tensor.reshape(batch_size * num_agents, original_shape[2], original_shape[3])
            state_tensor = state_tensor.reshape(batch_size * num_agents, -1)
            needs_reshape = True
        else:
            batch_size = scan_tensor.shape[0]
            needs_reshape = False
        
        # Initialize buffers if not provided
        if obs_buffer is None:
            obs_buffer = self.create_observation_buffer(batch_size, device)
        if obs_buffer.device != device:
            obs_buffer = obs_buffer.to(device)
            
        # CNN feature extraction and concatenation
        vision_features = self.conv_layers(scan_tensor)
        state_tensor = state_tensor / max_speed_estimate  # Normalize by max speed estimate
        state_tensor = self.odom_expand(state_tensor)
        combined_features = torch.cat((vision_features, state_tensor), dim=1)
        combined_features = self.norm_layer(combined_features)
        
        # # Project to LSTM input size
        # current_feature = self.feature_projection(combined_features)
        # # Update observation buffer (sliding window)
        # obs_buffer_updated = torch.cat([
        #     obs_buffer[:, 1:, :],  # Drop oldest
        #     current_feature.unsqueeze(1)  # Add newest
        # ], dim=1)
        
        
        # Mamba forward pass
        # mamba_out = self.mamba(obs_buffer)
        # mamba_final = mamba_out[:, -1, :]

        # Value head
        value = self.fc_layers(combined_features)
        
        if needs_reshape:
            value = value.reshape(batch_size, num_agents)
    
        
        return value#, obs_buffer_updated