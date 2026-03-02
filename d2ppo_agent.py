from collections import deque
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer, ListStorage
from torchrl.modules import ProbabilisticActor
from torch.distributions import Normal
from models import *
from utils.utils import to_birds_eye
import time

class PPOAgent:
    def __init__(self, num_agents, map_name, steps, params, transfer=[None, None]):
        # --- Hyperparameters ---
        torch.autograd.set_detect_anomaly(True)
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_actor = 3e-5  # Very conservative for pretrained model (reduced from 5e-5)
        self.lr_critic = 1e-4  # Slower critic learning to prevent gradient explosion (reduced from 5e-4)
        self.gamma = 0.99  # Discount factor for future rewards
        self.gae_lambda = 0.95  # Higher lambda for better credit assignment
        self.clip_epsilon = 0.1  # Conservative clip for fine-tuning pretrained model
        self.entropy_coeff = 0.005  # Entropy bonus coefficient for exploration
        self.max_grad_norm_actor = 0.1  # Very tight gradient clipping for actor (reduced from 0.5)
        self.max_grad_norm_critic = 0.5  # Tighter gradient clipping for critic (reduced from 1.0)
        self.state_dim = 4 # x_vel, y_vel, z_ang_vel, x_accel
        self.num_scan_beams = 1080
        self.lidar_fov = 4.7  # Radians
        self.image_size = 256
        self.minibatch_size = 512
        self.epochs = 3  # Reduced for speed - recurrent models need less epochs
        self.epochs_with_demos = 4
        self.bc_epochs = 3  # More BC epochs to learn from demonstrations
        self.params = params
        
        # --- Demonstration Retention ---
        self.demo_buffer = None  # Store demonstrations for continual learning
        
        # --- Waypoints for Raceline Reward ---
        self.waypoints_xy, self.waypoints_s, self.raceline_length = self._load_waypoints(map_name)
        self.last_cumulative_distance = np.zeros(self.num_agents) 
        self.last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        self.start_s = np.zeros(self.num_agents)
        self.current_lap_count = np.zeros(self.num_agents, dtype=int)
        self.last_checkpoint = np.zeros(self.num_agents, dtype=int)  # Track last checkpoint (0-9)
        
        # --- Reward Scalars ---
        self.PROGRESS_REWARD_SCALAR = 48.0  # Increased from 32.0 - main reward signal
        self.LAP_REWARD = 80.0
        self.CHECKPOINT_REWARD = self.LAP_REWARD * 0.1
        self.COLLISION_PENALTY = -4.0  # Reduced from -10.0
        self.SPEED_REWARD = 3.0  # Increased from 1.0
        self.AGENT_COLLISION_PENALTY = -2.0  # Reduced from -5.0 to allow aggressive racing
       
        # --- Networks & Wrappers ---
        
        # Separate encoders for actor and critic to prevent gradient conflicts
        actor_encoder = self._transfer_vision(transfer[0])
        critic_encoder = self._transfer_vision(transfer[1])  # Independent encoder

        # Create the recurrent networks with larger capacity
        self.actor_network = RecurrentActorNetwork(
            self.state_dim, 2, 
            encoder=actor_encoder,
            d_model=512,
            d_state=64,
            d_conv=4,
            expand=2,
            num_layers=4,
            memory_length=1024,
        ).to(self.device)
        
        self.critic_network = RecurrentCriticNetwork(
            self.state_dim, 
            encoder=critic_encoder
        ).to(self.device)
        
        self.actor_network = self._transfer_weights(transfer[0], self.actor_network)
        self.critic_network = self._transfer_weights(transfer[1], self.critic_network)
        
        self.actor_buffer = deque(
            [self.actor_network.create_observation_buffer(num_agents, self.device)],
            maxlen=2
        )
        # self.critic_buffer = deque(
        #     [self.critic_network.create_observation_buffer(num_agents, self.device)],
        #     maxlen=2
        # )
        
        # --- Optimizers ---
        self.actor_optimizer = optim.AdamW(self.actor_network.parameters(), lr=self.lr_actor, weight_decay=0.01)
        self.critic_optimizer = optim.AdamW(self.critic_network.parameters(), lr=self.lr_critic, weight_decay=0.01)
        self.actor_scaler = torch.amp.GradScaler("cuda")
        self.critic_scaler = torch.amp.GradScaler("cuda")
        # --- Storage ---
        self.buffer = TensorDictReplayBuffer(
            storage=ListStorage(max_size=steps)
        )
        
        # --- Diagnostics ---
        self.plot_save_path = "plots/training_diagnostics_history.png"
        plot_dir = os.path.dirname(self.plot_save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        # Initialize storage for historical averages
        self.diagnostic_keys = ["loss_actor", "loss_critic",
                                "entropy", "kl_approx", "clip_fraction", "collisions", "reward"]
        self.diagnostics_history = {key: [] for key in self.diagnostic_keys}
        self.generation_counter = 0 # Track generation for x-axis
        
    def update_buffer_size(self, new_size):
        self.buffer = TensorDictReplayBuffer(
            storage=ListStorage(max_size=new_size)
        )
        print(f"Updated replay buffer size to {new_size}")
    
    def _transfer_weights(self, path, network):
        if path is None:
            return network.to(self.device)

        checkpoint = torch.load(path)
        
        prefix = "0.module."

        state_dict = {}
        has_log_std = False
        
        for k, v in checkpoint.items():
            # Skip old log_std_head weights (incompatible with new log_std parameter)
            if 'log_std_head' in k:
                continue
            
            # Track if checkpoint has trained log_std
            if k == 'log_std' or k.endswith('.log_std'):
                has_log_std = True
                
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                state_dict[new_key] = v
            else: 
                state_dict[k] = v

        if state_dict:
            # Get the network's state dict to check for mismatches
            network_state_dict = network.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            
            for key, value in state_dict.items():
                if key in network_state_dict:
                    if network_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {network_state_dict[key].shape})")
                else:
                    skipped_keys.append(f"{key} (not in network)")
            
            if skipped_keys:
                print(f"Skipped loading weights for keys due to mismatches: {skipped_keys}")
            
            if filtered_state_dict:
                network.load_state_dict(filtered_state_dict, strict=False)
                print("Successfully loaded pre-trained weights!")
            else:
                print("No compatible weights found in checkpoint.")
            
            # Only initialize log_std if it wasn't in the checkpoint (e.g., from BC pretraining)
            if hasattr(network, 'log_std') and not has_log_std:
                nn.init.zeros_(network.log_std)
                print("Initialized log_std parameter (not found in checkpoint)")
            elif has_log_std:
                print("Loaded trained log_std from checkpoint - continuing RL training")
            
        return network.to(self.device)

    def _transfer_vision(self, path):
        new_encoder = VisionEncoder(self.num_scan_beams)
        if path is None:
            return new_encoder.to(self.device)
        
        checkpoint = torch.load(path)
        prefix = "conv_layers."

        encoder_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                encoder_state_dict[new_key] = v
            elif k.startswith("0.module." + prefix):
                new_key = k[len("0.module." + prefix):]
                encoder_state_dict[new_key] = v

        if encoder_state_dict:
            new_encoder.load_state_dict(encoder_state_dict)
            print("Successfully loaded pre-trained encoder weights!")
        else:
            print(checkpoint.keys())
            print(f"Warning: No weights found with prefix '{prefix}'. Starting with a random encoder.")

        return new_encoder.to(self.device)

    def _map_range(self, value, in_min, in_max, out_min=-1, out_max=1):
        if in_max == in_min:
            return out_min if value <= in_min else out_max

        return out_min + (float(value - in_min) / float(in_max - in_min)) * (out_max - out_min)

    def _load_waypoints(self, map_name):
        """
        Loads waypoints from a CSV file for the given map.
        """
        waypoint_file = f"maps/{map_name}/{map_name}_raceline.csv"
        waypoints = np.loadtxt(waypoint_file, delimiter=';')
        waypoints_xy = waypoints[:, 1:3]
        
        # 2. Calculate Cumulative Distance (s)
        positions = waypoints[:, 1:3]
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        waypoints_s = np.insert(np.cumsum(distances), 0, 0)
        raceline_length = waypoints_s[-1]

        return waypoints_xy, waypoints_s, raceline_length

    def _obs_to_tensors(self, obs):
        scans = obs['scans'][:self.num_agents]
        scan_tensors = torch.from_numpy(np.array(scans, dtype=np.float64)).float()

        scan_tensors = scan_tensors.unsqueeze(1)
        
        state_data = np.stack(
            (obs['linear_vels_x'], obs['linear_vels_y'], obs['ang_vels_z'],
             obs['linear_accel_x']), 
            axis=1
        )
        state_tensor = torch.from_numpy(state_data).float()[:self.num_agents]

        return scan_tensors.to(self.device), state_tensor.to(self.device)

    def get_action_and_value(self, scan_tensor, state_tensor, deterministic=False, store=True):
        """
        Gets an action from the Actor and a value from the Critic.
        Simple LSTM networks - no persistent state management needed.
        """
        self.actor_network.eval()
        self.critic_network.eval()
        
        
        if torch.isnan(scan_tensor).any():
            print(f"NaN in scan_tensor! Range: [{scan_tensor.min()}, {scan_tensor.max()}]")
            exit(0)
        if torch.isnan(state_tensor).any():
            print(f"NaN in state_tensor! Range: [{state_tensor.min()}, {state_tensor.max()}]")
            exit(0)
            
        with torch.no_grad():
            action = None
            log_prob = None
            
            if store:
                # Get action from actor
                loc, scale, actor_new_buffer = self.actor_network(
                    scan_tensor[:self.num_agents],
                    state_tensor[:self.num_agents],
                    self.actor_buffer[-1]
                )
                
                if torch.isnan(loc).any():
                    print(f"NaN in action loc! Range: [{loc.min()}, {loc.max()}]")
                    exit(0)
                if torch.isnan(scale).any():
                    print(f"NaN in action scale! Range: [{scale.min()}, {scale.max()}]")
                    exit(0)
                if torch.isnan(self.actor_buffer[-1]).any():
                    print(f"NaN in actor buffer! Range: [{self.actor_buffer[-1].min()}, {self.actor_buffer[-1].max()}]")
                    exit(0)

                # Create distribution and sample action
                dist = Normal(loc, scale)
                
                if torch.isnan(dist.mean).any():
                    print(f"NaN in dist mean! Range: [{dist.mean.min()}, {dist.mean.max()}]")
                    exit(0)
                
                if deterministic:
                    action = loc  # Use mean for deterministic policy
                else:
                    action = dist.rsample()  # Sample for stochastic policy
                    
                if torch.isnan(action).any():
                    print(f"NaN in sampled action! Range: [{action.min()}, {action.max()}]")
                    exit(0)
                
                log_prob = dist.log_prob(action)
            
            # Get value from critic
            value = self.critic_network(
                    scan_tensor[:self.num_agents],
                    state_tensor[:self.num_agents],
            )
            
            if torch.isnan(value).any():
                print(f"NaN in state value! Range: [{value.min()}, {value.max()}]")
                exit(0)
            
            if store:
                self.actor_buffer.append(actor_new_buffer)
                
        return action, log_prob, value
    
    def _compute_gae(self, data: TensorDict) -> TensorDict:
        """
        Vectorized GAE computation - much faster than nested loops.
        Computes advantages using reversed cumulative operations.
        """
        rewards = data.get(("next", "reward")).to(self.device)
        dones = data.get(("next", "done")).float().to(self.device)
        values = data.get("state_value").to(self.device)
        next_values = data.get(("next", "state_value")).to(self.device)
            
        # Squeeze extra dimensions if present: [time_steps, num_agents, 1] -> [time_steps, num_agents]
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
            dones = dones.squeeze(-1)
            values = values.squeeze(-1)
            next_values = next_values.squeeze(-1)
        
        # Handle different shapes
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
            dones = dones.unsqueeze(-1)
            values = values.unsqueeze(-1)
            next_values = next_values.unsqueeze(-1)
            timesteps = rewards.shape[0]
            num_agents = 1
        elif rewards.ndim == 2:
            timesteps = rewards.shape[0]
            num_agents = rewards.shape[1]
        else:
            raise ValueError(f"Unexpected rewards shape after squeezing: {rewards.shape}")
        
        print(f"Computing GAE for {timesteps} timesteps across {num_agents} agents")
        print(f"  Rewards: min={rewards.min():.2f}, max={rewards.max():.2f}, mean={rewards.mean():.2f}")
        print(f"  Values:  min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")
        
        # Debug: Check inputs for NaN
        if torch.isnan(rewards).any():
            print(f"NaN in rewards! Range: [{rewards.min()}, {rewards.max()}]")
            exit(0)
        if torch.isnan(values).any():
            print(f"NaN in values! Range: [{values.min()}, {values.max()}]")
            exit(0)
        if torch.isnan(next_values).any():
            print(f"NaN in next_values! Range: [{next_values.min()}, {next_values.max()}]")
            exit(0)
        
        # Compute TD errors (delta) for all timesteps at once
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Vectorized GAE computation using reversed cumulative product
        advantages = torch.zeros_like(values, device=self.device)
        gae = torch.zeros(num_agents, device=self.device)
        
        # Compute GAE backwards through time (vectorized across agents)
        for t in reversed(range(timesteps)):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        # Normalize advantages (mean=0, std=1) for stable policy gradient
        advantages_flat = advantages.flatten()
        advantages_mean = advantages_flat.mean()
        advantages_std = advantages_flat.std() + 1e-8
        advantages = (advantages - advantages_mean) / advantages_std
        
        # Squeeze if single agent
        if num_agents == 1 and advantages.shape[-1] == 1:
            advantages = advantages.squeeze(-1)
            returns = returns.squeeze(-1)
                    
        data.set("advantage", advantages)
        data.set("value_target", returns)
                
        # Explicitly delete intermediate tensors to free memory
        del deltas, gae, rewards, dones, values, next_values, advantages_flat
        
        return data
    
    def reset_buffers(self, agent_indices=None):
        if agent_indices is None:
            # Reset all
            self.actor_buffer = deque(
                [self.actor_network.create_observation_buffer(self.num_agents, self.device)],
                maxlen=2
            )
            # self.critic_buffer = deque(
            #     [self.critic_network.create_observation_buffer(self.num_agents, self.device)],
            #     maxlen=2
            # )
        else:
            agent_indices = agent_indices[agent_indices < self.num_agents]
            # Reset specific agents
            if self.actor_buffer[-1] is not None:
                for idx in agent_indices:
                    self.actor_buffer[-1][idx] = 0.0
                    # self.critic_buffer[-1][idx] = 0.0
                    
    def store_transition(self, obs, next, action, log_prob, reward, done, value):
        """
        Stores a single step of experience for ALL agents.
        This is a bit complex as we must convert from "list of obs" to "batch."
        """

        next_scans, next_states = self._obs_to_tensors(next)
        
        _, _, next_value = self.get_action_and_value(
            next_scans, next_states, self.params, store=False
        )
        
        done_tensor = torch.tensor(done, dtype=torch.bool).unsqueeze(-1)

        # This dict contains a *batch* of experiences (one for each agent)
        step_data = TensorDict({
            "observation_scan": obs[0],
            "observation_state": obs[1],
            "action": action,
            "action_log_prob": log_prob,
            "state_value": value,
            "next": TensorDict({
                # Only store what's needed for GAE - saves memory
                "state_value": next_value,
                "reward": reward,
                "done": done_tensor,
            })
        }, batch_size=[self.num_agents])
        
        # Add the whole batch to the buffer
        self.buffer.add(step_data.to(self.device))
    
    def _project_to_raceline(self, current_pos, start_idx, lookahead):
        """
        Projects the agent's current position onto the raceline segment defined
        by the search window to get the most accurate, continuous s-distance.
        
        Returns: projected_s (float), global_wp_index (int)
        """
        wp_count = len(self.waypoints_xy)
        
        # Create a wrapped search slice for the waypoints
        search_indices = np.arange(start_idx, start_idx + lookahead) % wp_count
        search_waypoints = self.waypoints_xy[search_indices]
        
        # Find the closest waypoint (W_curr) within the lookahead window
        distances_in_window = np.linalg.norm(search_waypoints - current_pos, axis=1)
        closest_wp_in_window = np.argmin(distances_in_window)
        
        # Map the local index back to the global index (Index C)
        closest_wp_index_global = search_indices[closest_wp_in_window]
        
        # Define the segment W_prev -> W_curr for projection
        W_curr = self.waypoints_xy[closest_wp_index_global]
        W_prev_index = (closest_wp_index_global - 1 + wp_count) % wp_count
        W_prev = self.waypoints_xy[W_prev_index]
        
        # Vector V: Segment direction (W_prev -> W_curr)
        V = W_curr - W_prev
        V_len_sq = np.dot(V, V)
        
        # Vector W: Vector from W_prev to Agent's Pos
        W = current_pos - W_prev
        
        # Calculate projection length (L) of W onto V. L is a scalar.
        if V_len_sq > 1e-6:
            L = np.dot(W, V) / V_len_sq
        else:
            L = 0.0

        # Clamp L to ensure the projected point P' is within the segment [0, 1]
        # L_clamped = np.clip(L, 0.0, 1.0) 
        
        # Calculate the true continuous s-value
        s_prev = self.waypoints_s[W_prev_index]
        s_curr = self.waypoints_s[closest_wp_index_global]
        
        segment_distance = s_curr - s_prev
        
        # Handle the lap wrap-around condition where s_curr is near 0 and s_prev is near max_length
        if segment_distance < 0:
            segment_distance += self.raceline_length
        
        # Projected S value: s(P') = s(W_prev) + L_clamped * segment_distance
        projected_s = s_prev + L * segment_distance
        
        return projected_s, closest_wp_index_global
        
    def calculate_reward(self, next_obs):
        collisions = np.array(next_obs['collisions'][:self.num_agents])
        speeds = np.array(next_obs['linear_vels_x'][:self.num_agents])
        # positions = np.stack([
        #     next_obs['poses_x'][:self.num_agents],
        #     next_obs['poses_y'][:self.num_agents]
        # ], axis=1)
        
        # Vectorized collision detection
        wall_collisions = collisions == 1
        agent_collisions = collisions == 2
        
        rewards = np.zeros(self.num_agents)
        
        # # --- 1. PROGRESS REWARD (Main driver for going fast) ---
        # for i in range(self.num_agents):
        #     projected_s, new_wp_idx = self._project_to_raceline(
        #         positions[i], 
        #         self.last_wp_index[i], 
        #         lookahead=50
        #     )
            
        #     # Calculate progress (handles wrap-around)
        #     progress = projected_s - self.last_cumulative_distance[i]
        #     if progress < -self.raceline_length / 2:
        #         progress += self.raceline_length  # Crossed start/finish
        #     elif progress > self.raceline_length / 2:
        #         progress -= self.raceline_length  # Went backwards
            
        #     # Only reward forward progress
        #     if progress > 0:
        #         rewards[i] += progress * self.PROGRESS_REWARD_SCALAR
        #     else:
        #         rewards[i] += progress * self.PROGRESS_REWARD_SCALAR * 2.0  # Penalize reversing harder
            
        #     # Update trackers
        #     self.last_cumulative_distance[i] = projected_s
        #     self.last_wp_index[i] = new_wp_idx
        
        # --- 2. SPEED BONUS (Reward high speeds) ---
        target_speed = 9.0  # Aggressive target
        speed_bonus = np.clip(speeds - target_speed, 0, 3.0) * self.SPEED_REWARD
        rewards += speed_bonus
        
        # --- 3. OVERTAKE REWARD ---
        # Compare relative positions: reward for being ahead of other agents
        # for i in range(self.num_agents):
        #     my_s = self.last_cumulative_distance[i]
        #     for j in range(self.num_agents):
        #         if i != j:
        #             their_s = self.last_cumulative_distance[j]
        #             # Calculate relative position (positive = I'm ahead)
        #             relative = my_s - their_s
        #             # Handle wrap-around
        #             if relative < -self.raceline_length / 2:
        #                 relative += self.raceline_length
        #             elif relative > self.raceline_length / 2:
        #                 relative -= self.raceline_length
                    
        #             # Small bonus for being ahead of each opponent
        #             if relative > 0:
        #                 rewards[i] += 0.1  # Per-step bonus for each agent behind you
        
        # --- 4. COLLISION PENALTIES (Reduced for agent collisions to allow aggressive racing) ---
        rewards += wall_collisions * self.COLLISION_PENALTY
        rewards += agent_collisions * (self.AGENT_COLLISION_PENALTY * 0.5)  # Less penalty for trading paint
        
        # Convert to tensor
        rewards_tensor = torch.from_numpy(rewards.astype(np.float32)).unsqueeze(-1)
        avg_reward = rewards.mean()
        
        return rewards_tensor, avg_reward
    
    def reset_progress_trackers(self, initial_poses_xy, agent_idxs=None):
        """Resets the cumulative distance tracker for all agents after an episode reset."""
        if agent_idxs is not None:
            agent_idxs = agent_idxs[agent_idxs < self.num_agents]
            for i in agent_idxs:
                current_pos = initial_poses_xy[i]
                
                # Find the globally closest waypoint (no lookahead needed here)
                distances = np.linalg.norm(self.waypoints_xy - current_pos, axis=1)
                closest_wp_index = np.argmin(distances)
                
                start_s_val = self.waypoints_s[closest_wp_index]
                self.last_cumulative_distance[i] = start_s_val
                self.last_wp_index[i] = closest_wp_index
                
                self.start_s[i] = start_s_val
                self.current_lap_count[i] = 0
                
                # Reset checkpoint tracking for crashed agents
                self.last_checkpoint[i] = 0
            return
        
        new_last_cumulative_distance = np.zeros(self.num_agents)
        new_last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        
        # Iterate over all starting positions
        for i in range(self.num_agents):
            current_pos = initial_poses_xy[i]
            
            # Find the globally closest waypoint (no lookahead needed here)
            distances = np.linalg.norm(self.waypoints_xy - current_pos, axis=1)
            closest_wp_index = np.argmin(distances)

            # Set the initial cumulative distance and index            
            start_s_val = self.waypoints_s[closest_wp_index]
            new_last_cumulative_distance[i] = start_s_val
            new_last_wp_index[i] = closest_wp_index
            
            self.start_s[i] = start_s_val
            self.current_lap_count[i] = 0
            
            # Reset checkpoint tracking for crashed agents
            self.last_checkpoint[i] = 0
            
        self.last_cumulative_distance = new_last_cumulative_distance
        self.last_wp_index = new_last_wp_index

    def pretrain_from_demonstrations(self, demo_buffer=None, epochs=2, gradient_accumulation_steps=4, bc_weights=None):
        """
        Supervised learning from human demonstrations using behavior cloning.
        Also stores demos for continual learning (prevents catastrophic forgetting).
        Pretrains BOTH actor (action prediction) and critic (value estimation).
        
        Optimizations:
        - Micro-batching: Process small sequential chunks together
        - Gradient accumulation: Update weights less frequently
        - Reduced epochs: 5 instead of 20 (sequential data = more info per sample)
        """
        # Store demos for continual BC regularization during RL training
        if demo_buffer is not None:
            self.demo_buffer = demo_buffer
            self.demo_pretrain_generation = self.generation_counter  # Mark when pretraining occurred
            print(f"\nPretraining from {len(self.demo_buffer)} human demonstrations...")
            print(f"Stored {len(self.demo_buffer)} demos for continual learning")
            print(f"Config: {epochs} epochs, grad_accum={gradient_accumulation_steps}")
            bc_weights = (1., 1.) # Full weight during pretraining
        elif demo_buffer is None and self.demo_buffer is not None:
            demo_buffer = self.demo_buffer
            print(f"\n  Running BC regularization from stored {len(self.demo_buffer)} human demonstrations...")
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        self.actor_network.train()
        self.critic_network.train()
        
        for epoch in range(epochs):
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0

            actor_buffer = None
            actor_hidden = (None, None)
            critic_buffer = None
            critic_hidden = (None, None)
            
            # Zero gradients at start of epoch
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            update_counter = 0

            for i, d in enumerate(demo_buffer):   
                # Add batch dimension to scan and state
                scan = torch.from_numpy(d['scan']).float().unsqueeze(0).to(self.device)
                scan = to_birds_eye(
                    scan.flatten(1),
                    num_beams=self.num_scan_beams,
                    fov=self.lidar_fov,
                    image_size=self.image_size
                ).unsqueeze(1).to(self.device)
                state = torch.from_numpy(d['state']).float().unsqueeze(0).to(self.device)
                action = torch.from_numpy(d['action']).float().unsqueeze(0).to(self.device)
                value = torch.tensor([d['value']], dtype=torch.float32).unsqueeze(0).to(self.device)

                # --- Actor forward pass ---
                predicted, _, actor_buffer_n, actor_hidden_h, actor_hidden_c = self.actor_network(
                    scan, state, actor_buffer, actor_hidden[0], actor_hidden[1]
                )
                actor_loss = torch.nn.functional.huber_loss(predicted, action, delta=1.0) * bc_weights[0] / gradient_accumulation_steps
                actor_loss.backward()
                epoch_actor_loss += actor_loss.item() * gradient_accumulation_steps / bc_weights[0]
                
                # Detach LSTM states to prevent backprop through entire sequence
                actor_buffer = actor_buffer_n.detach()
                actor_hidden = (actor_hidden_h.detach(), actor_hidden_c.detach())

                # --- Critic forward pass ---
                # predicted_values, critic_buffer_n, critic_hidden_h, critic_hidden_c = self.critic_network(
                #     scan, state, critic_buffer, critic_hidden[0], critic_hidden[1]
                # )
                # critic_loss = torch.nn.functional.huber_loss(predicted_values, value, delta=10.0) * bc_weights[1] / gradient_accumulation_steps
                # critic_loss.backward()
                # epoch_critic_loss += critic_loss.item() * gradient_accumulation_steps / bc_weights[1]
                
                # Detach LSTM states to prevent backprop through entire sequence
                # critic_buffer = critic_buffer_n.detach()
                # critic_hidden = (critic_hidden_h.detach(), critic_hidden_c.detach())
                
                # Update weights every gradient_accumulation_steps
                update_counter += 1
                if update_counter >= gradient_accumulation_steps:
                    self.actor_optimizer.step()
                    # self.critic_optimizer.step()
                    self.actor_optimizer.zero_grad()
                    # self.critic_optimizer.zero_grad()
                    update_counter = 0
                
                    progress = (i + 1) / len(demo_buffer) * 100
                    avg_actor_loss = epoch_actor_loss / (i + 1)
                    # avg_critic_loss = epoch_critic_loss / (i + 1)
                    print(f"    Epoch {epoch+1}/{epochs}, Actor Loss: {avg_actor_loss:.4f} - {progress:.1f}% complete", end='\r')
            
            # Final update if there are remaining gradients
            if update_counter > 0:
                self.actor_optimizer.step()
                # self.critic_optimizer.step()
                self.actor_optimizer.zero_grad()
                # self.critic_optimizer.zero_grad()
                
            avg_actor_loss = epoch_actor_loss / len(demo_buffer)
            # avg_critic_loss = epoch_critic_loss / len(demo_buffer)
            
            if i % 5 == 0: print(f"    Epoch {epoch+1}/{epochs}, Actor Loss: {avg_actor_loss:.4f}")
            
            total_actor_loss += avg_actor_loss
            # total_critic_loss += avg_critic_loss
        
        self.buffer.empty()
        print(f"Pretraining complete. Avg Actor Loss: {total_actor_loss/epochs:.4f}, Avg Critic Loss: {total_critic_loss/epochs:.4f}\n")

    def learn(self, collisions, reward):
        print("Starting learning phase...")
        print(f"Buffer size: {len(self.buffer)}")
        data = self.buffer.sample(batch_size=len(self.buffer))
        
        print("Preparing diagnostics storage...")
        current_gen_diagnostics = {key: [] for key in self.diagnostic_keys}
        current_gen_diagnostics["collisions"] = [collisions]
        current_gen_diagnostics["reward"] = [reward]
        
        # Compute GAE on full data first (temporal order preserved)
        with torch.no_grad():
            data = self._compute_gae(data)
        
        # Move all data to GPU once
        obs_scan_all = data["observation_scan"]
        obs_state_all = data["observation_state"]
        actions_all = data["action"]
        old_log_probs_all = data["action_log_prob"].sum(-1)
        advantages_all = data["advantage"]
        value_targets_all = data["value_target"]
        
        num_timesteps = len(data)
        update_every = 64  # Larger BPTT window = fewer updates = faster training
        
        self.actor_network.train()
        self.critic_network.train()
        
        print(f"Training: {self.epochs} epochs, {num_timesteps} timesteps, update every {update_every} steps")
        
        for epoch in range(self.epochs):
            # Fresh Mamba states each epoch
            actor_mamba_state = self.actor_network.create_observation_buffer(self.num_agents, self.device)
            # critic_mamba_state = self.critic_network.create_observation_buffer(self.num_agents, self.device)
            
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            num_updates = 0
            
            # Accumulators for gradient accumulation
            all_locs = []
            all_scales = []
            all_values = []
            
            for t in range(num_timesteps):
                # Get single timestep for all agents: [num_agents, ...]
                obs_scan_t = obs_scan_all[t]
                obs_state_t = obs_state_all[t]
                
                # Forward pass with mixed precision
                loc_t, scale_t, actor_mamba_state = self.actor_network(
                    obs_scan_t, obs_state_t, actor_mamba_state
                )
                value_t = self.critic_network(
                    obs_scan_t, obs_state_t
                )
                
                all_locs.append(loc_t)
                all_scales.append(scale_t)
                all_values.append(value_t)
                
                # Update weights every N steps (or at the end)
                if (t + 1) % update_every == 0 or t == num_timesteps - 1:
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    
                    window_start = t + 1 - len(all_locs)
                    window_end = t + 1
                    
                    # Stack accumulated outputs
                    loc = torch.stack(all_locs, dim=0)
                    scale = torch.stack(all_scales, dim=0)
                    predicted_values = torch.stack(all_values, dim=0)
                    
                    # Get corresponding targets for this window
                    actions = actions_all[window_start:window_end]
                    old_log_probs = old_log_probs_all[window_start:window_end]
                    advantages = advantages_all[window_start:window_end]
                    value_targets = value_targets_all[window_start:window_end]
                    
                    # Compute new log probs
                    new_log_probs = torch.distributions.Normal(loc, scale).log_prob(actions).sum(-1)
                    
                    # PPO ratio and clipped objective
                    log_ratio = new_log_probs.view(-1) - old_log_probs.view(-1)
                    ratio = torch.exp(log_ratio)
                    
                    with torch.no_grad():
                        kl_approx = ((ratio - 1) - log_ratio).mean()
                        clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
                    
                    surr1 = ratio * advantages.view(-1)
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages.view(-1)
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    entropy = torch.distributions.Normal(loc.detach(), scale).entropy().sum(-1).mean()
                    entropy_loss = -self.entropy_coeff * entropy
                    actor_loss = policy_loss + entropy_loss
                
                    critic_loss = nn.functional.mse_loss(
                        predicted_values.view(-1), value_targets.view(-1), reduction='mean'
                    )
                        
                    # Truncated BPTT: detach states to prevent backprop through entire sequence
                    actor_mamba_state = actor_mamba_state.detach()
                    # critic_mamba_state = critic_mamba_state.detach()
                    
                    # Backward and update with mixed precision
                    self.actor_scaler.scale(actor_loss).backward()
                    self.actor_scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=self.max_grad_norm_actor)
                    self.actor_scaler.step(self.actor_optimizer)
                    self.actor_scaler.update()

                    self.critic_scaler.scale(critic_loss).backward()
                    self.critic_scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=self.max_grad_norm_critic)
                    self.critic_scaler.step(self.critic_optimizer)
                    self.critic_scaler.update()

                    epoch_actor_loss += actor_loss.item()
                    epoch_critic_loss += critic_loss.item()
                    num_updates += 1
                    
                    # Collect diagnostics
                    current_gen_diagnostics["loss_actor"].append(actor_loss.detach().cpu().item())
                    current_gen_diagnostics["loss_critic"].append(critic_loss.detach().cpu().item())
                    current_gen_diagnostics["entropy"].append(entropy.detach().cpu().item())
                    current_gen_diagnostics["kl_approx"].append(kl_approx.cpu().item())
                    current_gen_diagnostics["clip_fraction"].append(clip_fraction.cpu().item())
                    
                    # Reset accumulators
                    all_locs = []
                    all_scales = []
                    all_values = []
            
            avg_entropy = np.mean(current_gen_diagnostics["entropy"]) if current_gen_diagnostics["entropy"] else 0.0
            print(f"  Epoch {epoch+1}/{self.epochs}: Actor Loss: {epoch_actor_loss/num_updates:.4f}, Critic Loss: {epoch_critic_loss/num_updates:.4f}, Entropy: {avg_entropy:.4f}")
            
            torch.cuda.empty_cache()
                        
        self.generation_counter += 1
        # Track min, max, avg for each diagnostic metric
        for key in self.diagnostic_keys:
            values = current_gen_diagnostics.get(key)
            if values:
                avg_value = np.mean(values)
                min_value = np.min(values)
                max_value = np.max(values)
                # Store as tuple (avg, min, max)
                if key not in self.diagnostics_history:
                    self.diagnostics_history[key] = []
                self.diagnostics_history[key].append((avg_value, min_value, max_value))
        
        if self.generation_counter > 0: self._plot_historical_diagnostics()
        
        # Clear the buffer for the next "generation"
        self.buffer.empty()
        # self.reset_buffers()
        
        # Explicitly free GPU memory
        del obs_scan_all, obs_state_all, actions_all, old_log_probs_all, advantages_all, value_targets_all
        del data
        torch.cuda.empty_cache()
        
        print("Learning complete.")
        
    def _plot_historical_diagnostics(self):
        """
        Generates and saves a plot showing the trend of average diagnostics
        across all completed generations. Overwrites the file each time.
        """
        # Define keys to plot (exclude generation if it's not in history dict)
        keys_to_plot = [k for k in self.diagnostic_keys if k != "generation" and k in self.diagnostics_history]
        num_metrics = len(keys_to_plot)

        if num_metrics == 0 or self.generation_counter == 0:
            print("No diagnostics data to plot yet.")
            return

        plt.style.use('dark_background')
        fig, axes = plt.subplots(num_metrics, 1, figsize=(25, 5 * num_metrics), sharex=True)
        if num_metrics == 1: axes = [axes] # Ensure axes is always iterable
        
        # Set global font size
        plt.rcParams['font.size'] = 24  # Adjust as needed

        # Set global line width
        plt.rcParams['lines.linewidth'] = 3 
        
        # X-axis: Generation number
        x_axis = np.arange(1, self.generation_counter + 1) # Generations 1, 2, 3...
        # Plot each metric's history
        for idx, key in enumerate(keys_to_plot):
            values = self.diagnostics_history.get(key, [])
            ax = axes[idx] # Get the correct subplot axis

            if not values: # Skip if no data for this key
                ax.set_ylabel(key)
                ax.grid(True)
                continue

            # Convert to numpy array of shape (generations, 3) for (avg, min, max)
            values_np = np.array(values)
            if values_np.ndim == 1:
                # Old format: just avg, upgrade to (avg, avg, avg)
                values_np = np.stack([values_np, values_np, values_np], axis=1)

            # Plot only valid (non-NaN) points for each line
            for i, stat in enumerate(["Avg", "Min", "Max"]):
                stat_values = values_np[:, i]
                valid_indices = ~np.isnan(stat_values)
                if np.any(valid_indices):
                    ax.plot(x_axis[valid_indices], stat_values[valid_indices], marker='.', linestyle='-', label=f'{stat}')
            ax.set_ylabel(key)
            # Move legend outside the plot area to the left
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
            ax.grid(True)

        axes[-1].set_xlabel("Generation Number")
        fig.suptitle("Training Diagnostics History", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap

        try:
            plt.savefig(self.plot_save_path)
            print(f"Diagnostics history plot saved to {self.plot_save_path}")
        except Exception as e:
            print(f"Error saving diagnostics history plot: {e}")
        plt.close(fig) # Close the figure to free memory