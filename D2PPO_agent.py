"""
D2PPO: Diffusion Policy Policy Optimization with Dispersive Loss
================================================================
Implementation based on: "D²PPO: Diffusion Policy Policy Optimization with 
Dispersive Loss" (Zou et al., 2025) - arXiv:2508.02644

Two-stage training paradigm:
  Stage 1: Pre-training with diffusion loss + dispersive loss regularization
  Stage 2: PPO fine-tuning with importance-sampled denoising steps

The diffusion policy models the action distribution as an iterative denoising
process (DDPM). Dispersive loss combats representation collapse by treating all
hidden representations within each batch as negative pairs, encouraging the
network to learn discriminative representations of similar observations.
"""

from collections import deque
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, ListStorage
from models.AuxModels import VisionEncoder, ResidualBlock
from models.CriticNetworks import CriticNetwork
from models.DiffusionLSTM import DiffusionLSTM
from utils.utils import to_birds_eye

def dispersive_loss_infonce_l2(features, temperature=0.5):
    """
    InfoNCE-based Dispersive Loss with L2 Distance (Eq. 8 / 19).
    
    L_disp^{L2} = log( 1/(B(B-1)) Σ_i Σ_{j≠i} exp(-||h_i - h_j||^2 / τ) )
    
    Encourages all representations in a batch to be maximally dispersed
    using squared Euclidean distance.
    """
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device)

    # Pairwise squared L2 distances: (B, B)
    diff = features.unsqueeze(0) - features.unsqueeze(1)  # (B, B, D)
    sq_dist = (diff ** 2).sum(dim=-1)                       # (B, B)

    # Mask out diagonal (self-pairs)
    mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
    sq_dist_masked = sq_dist[mask].reshape(B, B - 1)

    # Log-mean-exp for numerical stability
    log_exp = -sq_dist_masked / temperature                 # (B, B-1)
    loss = torch.logsumexp(log_exp.reshape(-1), dim=0) - math.log(B * (B - 1))
    return loss


def dispersive_loss_infonce_cosine(features, temperature=0.5):
    """
    InfoNCE-based Dispersive Loss with Cosine Distance (Eq. 9 / 20).
    
    L_disp^{cos} = log( 1/(B(B-1)) Σ_i Σ_{j≠i} exp(-(1 - cos(h_i, h_j)) / τ) )
    
    Scale-invariant variant focusing on directional diversity.
    """
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device)

    # Normalize features
    features_norm = F.normalize(features, p=2, dim=-1)

    # Cosine similarity matrix: (B, B)
    cos_sim = torch.mm(features_norm, features_norm.t())

    # Cosine dissimilarity
    cos_dissim = 1.0 - cos_sim

    # Mask out diagonal
    mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
    cos_dissim_masked = cos_dissim[mask].reshape(B, B - 1)

    log_exp = -cos_dissim_masked / temperature
    loss = torch.logsumexp(log_exp.reshape(-1), dim=0) - math.log(B * (B - 1))
    return loss


def dispersive_loss_hinge(features, margin=1.0):
    """
    Hinge Loss-based Dispersive Loss (Eq. 10 / 21).
    
    L_disp^{hinge} = 1/(B(B-1)) Σ_i Σ_{j≠i} max(0, ε - ||h_i - h_j||^2)^2
    
    Directly penalizes representations closer than margin threshold.
    """
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device)

    diff = features.unsqueeze(0) - features.unsqueeze(1)
    sq_dist = (diff ** 2).sum(dim=-1)

    mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
    sq_dist_masked = sq_dist[mask].reshape(B, B - 1)

    hinge = F.relu(margin - sq_dist_masked) ** 2
    loss = hinge.mean()
    return loss


DISPERSIVE_LOSS_VARIANTS = {
    "infonce_l2": dispersive_loss_infonce_l2,
    "infonce_cosine": dispersive_loss_infonce_cosine,
    "hinge": dispersive_loss_hinge,
}


class D2PPOAgent:
    """
    D²PPO Agent: Diffusion Policy Policy Optimization with Dispersive Loss.
    
    Follows the same interface as PPOAgent for compatibility with the F1Tenth
    training loop (train.py), but replaces the Gaussian policy with a diffusion
    policy and adds dispersive loss regularization.
    
    Two-stage training:
        1. pretrain_from_demonstrations() – supervised diffusion training + 
           dispersive loss (Stage 1)
        2. learn() – PPO fine-tuning with importance-sampled denoising steps
           (Stage 2)
    """
    def __init__(
        self,
        num_agents,
        map_name,
        steps,
        params,
        transfer=(None, None),
        # Diffusion config
        num_diffusion_steps=50,
        beta_schedule="cosine",
        # Dispersive loss config
        dispersive_variant="infonce_l2",  # "infonce_l2", "infonce_cosine", "hinge"
        dispersive_lambda=0.5,
        dispersive_temperature=0.5,
        dispersive_layer="late",          # "early", "mid", "late", or int/list
        dispersive_hinge_margin=1.0,
        # PPO diffusion config
        ppo_sample_steps=4,               # |S| – number of denoising steps to sample per PPO update
    ):
        # --- Hyperparameters ---
        torch.autograd.set_detect_anomaly(True)
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_actor = 3e-5
        self.lr_critic = 1e-4
        self.gamma = 0.999            # Paper uses 0.999 (Table 6)
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.1       # PPO clip (Table 6)
        self.entropy_coeff = 0.01     # Table 6 uses 0.01
        self.max_grad_norm_actor = 0.5
        self.max_grad_norm_critic = 1.0
        self.state_dim = 4
        self.num_scan_beams = 1080
        self.lidar_fov = 4.7
        self.image_size = 256
        self.minibatch_size = 512
        self.epochs = 10              # Table 6: 10 PPO epochs
        self.epochs_with_demos = 4
        self.bc_epochs = 3
        self.params = params

        # --- Diffusion config ---
        self.num_diffusion_steps = num_diffusion_steps
        self.ppo_sample_steps = ppo_sample_steps

        # --- Dispersive loss config ---
        self.dispersive_variant = dispersive_variant
        self.dispersive_lambda = dispersive_lambda
        self.dispersive_temperature = dispersive_temperature
        self.dispersive_layer = dispersive_layer
        self.dispersive_hinge_margin = dispersive_hinge_margin

        # --- Demonstration Retention ---
        self.demo_buffer = None

        # --- Waypoints for Raceline Reward ---
        self.waypoints_xy, self.waypoints_s, self.raceline_length = self._load_waypoints(map_name)
        self.last_cumulative_distance = np.zeros(self.num_agents)
        self.last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        self.start_s = np.zeros(self.num_agents)
        self.current_lap_count = np.zeros(self.num_agents, dtype=int)
        self.last_checkpoint = np.zeros(self.num_agents, dtype=int)

        # --- Reward Scalars ---
        self.PROGRESS_REWARD_SCALAR = 48.0
        self.LAP_REWARD = 80.0
        self.CHECKPOINT_REWARD = self.LAP_REWARD * 0.1
        self.COLLISION_PENALTY = -4.0
        self.SPEED_REWARD = 3.0
        self.AGENT_COLLISION_PENALTY = -2.0

        # --- Networks ---
        actor_encoder = self._transfer_vision(transfer[0])
        critic_encoder = self._transfer_vision(transfer[1])

        # Diffusion Policy Actor
        self.actor_network = DiffusionLSTM(
            state_dim=self.state_dim,
            action_dim=2,
            encoder=actor_encoder,
            num_diffusion_steps=num_diffusion_steps,
            time_emb_dim=32,
            hidden_dims=(768, 768, 768),
            beta_schedule=beta_schedule,
            odom_expand=64,
            lstm_hidden_size=512,
            lstm_num_layers=2,
            memory_length=350,
            memory_stride=20
        ).to(self.device)

        # Register dispersive hooks for pre-training
        self.actor_network.denoise_net.register_dispersive_hooks(self.dispersive_layer)

        # Critic (same as PPOAgent – not diffusion-based)
        self.critic_network = CriticNetwork(
            state_dim=self.state_dim,
            encoder=critic_encoder,
        ).to(self.device)

        self.actor_network = self._transfer_weights(transfer[0], self.actor_network)
        self.critic_network = self._transfer_weights(transfer[1], self.critic_network)

        # --- Optimizers ---
        self.actor_optimizer = optim.AdamW(
            self.actor_network.parameters(), lr=self.lr_actor, weight_decay=0.01
        )
        self.critic_optimizer = optim.AdamW(
            self.critic_network.parameters(), lr=self.lr_critic, weight_decay=0.01
        )
        self.actor_scaler = torch.amp.GradScaler("cuda")
        self.critic_scaler = torch.amp.GradScaler("cuda")

        # --- Replay Storage ---
        self.buffer = TensorDictReplayBuffer(storage=ListStorage(max_size=steps))

        # --- Diagnostics ---
        self.plot_save_path = "plots/d2ppo_training_diagnostics.png"
        plot_dir = os.path.dirname(self.plot_save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        self.diagnostic_keys = [
            "loss_actor", "loss_critic", "loss_diffusion", "loss_dispersive",
            "entropy", "kl_approx", "clip_fraction", "collisions", "reward",
        ]
        self.diagnostics_history = {key: [] for key in self.diagnostic_keys}
        self.generation_counter = 0

        # --- LSTM temporal state (matches ppo_agent.py pattern) ---
        self.actor_buffer = deque(
            [self.actor_network.create_observation_buffer(self.num_agents, self.device)],
            maxlen=2,
        )
        # Hidden states stored as (hidden_h, hidden_c), each (B, num_layers, H)
        self.actor_hidden = self.actor_network.get_init_hidden(
            self.num_agents, self.device, transpose=True
        )

    # -------------------------------------------------------------------
    # Buffer / utility methods (same interface as PPOAgent)
    # -------------------------------------------------------------------

    def update_buffer_size(self, new_size):
        self.buffer = TensorDictReplayBuffer(storage=ListStorage(max_size=new_size))
        print(f"Updated replay buffer size to {new_size}")

    def _transfer_weights(self, path, network):
        if path is None:
            return network.to(self.device)
        checkpoint = torch.load(path, weights_only=False)
        prefix = "0.module."
        state_dict = {}
        for k, v in checkpoint.items():
            if "log_std_head" in k:
                continue
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = v
            else:
                state_dict[k] = v
        if state_dict:
            net_sd = network.state_dict()
            filtered = {
                k: v for k, v in state_dict.items()
                if k in net_sd and net_sd[k].shape == v.shape
            }
            if filtered:
                network.load_state_dict(filtered, strict=False)
                print(f"Loaded {len(filtered)}/{len(net_sd)} compatible weight tensors.")
            else:
                print("No compatible weights found in checkpoint.")
        return network.to(self.device)

    def _transfer_vision(self, path):
        new_encoder = VisionEncoder(self.num_scan_beams)
        if path is None:
            return new_encoder.to(self.device)
        checkpoint = torch.load(path, weights_only=False)
        prefix = "conv_layers."
        encoder_sd = {}
        for k, v in checkpoint.items():
            if k.startswith(prefix):
                encoder_sd[k[len(prefix):]] = v
            elif k.startswith("0.module." + prefix):
                encoder_sd[k[len("0.module." + prefix):]] = v
        if encoder_sd:
            new_encoder.load_state_dict(encoder_sd)
            print("Loaded pre-trained vision encoder weights.")
        return new_encoder.to(self.device)

    def _load_waypoints(self, map_name):
        waypoint_file = f"maps/{map_name}/{map_name}_raceline.csv"
        waypoints = np.loadtxt(waypoint_file, delimiter=";")
        waypoints_xy = waypoints[:, 1:3]
        positions = waypoints[:, 1:3]
        distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        waypoints_s = np.insert(np.cumsum(distances), 0, 0)
        raceline_length = waypoints_s[-1]
        return waypoints_xy, waypoints_s, raceline_length

    def _obs_to_tensors(self, obs):
        scans = obs["scans"][: self.num_agents]
        scan_tensors = torch.from_numpy(np.array(scans, dtype=np.float64)).float().unsqueeze(1)
        state_data = np.stack(
            (
                obs["linear_vels_x"],
                obs["linear_vels_y"],
                obs["ang_vels_z"],
                obs["linear_accel_x"],
            ),
            axis=1,
        )
        state_tensor = torch.from_numpy(state_data).float()[: self.num_agents]
        return scan_tensors.to(self.device), state_tensor.to(self.device)

    def reset_buffers(self, agent_indices=None):
        """Reset LSTM hidden states and observation buffers."""
        if agent_indices is None:
            # Full reset for all agents
            self.actor_buffer = deque(
                [self.actor_network.create_observation_buffer(self.num_agents, self.device)],
                maxlen=2,
            )
            self.actor_hidden = self.actor_network.get_init_hidden(
                self.num_agents, self.device, transpose=True
            )
        else:
            agent_indices = agent_indices[agent_indices < self.num_agents]
            # Per-agent reset: zero out specific agent slots
            if self.actor_buffer[-1] is not None:
                for idx in agent_indices:
                    self.actor_buffer[-1][idx] = 0.0
            # Zero out hidden states for specific agents
            h, c = self.actor_hidden
            for idx in agent_indices:
                h[idx] = 0.0
                c[idx] = 0.0
            self.actor_hidden = (h, c)

    # -------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------

    def get_action_and_value(self, scan_tensor, state_tensor, deterministic=False, store=True):
        """
        Sample an action from the diffusion policy and compute state value.
        
        Compatible with PPOAgent interface: returns (action, log_prob, value).
        
        When ``store=True`` the LSTM hidden states and observation buffer are
        advanced (used during rollout collection).  When ``store=False`` we
        only need the value estimate (e.g. for bootstrapping next-state value)
        and the temporal state is left untouched.
        """
        self.actor_network.eval()
        self.critic_network.eval()

        with torch.no_grad():
            # Encode observation through CNN + LSTM
            obs_features, new_buffer, new_h, new_c = self.actor_network.encode_observation(
                scan_tensor[: self.num_agents],
                state_tensor[: self.num_agents],
                self.actor_buffer[-1],
                self.actor_hidden[0],
                self.actor_hidden[1],
            )

            # Sample action via full reverse diffusion
            action, chain, log_prob = self.actor_network.sample_action_with_chain(
                obs_features, deterministic=deterministic
            )

            # Value estimate (critic is feedforward — no temporal state)
            value = self.critic_network(
                scan_tensor[: self.num_agents],
                state_tensor[: self.num_agents],
            )

            # Advance temporal state only during rollout collection
            if store:
                self.actor_buffer = new_buffer
                self.actor_hidden = (new_h, new_c)

        return action, log_prob, value

    # -------------------------------------------------------------------
    # Transition storage
    # -------------------------------------------------------------------

    def store_transition(self, obs, next, action, log_prob, reward, done, value):
        next_scans, next_states = self._obs_to_tensors(next)

        _, _, next_value = self.get_action_and_value(
            next_scans, next_states, store=False
        )

        done_tensor = torch.tensor(done, dtype=torch.bool).unsqueeze(-1)

        step_data = TensorDict(
            {
                "observation_scan": obs[0],
                "observation_state": obs[1],
                "action": action,
                "action_log_prob": log_prob,
                "state_value": value,
                "next": TensorDict(
                    {
                        "state_value": next_value,
                        "reward": reward,
                        "done": done_tensor,
                    }
                ),
            },
            batch_size=[self.num_agents],
        )
        self.buffer.add(step_data.to(self.device))

    # -------------------------------------------------------------------
    # GAE computation (identical to PPOAgent)
    # -------------------------------------------------------------------

    def _compute_gae(self, data):
        rewards = data.get(("next", "reward")).to(self.device)
        dones = data.get(("next", "done")).float().to(self.device)
        values = data.get("state_value").to(self.device)
        next_values = data.get(("next", "state_value")).to(self.device)

        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
            dones = dones.squeeze(-1)
            values = values.squeeze(-1)
            next_values = next_values.squeeze(-1)

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
            dones = dones.unsqueeze(-1)
            values = values.unsqueeze(-1)
            next_values = next_values.unsqueeze(-1)

        timesteps = rewards.shape[0]
        num_agents = rewards.shape[1] if rewards.ndim == 2 else 1

        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(values, device=self.device)
        gae = torch.zeros(num_agents, device=self.device)

        for t in reversed(range(timesteps)):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        adv_flat = advantages.flatten()
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        data.set("advantage", advantages)
        data.set("value_target", returns)
        return data

    # -------------------------------------------------------------------
    # Reward calculation (same as PPOAgent)
    # -------------------------------------------------------------------

    def _project_to_raceline(self, current_pos, start_idx, lookahead):
        wp_count = len(self.waypoints_xy)
        search_indices = np.arange(start_idx, start_idx + lookahead) % wp_count
        search_waypoints = self.waypoints_xy[search_indices]
        
        distances_in_window = np.linalg.norm(search_waypoints - current_pos, axis=1)
        closest_wp_in_window = np.argmin(distances_in_window)
        closest_wp_index_global = search_indices[closest_wp_in_window]
        
        W_curr = self.waypoints_xy[closest_wp_index_global]
        W_prev_index = (closest_wp_index_global - 1 + wp_count) % wp_count
        W_prev = self.waypoints_xy[W_prev_index]
        
        V = W_curr - W_prev
        V_len_sq = np.dot(V, V)
        W = current_pos - W_prev
        L = np.dot(W, V) / V_len_sq if V_len_sq > 1e-6 else 0.0
        
        s_prev = self.waypoints_s[W_prev_index]
        s_curr = self.waypoints_s[closest_wp_index_global]
        
        segment_distance = s_curr - s_prev
        if segment_distance < 0:
            segment_distance += self.raceline_length
        projected_s = s_prev + L * segment_distance
        
        return projected_s, closest_wp_index_global

    def calculate_reward(self, next_obs):
        collisions = np.array(next_obs["collisions"][: self.num_agents])
        speeds = np.array(next_obs["linear_vels_x"][: self.num_agents])
        wall_collisions = collisions == 1
        agent_collisions = collisions == 2
        rewards = np.zeros(self.num_agents)

        target_speed = 9.0
        speed_bonus = np.clip(speeds - target_speed, 0, 3.0) * self.SPEED_REWARD
        rewards += speed_bonus
        rewards += wall_collisions * self.COLLISION_PENALTY
        rewards += agent_collisions * (self.AGENT_COLLISION_PENALTY * 0.5)

        rewards_tensor = torch.from_numpy(rewards.astype(np.float32)).unsqueeze(-1)
        avg_reward = rewards.mean()
        return rewards_tensor, avg_reward

    def reset_progress_trackers(self, initial_poses_xy, agent_idxs=None):
        if agent_idxs is not None:
            agent_idxs = agent_idxs[agent_idxs < self.num_agents]
            for i in agent_idxs:
                current_pos = initial_poses_xy[i]
                distances = np.linalg.norm(self.waypoints_xy - current_pos, axis=1)
                closest = np.argmin(distances)
                self.last_cumulative_distance[i] = self.waypoints_s[closest]
                self.last_wp_index[i] = closest
                self.start_s[i] = self.waypoints_s[closest]
                self.current_lap_count[i] = 0
                self.last_checkpoint[i] = 0
            return

        for i in range(self.num_agents):
            current_pos = initial_poses_xy[i]
            distances = np.linalg.norm(self.waypoints_xy - current_pos, axis=1)
            closest = np.argmin(distances)
            self.last_cumulative_distance[i] = self.waypoints_s[closest]
            self.last_wp_index[i] = closest
            self.start_s[i] = self.waypoints_s[closest]
            self.current_lap_count[i] = 0
            self.last_checkpoint[i] = 0

    # ===================================================================
    # STAGE 1: Pre-training with Dispersive Loss
    # ===================================================================

    def _compute_dispersive_loss(self, features_dict):
        """
        Compute dispersive loss averaged over hooked layers and denoising
        timesteps (Eq. 7).
        
        L_disp = (1/K) Σ_{k=1}^{K} L_disp_variant(H_k)
        
        In practice, features_dict maps layer_idx → features from a single
        forward pass at one denoising timestep. We call this function once per
        sampled timestep and average outside.
        """
        loss_fn = DISPERSIVE_LOSS_VARIANTS[self.dispersive_variant]
        total = torch.tensor(0.0, device=self.device)
        count = 0

        for layer_idx, features in features_dict.items():
            # Global average pooling if features are high-dimensional
            if features.ndim > 2:
                features = features.mean(dim=list(range(1, features.ndim - 1)))

            if self.dispersive_variant == "hinge":
                total = total + loss_fn(features, margin=self.dispersive_hinge_margin)
            else:
                total = total + loss_fn(features, temperature=self.dispersive_temperature)
            count += 1

        return total / max(count, 1)

    def pretrain_from_demonstrations(self, demo_buffer=None, epochs=100, gradient_accumulation_steps=4, bc_weights=None):
        """
        Stage 1: Pre-training with diffusion loss + dispersive loss.
        
        L_{D2PPO}^{pre-train} = L_diff + λ · L_disp   (Eq. 6)
        
        Uses expert demonstrations to train the diffusion policy's noise
        prediction network while encouraging representational diversity via
        dispersive loss on intermediate MLP features.
        """
        if demo_buffer is not None:
            self.demo_buffer = demo_buffer
            print(f"\n[D²PPO Stage 1] Pre-training from {len(self.demo_buffer)} demonstrations...")
            print(f"  Dispersive variant: {self.dispersive_variant}, λ={self.dispersive_lambda}")
            print(f"  Layer hook: {self.dispersive_layer}, τ={self.dispersive_temperature}")
            bc_weights = (1.0, 1.0)
        elif demo_buffer is None and self.demo_buffer is not None:
            demo_buffer = self.demo_buffer
            print(f"\n  [D²PPO] BC regularisation from stored {len(self.demo_buffer)} demos...")
        else:
            print("[D²PPO] No demonstrations available for pre-training.")
            return

        # Ensure hooks are registered
        self.actor_network.denoise_net.register_dispersive_hooks(self.dispersive_layer)

        self.actor_network.train()
        total_diff_loss = 0.0
        total_disp_loss = 0.0

        for epoch in range(epochs):
            epoch_diff = 0.0
            epoch_disp = 0.0
            self.actor_optimizer.zero_grad()
            update_counter = 0

            # Reset temporal state at the start of each epoch
            demo_buffer_obs = None
            demo_hidden = (None, None)

            for i, d in enumerate(demo_buffer):
                # Prepare single-sample batch
                scan = torch.from_numpy(d["scan"]).float().unsqueeze(0).to(self.device)
                state = torch.from_numpy(d["state"]).float().unsqueeze(0).to(self.device)
                action = torch.from_numpy(d["action"]).float().unsqueeze(0).to(self.device)

                # Encode observation (track LSTM state across sequential demos)
                obs_features, demo_buffer_obs, demo_h, demo_c = self.actor_network.encode_observation(
                    scan, state, demo_buffer_obs, demo_hidden[0], demo_hidden[1]
                )

                # Detach temporal state to prevent backprop through entire sequence
                demo_buffer_obs = demo_buffer_obs.detach()
                demo_hidden = (demo_h.detach(), demo_c.detach())

                # --- Diffusion loss ---
                diff_loss = self.actor_network.compute_diffusion_loss(action, obs_features)

                # --- Dispersive loss (on intermediate features from denoise forward pass) ---
                # The diffusion_loss call already triggered the denoise_net forward pass;
                # however, we also need per-timestep features. We do an extra explicit pass:
                B = action.shape[0]
                t_rand = torch.randint(0, self.num_diffusion_steps, (B,), device=self.device)
                noise = torch.randn_like(action)
                noisy_action = self.actor_network.q_sample(action, t_rand, noise=noise)
                _ = self.actor_network.denoise_net(noisy_action, obs_features, t_rand)
                feat_dict = self.actor_network.denoise_net.get_intermediate_features()
                disp_loss = self._compute_dispersive_loss(feat_dict)

                # Combined loss (Eq. 6)
                loss = (diff_loss + self.dispersive_lambda * disp_loss) / gradient_accumulation_steps
                loss.backward()

                epoch_diff += diff_loss.item()
                epoch_disp += disp_loss.item()

                update_counter += 1
                if update_counter >= gradient_accumulation_steps:
                    nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm_actor)
                    self.actor_optimizer.step()
                    self.actor_optimizer.zero_grad()
                    update_counter = 0

                if (i + 1) % 100 == 0:
                    progress = (i + 1) / len(demo_buffer) * 100
                    print(
                        f"    Epoch {epoch+1}/{epochs} [{progress:.0f}%] "
                        f"Diff: {epoch_diff/(i+1):.4f}  Disp: {epoch_disp/(i+1):.4f}",
                        end="\r",
                    )

            # Final gradient step
            if update_counter > 0:
                nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm_actor)
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()

            n = len(demo_buffer)
            avg_diff = epoch_diff / n
            avg_disp = epoch_disp / n
            total_diff_loss += avg_diff
            total_disp_loss += avg_disp
            print(
                f"    Epoch {epoch+1}/{epochs}: Diff Loss={avg_diff:.4f}, "
                f"Disp Loss={avg_disp:.4f}, Combined={avg_diff + self.dispersive_lambda*avg_disp:.4f}"
            )

        self.buffer.empty()
        print(
            f"[D²PPO Stage 1] Complete. Avg Diff Loss: {total_diff_loss/epochs:.4f}, "
            f"Avg Disp Loss: {total_disp_loss/epochs:.4f}\n"
        )

    # ===================================================================
    # STAGE 2: PPO Fine-tuning for Diffusion Policies
    # ===================================================================

    def learn(self, collisions, reward):
        """
        Stage 2: PPO fine-tuning with importance-sampled denoising steps.
        
        Adapts the standard PPO clipped objective for diffusion policies
        by computing probability ratios at sampled denoising timesteps and
        accumulating gradients via the chain rule (Eq. 13-14, 34).
        
        L_{D2PPO}(θ) = E_t[ Σ_{k∈S} (K / (|S|·p(k))) 
                         min(r_t^(k)(θ) Â_t^(k), clip(r_t^(k)(θ), 1-ε, 1+ε) Â_t^(k)) ]
        """
        print("[D²PPO Stage 2] Starting PPO fine-tuning...")
        print(f"  Buffer size: {len(self.buffer)}")

        data = self.buffer.sample(batch_size=len(self.buffer))

        current_gen_diagnostics = {key: [] for key in self.diagnostic_keys}
        current_gen_diagnostics["collisions"] = [collisions]
        current_gen_diagnostics["reward"] = [reward]

        # Compute GAE
        with torch.no_grad():
            data = self._compute_gae(data)

        obs_scan_all = data["observation_scan"]
        obs_state_all = data["observation_state"]
        actions_all = data["action"]
        old_log_probs_all = data["action_log_prob"]
        advantages_all = data["advantage"]
        value_targets_all = data["value_target"]

        num_timesteps = len(data)

        self.actor_network.train()
        self.critic_network.train()

        K = self.num_diffusion_steps
        S_size = self.ppo_sample_steps

        print(f"  Training: {self.epochs} epochs, {num_timesteps} timesteps, "
              f"sampling {S_size}/{K} denoising steps per update")

        for epoch in range(self.epochs):
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_diff_loss = 0.0
            epoch_disp_loss = 0.0
            num_updates = 0

            # Shuffle timestep indices for minibatch training
            indices = torch.randperm(num_timesteps)

            for mb_start in range(0, num_timesteps, self.minibatch_size):
                mb_end = min(mb_start + self.minibatch_size, num_timesteps)
                mb_idx = indices[mb_start:mb_end]

                obs_scan = obs_scan_all[mb_idx]
                obs_state = obs_state_all[mb_idx]
                actions = actions_all[mb_idx]
                old_lp = old_log_probs_all[mb_idx]
                advantages = advantages_all[mb_idx]
                value_targets = value_targets_all[mb_idx]

                # Flatten agents dimension if present: [T, A, ...] → [T*A, ...]
                if obs_scan.ndim == 4:  # [T, A, 1, beams]
                    T, A = obs_scan.shape[:2]
                    obs_scan = obs_scan.reshape(T * A, *obs_scan.shape[2:])
                    obs_state = obs_state.reshape(T * A, *obs_state.shape[2:])
                    actions = actions.reshape(T * A, *actions.shape[2:])
                    old_lp = old_lp.reshape(T * A)
                    advantages = advantages.reshape(T * A)
                    value_targets = value_targets.reshape(T * A)
                elif obs_scan.ndim == 3 and old_lp.ndim == 2:
                    # [T, A, beams] but scan has channel dim
                    T, A = old_lp.shape
                    obs_scan = obs_scan.reshape(T * A, *obs_scan.shape[2:]) if obs_scan.shape[0] == T else obs_scan
                    obs_state = obs_state.reshape(T * A, -1) if obs_state.shape[0] == T else obs_state
                    actions = actions.reshape(T * A, -1)
                    old_lp = old_lp.reshape(T * A)
                    advantages = advantages.reshape(T * A)
                    value_targets = value_targets.reshape(T * A)

                B = obs_scan.shape[0]

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # --- Encode observations (no temporal continuity in shuffled minibatches) ---
                obs_features, _, _, _ = self.actor_network.encode_observation(
                    obs_scan, obs_state, obs_buffer=None,
                    hidden_h=None, hidden_c=None,
                )

                # --- Critic value prediction ---
                predicted_values = self.critic_network(obs_scan, obs_state)
                if predicted_values.ndim > 1:
                    predicted_values = predicted_values.squeeze(-1)
                critic_loss = F.mse_loss(predicted_values, value_targets.detach())

                # --- Actor: importance-sampled denoising-step PPO ---
                # Sample a subset S of denoising timesteps (Eq. 14)
                sampled_k = torch.randint(1, K + 1, (S_size,), device=self.device)
                # Uniform sampling probability: p(k) = 1/K
                importance_weight = K / S_size  # K / (|S| * p(k)) with p(k)=1/K

                total_policy_loss = torch.tensor(0.0, device=self.device)

                for k_val in sampled_k:
                    k = k_val.item()
                    t = torch.full((B,), k, device=self.device, dtype=torch.long)
                    t_prev = t - 1  # k-1

                    # Forward diffusion to get a^k from clean action a^0
                    noise = torch.randn_like(actions)
                    a_k = self.actor_network.q_sample(actions, t, noise=noise)

                    # Use DDPM posterior to get a^{k-1} (the "observed" previous step)
                    # a^{k-1} = q(a^{k-1}|a^0, a^k) — this is the target of the denoising step
                    if k > 1:
                        a_k_minus_1 = self.actor_network.q_sample(actions, t_prev, noise=noise)
                    else:
                        a_k_minus_1 = actions  # a^0

                    # Current policy log-prob: log p_θ(a^{k-1}|a^k, s)
                    new_log_prob_k = self.actor_network.compute_denoising_log_prob(
                        a_k_minus_1, a_k, obs_features, t
                    )

                    # Old policy log-prob (from stored old_lp, uniformly distributed)
                    # Each stored log_prob covers the full chain; divide by K for per-step
                    old_log_prob_k = old_lp / K

                    # Probability ratio r_t^(k) (Eq. 35-36)
                    log_ratio = new_log_prob_k - old_log_prob_k
                    log_ratio = torch.clamp(log_ratio, -5.0, 5.0)  # Stability
                    ratio = torch.exp(log_ratio)

                    # PPO clipped surrogate (Eq. 34)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    ) * advantages
                    step_loss = -torch.min(surr1, surr2).mean()

                    total_policy_loss = total_policy_loss + importance_weight * step_loss

                # Average over sampled steps
                policy_loss = total_policy_loss / S_size

                # --- Entropy bonus (approximate via diffusion noise level) ---
                # For diffusion policies, entropy is implicitly controlled by the
                # stochasticity of the denoising process. We add a small regularizer
                # based on the predicted noise magnitude diversity.
                with torch.no_grad():
                    t_entropy = torch.randint(0, K, (B,), device=self.device)
                    noise_ent = torch.randn_like(actions)
                    noisy_ent = self.actor_network.q_sample(actions, t_entropy, noise=noise_ent)
                pred_noise_ent = self.actor_network.predict_noise(noisy_ent, obs_features, t_entropy)
                entropy_proxy = pred_noise_ent.var(dim=0).mean()  # Higher variance → more diverse predictions
                entropy_loss = -self.entropy_coeff * entropy_proxy

                actor_loss = policy_loss + entropy_loss

                # --- Backward & update ---
                self.actor_scaler.scale(actor_loss).backward()
                self.actor_scaler.unscale_(self.actor_optimizer)
                nn.utils.clip_grad_norm_(
                    self.actor_network.parameters(), self.max_grad_norm_actor
                )
                self.actor_scaler.step(self.actor_optimizer)
                self.actor_scaler.update()

                self.critic_scaler.scale(critic_loss).backward()
                self.critic_scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    self.critic_network.parameters(), self.max_grad_norm_critic
                )
                self.critic_scaler.step(self.critic_optimizer)
                self.critic_scaler.update()

                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                num_updates += 1

                # Diagnostics
                with torch.no_grad():
                    kl_approx = (
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).mean().item()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean().item()
                    )

                current_gen_diagnostics["loss_actor"].append(actor_loss.item())
                current_gen_diagnostics["loss_critic"].append(critic_loss.item())
                current_gen_diagnostics["loss_diffusion"].append(0.0)  # Not used in Stage 2
                current_gen_diagnostics["loss_dispersive"].append(0.0)
                current_gen_diagnostics["entropy"].append(entropy_proxy.item())
                current_gen_diagnostics["kl_approx"].append(kl_approx)
                current_gen_diagnostics["clip_fraction"].append(clip_frac)

            if num_updates > 0:
                print(
                    f"  Epoch {epoch+1}/{self.epochs}: "
                    f"Actor={epoch_actor_loss/num_updates:.4f}, "
                    f"Critic={epoch_critic_loss/num_updates:.4f}"
                )

            torch.cuda.empty_cache()

        # --- Post-training bookkeeping ---
        self.generation_counter += 1
        for key in self.diagnostic_keys:
            values = current_gen_diagnostics.get(key)
            if values:
                avg_val = np.mean(values)
                min_val = np.min(values)
                max_val = np.max(values)
                self.diagnostics_history[key].append((avg_val, min_val, max_val))

        if self.generation_counter > 0:
            self._plot_historical_diagnostics()

        self.buffer.empty()
        del data
        torch.cuda.empty_cache()
        print("[D²PPO Stage 2] Learning complete.")

    # -------------------------------------------------------------------
    # Diagnostics plotting
    # -------------------------------------------------------------------

    def _plot_historical_diagnostics(self):
        keys_to_plot = [
            k for k in self.diagnostic_keys
            if k in self.diagnostics_history and self.diagnostics_history[k]
        ]
        num_metrics = len(keys_to_plot)
        if num_metrics == 0 or self.generation_counter == 0:
            return

        plt.style.use("dark_background")
        fig, axes = plt.subplots(num_metrics, 1, figsize=(25, 5 * num_metrics), sharex=True)
        if num_metrics == 1:
            axes = [axes]
        plt.rcParams["font.size"] = 24
        plt.rcParams["lines.linewidth"] = 3

        x_axis = np.arange(1, self.generation_counter + 1)

        for idx, key in enumerate(keys_to_plot):
            values = self.diagnostics_history.get(key, [])
            ax = axes[idx]
            if not values:
                ax.set_ylabel(key)
                ax.grid(True)
                continue
            values_np = np.array(values)
            if values_np.ndim == 1:
                values_np = np.stack([values_np, values_np, values_np], axis=1)
            for i, stat in enumerate(["Avg", "Min", "Max"]):
                stat_values = values_np[:, i]
                valid = ~np.isnan(stat_values)
                if np.any(valid):
                    ax.plot(x_axis[valid], stat_values[valid], marker=".", linestyle="-", label=f"{stat}")
            ax.set_ylabel(key)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            ax.grid(True)

        axes[-1].set_xlabel("Generation")
        fig.suptitle("D²PPO Training Diagnostics", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        try:
            plt.savefig(self.plot_save_path)
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close(fig)
