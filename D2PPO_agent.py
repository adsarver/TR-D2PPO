"""
D2PPO: Diffusion Policy Policy Optimization — Pure RL
=====================================================
Implementation based on: "D²PPO: Diffusion Policy Policy Optimization with
Dispersive Loss" (Zou et al., 2025) - arXiv:2508.02644

The diffusion policy models the action distribution as an iterative denoising
process (DDPM).  This agent uses **only** the PPO (Stage 2) objective — no
supervised / BC pre-training or dispersive loss.
"""

from collections import deque
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


class D2PPOAgent:
    """
    D²PPO Agent: Diffusion Policy Policy Optimization.
    
    Follows the same interface as PPOAgent for compatibility with the F1Tenth
    training loop (train.py), but replaces the Gaussian policy with a diffusion
    policy.  Pure RL — no supervised / BC pre-training.
    """
    def __init__(
        self,
        num_agents,
        map_name,
        steps,
        params,
        transfer=(None, None),
        # Diffusion config
        num_diffusion_steps=25,
        beta_schedule="cosine",
        # PPO diffusion config
        ppo_sample_steps=2,               # |S| – number of denoising steps to sample per PPO update
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
        self.max_grad_norm_actor = 0.5
        self.max_grad_norm_critic = 1.0
        self.state_dim = 4
        self.num_scan_beams = 1080
        self.lidar_fov = 4.7
        self.image_size = 256
        self.minibatch_size = 512
        self.epochs = 10              # Table 6: 10 PPO epochs
        self.params = params

        # --- Diffusion config ---
        self.num_diffusion_steps = num_diffusion_steps
        self.ppo_sample_steps = ppo_sample_steps

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
            hidden_dims=(512, 512, 512),
            beta_schedule=beta_schedule,
            odom_expand=64,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            memory_length=350,
            memory_stride=20
        ).to(self.device)

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
            "loss_actor", "loss_critic",
            "kl_approx", "clip_fraction", "collisions", "reward",
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
        
    def update_buffer_size(self, new_size):
        self.buffer = TensorDictReplayBuffer(storage=ListStorage(max_size=new_size))
        print(f"Updated replay buffer size to {new_size}")

    def _transfer_weights(self, path, network):
        if path is None:
            return network.to(self.device)
        if not os.path.exists(path):
            print(f"Warning: checkpoint '{path}' not found — using random init.")
            return network.to(self.device)

        checkpoint = torch.load(path, weights_only=False)

        # Accept both raw state_dict (OrderedDict) and wrapped formats
        if isinstance(checkpoint, dict):
            state_dict_raw = checkpoint
        elif isinstance(checkpoint, list):
            print(f"Warning: '{path}' is a list (demos?), not a state_dict — skipping.")
            return network.to(self.device)
        else:
            print(f"Warning: '{path}' has unexpected type {type(checkpoint).__name__} — skipping.")
            return network.to(self.device)

        prefix = "0.module."
        state_dict = {}
        for k, v in state_dict_raw.items():
            if not isinstance(v, torch.Tensor):
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
                print(f"Loaded {len(filtered)}/{len(net_sd)} compatible weight tensors from '{path}'.")
            else:
                print(f"No compatible weights found in '{path}'.")
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
        # Robustly handle several possible shapes returned by the environment:
        # - (num_agents, scan_len)
        # - (num_timesteps, num_agents, scan_len) (pick last timestep)
        scans_arr = np.array(obs["scans"])
        if scans_arr.ndim >= 3 and scans_arr.shape[0] != self.num_agents and scans_arr.shape[1] == self.num_agents:
            # e.g., (T, N, scan_len) -> take last timestep
            scans_arr = scans_arr[-1]
        scans_arr = scans_arr[: self.num_agents]
        scan_tensors = torch.from_numpy(scans_arr.astype(np.float32)).unsqueeze(1)

        # State fields may also be time-major (T, N). Handle similarly.
        def last_or_first(arr):
            a = np.array(arr)
            if a.ndim > 1 and a.shape[0] != self.num_agents and a.shape[1] == self.num_agents:
                return a[-1]
            return a

        lvx = last_or_first(obs["linear_vels_x"])[: self.num_agents]
        lvy = last_or_first(obs["linear_vels_y"])[: self.num_agents]
        avz = last_or_first(obs["ang_vels_z"])[: self.num_agents]
        lax = last_or_first(obs["linear_accel_x"])[: self.num_agents]

        state_data = np.stack((lvx, lvy, avz, lax), axis=1)
        state_tensor = torch.from_numpy(state_data.astype(np.float32))

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
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
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
                self.actor_buffer.append(new_buffer)
                self.actor_hidden = (new_h, new_c)

        return action, log_prob, value


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

    def learn(self, collisions, reward):
        print("Starting PPO learning...")
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
                # Valid denoising transitions are t=1→0, t=2→1, ..., t=K-1→K-2
                # so t ∈ {1, ..., K-1} (schedule indices are 0-based, 0 to K-1)
                sampled_k = torch.randint(1, K, (S_size,), device=self.device)
                # Uniform sampling probability: p(k) = 1/(K-1)
                importance_weight = (K - 1) / S_size

                total_policy_loss = torch.tensor(0.0, device=self.device)

                # Normalise stored raw actions into [-1,1] for diffusion
                actions_norm = self.actor_network.normalize_action(actions)

                for k_val in sampled_k:
                    k = k_val.item()
                    t = torch.full((B,), k, device=self.device, dtype=torch.long)
                    t_prev = t - 1  # k-1

                    # Forward diffusion to get a^k from clean action a^0
                    noise = torch.randn_like(actions_norm)
                    a_k = self.actor_network.q_sample(actions_norm, t, noise=noise)

                    # Use DDPM posterior to get a^{k-1} (the "observed" previous step)
                    # a^{k-1} = q(a^{k-1}|a^0, a^k) — this is the target of the denoising step
                    if k > 1:
                        a_k_minus_1 = self.actor_network.q_sample(actions_norm, t_prev, noise=noise)
                    else:
                        a_k_minus_1 = actions_norm  # a^0

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

                # Diffusion policies control exploration implicitly through
                # stochastic denoising — no explicit entropy bonus needed.
                actor_loss = policy_loss

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
