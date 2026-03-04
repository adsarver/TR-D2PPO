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
from baselines.gap_follow_pure_pursuit import GapFollowPurePursuit
from models.AuxModels import VisionEncoder
from models.CriticNetworks import CriticNetwork
from models.DiffusionLSTM import DiffusionLSTM


# ── Dispersive loss (InfoNCE-L2) ─────────────────────────────────────
def dispersive_loss_infonce_l2(features, temperature=0.5):
    """InfoNCE-style contrastive loss on L2-normalised features.

    Encourages intermediate denoiser representations to spread on the unit
    hyper-sphere, preventing mode collapse during RL fine-tuning.
    """
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device)
    features = F.normalize(features, dim=-1)
    diff = features.unsqueeze(0) - features.unsqueeze(1)
    sq_dist = (diff ** 2).sum(dim=-1)
    mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
    sq_dist_masked = sq_dist[mask].reshape(B, B - 1)
    log_exp = -sq_dist_masked / temperature
    loss = torch.logsumexp(log_exp.reshape(-1), dim=0) - math.log(B * (B - 1))
    return loss


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
        num_diffusion_steps=10,
        beta_schedule="cosine",
    ):
        # --- Hyperparameters ---
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_actor = 3e-5
        self.lr_critic = 1e-4
        self.gamma = 0.999            # Paper uses 0.999 (Table 6)
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2       # Value-clipping range for critic
        self.max_grad_norm_actor = 0.5
        self.max_grad_norm_critic = 1.0
        self.state_dim = 4
        self.num_scan_beams = 1080
        self.lidar_fov = 4.7
        self.image_size = 256
        self.minibatch_size = 128
        self.epochs = 10 
        self.params = params
        self.gfpp = GapFollowPurePursuit(
            map_name=map_name,
            wheelbase=params['lf'] + params['lr'],
            max_steering=params['s_max'],
            max_speed=8.0,
            min_speed=1.0,
            num_beams=self.num_scan_beams,
            fov=self.lidar_fov,
        )

        # --- Advantage-weighted diffusion config ---
        self.num_diffusion_steps = num_diffusion_steps
        self.awr_temperature = 1.0    # β for exp(A/β) advantage weighting
        self.awr_max_weight = 20.0    # Clamp max weight for stability
        self.diff_reg_lambda = 1.0    # Unweighted diffusion MSE regulariser
        self.dispersive_lambda = 0.1  # Dispersive loss weight (reduced for Stage-2 RL)
        self.dispersive_temperature = 0.5

        # --- Waypoints for Raceline Reward ---
        self.waypoints_xy, self.waypoints_s, self.raceline_length = self._load_waypoints(map_name)
        self.last_cumulative_distance = np.zeros(self.num_agents)
        self.last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        self.start_s = np.zeros(self.num_agents)
        self.current_lap_count = np.zeros(self.num_agents, dtype=int)
        self.last_checkpoint = np.zeros(self.num_agents, dtype=int)

        # --- Reward Scalars ---
        self.LAP_REWARD = 80.0
        self.CHECKPOINT_REWARD = self.LAP_REWARD * 0.1
        self.COLLISION_PENALTY = -4.0
        self.SPEED_REWARD = 5.0
        self.AGENT_COLLISION_PENALTY = -2.0
        self.NUM_CHECKPOINTS = 10
        self._prev_lap_counts = np.zeros(self.num_agents, dtype=int)

        # --- Networks ---
        actor_encoder = self._transfer_vision(transfer[0])
        critic_encoder = self._transfer_vision(transfer[0])

        # Diffusion Policy Actor
        self.actor_network = DiffusionLSTM(
            state_dim=self.state_dim,
            action_dim=2,
            encoder=actor_encoder,
            num_diffusion_steps=num_diffusion_steps,
            inference_steps=5,          # DDIM fast sampling for rollout/deploy
            time_emb_dim=32,
            hidden_dims=(256, 256),
            beta_schedule=beta_schedule,
            odom_expand=32,
            proj_hidden=384,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            memory_length=64,
            memory_stride=100
        ).to(self.device)

        # Register dispersive hooks on last denoiser block (same as pretraining)
        self.actor_network.denoise_net.register_dispersive_hooks("late")

        # Critic with LSTM temporal backbone (mirrors actor architecture)
        self.critic_network = CriticNetwork(
            state_dim=self.state_dim,
            encoder=critic_encoder,
            lstm_hidden_size=64,
            lstm_num_layers=2,
            memory_length=48,
            memory_stride=100,
            odom_expand=64,
            proj_hidden=256,
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
        self._last_obs_features = None  # Cached by get_action_and_value for store_transition

        # --- Diagnostics ---
        self.plot_save_path = "plots/d2ppo_training_diagnostics.png"
        plot_dir = os.path.dirname(self.plot_save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        self.diagnostic_keys = [
            "loss_actor", "loss_critic", "loss_diffusion",
            "loss_dispersive", "adv_weight_std", "collisions", "reward", "avg_speed"
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

        # --- Critic LSTM temporal state ---
        self.critic_buffer = deque(
            [self.critic_network.create_observation_buffer(self.num_agents, self.device)],
            maxlen=2,
        )
        self.critic_hidden = self.critic_network.get_init_hidden(
            self.num_agents, self.device, transpose=True
        )

        # For computing acceleration from consecutive velocity observations
        # (the sim's linear_accel_x is always 0)
        self._prev_vels_x = np.zeros(self.num_agents)
        self._sim_dt = 0.01  # f110_gym default timestep
        
    def update_buffer_size(self, new_size):
        self.buffer = TensorDictReplayBuffer(storage=ListStorage(max_size=new_size))
        print(f"Updated replay buffer size to {new_size}")

    # ------------------------------------------------------------------
    # Critic pretraining  (run once before RL loop)
    # ------------------------------------------------------------------
    def pretrain_critic(self, env, pp_driver, num_agents_total, maps,
                        rollout_steps=512, num_rollouts=3, epochs=10,
                        lr=5e-4, batch_size=256):
        """Pre-train the critic on MC returns collected by the pretrained actor.

        Runs the frozen actor for *num_rollouts* episodes on a diverse set of
        *maps*, computes discounted Monte-Carlo returns, then trains the
        critic's LSTM + value head via supervised MSE regression.  The vision
        encoder is frozen (already pretrained) so only the temporal and value
        layers are fitted.

        Should be called **once** right after agent construction, before the
        main RL loop begins.
        """
        from utils.utils import generate_start_poses, get_map_dir
        from baselines.pure_pursuit import PurePursuit

        print("\n" + "=" * 60)
        print("  CRITIC PRE-TRAINING")
        print("=" * 60)

        # Freeze the vision encoder — only train LSTM + projection + value head
        for p in self.critic_network.conv_layers.parameters():
            p.requires_grad = False

        pretrain_optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.critic_network.parameters()),
            lr=lr, weight_decay=0.01,
        )

        all_scans = []
        all_states = []
        all_returns = []

        self.actor_network.eval()
        self.critic_network.train()

        for rollout_idx in range(num_rollouts):
            # Pick a map (cycle through the list)
            map_name = maps[rollout_idx % len(maps)]
            print(f"  Rollout {rollout_idx + 1}/{num_rollouts} on {map_name}")

            # Reconfigure env and waypoints
            env.update_map(get_map_dir(map_name) + f"/{map_name}_map", ".png")
            wp_xy, wp_s, rl = self._load_waypoints(map_name)
            self.waypoints_xy, self.waypoints_s, self.raceline_length = wp_xy, wp_s, rl
            self.last_cumulative_distance[:] = 0
            self.last_wp_index[:] = 0
            self.last_checkpoint[:] = 0
            self._prev_lap_counts[:] = 0

            pp_driver.update_map(map_name)

            poses = generate_start_poses(map_name, num_agents_total)
            obs, _, _, _ = env.reset(poses=poses)
            self.reset_progress_trackers(initial_poses_xy=poses[:, :2])
            self.reset_buffers()
            self._prev_vels_x[:] = 0

            collision_timers = np.zeros(num_agents_total, dtype=np.int32)
            rollout_scans = []
            rollout_states = []
            rollout_rewards = []

            for step in range(rollout_steps):
                scan_t, state_t = self._obs_to_tensors(obs)
                with torch.no_grad():
                    action, _, _ = self.get_action_and_value(scan_t, state_t, store=True)

                action_np = action.cpu().numpy()
                if action_np.shape[0] < num_agents_total:
                    pp_act = pp_driver.get_actions_batch(obs).astype(np.float32)
                    action_np = np.vstack((action_np, pp_act[action_np.shape[0]:]))

                next_obs, _, _, _ = env.step(action_np)
                rew_t, _ = self.calculate_reward(next_obs)

                # Handle stuck agents
                cols = np.array(next_obs['collisions'][:num_agents_total])
                vels = np.array(next_obs['linear_vels_x'][:num_agents_total])
                collision_timers[(cols == 1) | ((vels < 0.1) & (vels > -0.1))] += 1
                collision_timers[cols == 0] = 0
                stuck = np.where(collision_timers >= 32)[0]
                if len(stuck) > 0:
                    cur_poses = np.stack([next_obs['poses_x'], next_obs['poses_y'],
                                          next_obs['poses_theta']], axis=1)
                    new_poses = generate_start_poses(map_name, num_agents_total, agent_poses=cur_poses)
                    next_obs, _, _, _ = env.reset(poses=new_poses, agent_idxs=stuck)
                    self.reset_buffers(stuck)
                    self.reset_progress_trackers(initial_poses_xy=new_poses[:, :2], agent_idxs=stuck)
                    collision_timers[stuck] = 0

                # Store per-agent data (scans/states only for AI agents)
                rollout_scans.append(scan_t[:self.num_agents].cpu())
                rollout_states.append(state_t[:self.num_agents].cpu())
                rollout_rewards.append(rew_t[:self.num_agents].cpu())  # (A, 1)

                obs = next_obs
                if (step + 1) % 100 == 0:
                    print(f"    step {step + 1}/{rollout_steps}", end='\r')

            print()

            # Compute discounted MC returns  (reverse cumsum)
            T = len(rollout_rewards)
            rewards_arr = torch.stack(rollout_rewards).squeeze(-1)  # (T, A)
            # Per-generation normalisation
            r_std = rewards_arr.std().clamp(min=1e-4)
            rewards_arr = rewards_arr / r_std

            returns = torch.zeros_like(rewards_arr)
            G = torch.zeros(self.num_agents)
            for t in reversed(range(T)):
                G = rewards_arr[t] + self.gamma * G
                returns[t] = G

            # Flatten time × agents → samples
            scans_flat = torch.cat(rollout_scans, dim=0)     # (T*A, 1, beams)
            states_flat = torch.cat(rollout_states, dim=0)    # (T*A, state_dim)
            returns_flat = returns.reshape(-1)                # (T*A,)

            all_scans.append(scans_flat)
            all_states.append(states_flat)
            all_returns.append(returns_flat)

        # --- Combine all rollout data ---
        X_scans = torch.cat(all_scans, dim=0).to(self.device)
        X_states = torch.cat(all_states, dim=0).to(self.device)
        Y_returns = torch.cat(all_returns, dim=0).to(self.device)

        N = X_scans.shape[0]
        print(f"  Collected {N} samples for critic pretraining."
              f"  Returns: mean={Y_returns.mean():.2f}, std={Y_returns.std():.2f}")

        # --- Supervised training (no LSTM context — feedforward on single frames) ---
        # This is intentional: we can't replay temporal context from stored
        # transitions. Instead we train the projection + value head to give a
        # reasonable baseline prediction from single observations, then the
        # LSTM will refine this during RL training with live temporal state.
        self.critic_network.train()  # ensure train mode (get_action_and_value sets eval)
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(N, device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                s = X_scans[idx]
                st = X_states[idx]
                y = Y_returns[idx]

                # Forward: encode single frame (no obs_buffer / hidden)
                feat, _, _, _ = self.critic_network.encode_observation(s, st, obs_buffer=None)
                pred = self.critic_network.fc_layers(feat).squeeze(-1)
                loss = F.smooth_l1_loss(pred, y)

                pretrain_optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1.0)
                pretrain_optim.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  best={best_loss:.4f}")

        # Unfreeze vision encoder for RL fine-tuning
        for p in self.critic_network.conv_layers.parameters():
            p.requires_grad = True

        # Re-create the optimizer so it includes all params with fresh state
        self.critic_optimizer = torch.optim.AdamW(
            self.critic_network.parameters(), lr=self.lr_critic, weight_decay=0.01,
        )
        self.critic_scaler = torch.amp.GradScaler("cuda")

        # Reset temporal state for clean start
        self.reset_buffers()
        self._prev_vels_x[:] = 0

        print(f"  Critic pretraining complete.  Best loss: {best_loss:.4f}")
        print("=" * 60 + "\n")

    def reset_buffers(self, agent_indices=None):
        """Reset LSTM hidden states and observation buffers for both actor and critic."""
        if agent_indices is None:
            # Full reset for all agents
            self.actor_buffer = deque(
                [self.actor_network.create_observation_buffer(self.num_agents, self.device)],
                maxlen=2,
            )
            self.actor_hidden = self.actor_network.get_init_hidden(
                self.num_agents, self.device, transpose=True
            )
            self.critic_buffer = deque(
                [self.critic_network.create_observation_buffer(self.num_agents, self.device)],
                maxlen=2,
            )
            self.critic_hidden = self.critic_network.get_init_hidden(
                self.num_agents, self.device, transpose=True
            )
            self._prev_vels_x = np.zeros(self.num_agents)
        else:
            agent_indices = agent_indices[agent_indices < self.num_agents]
            # Per-agent reset: zero out specific agent slots
            if self.actor_buffer[-1] is not None:
                for idx in agent_indices:
                    self.actor_buffer[-1][idx] = 0.0
            h, c = self.actor_hidden
            for idx in agent_indices:
                h[idx] = 0.0
                c[idx] = 0.0
            self.actor_hidden = (h, c)
            # Critic per-agent reset
            if self.critic_buffer[-1] is not None:
                for idx in agent_indices:
                    self.critic_buffer[-1][idx] = 0.0
            ch, cc = self.critic_hidden
            for idx in agent_indices:
                ch[idx] = 0.0
                cc[idx] = 0.0
            self.critic_hidden = (ch, cc)
            self._prev_vels_x[agent_indices] = 0.0
    
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
            # Value estimate (critic with LSTM temporal state)
            critic_features, new_critic_buf, new_ch, new_cc = self.critic_network.encode_observation(
                scan_tensor[: self.num_agents],
                state_tensor[: self.num_agents],
                self.critic_buffer[-1],
                self.critic_hidden[0],
                self.critic_hidden[1],
            )
            value = self.critic_network.fc_layers(critic_features)

            # Always advance critic temporal state when we have valid obs
            self.critic_buffer.append(new_critic_buf)
            self.critic_hidden = (new_ch, new_cc)

            # Cache critic features for store_transition
            self._last_critic_features = critic_features.float()

            if not store:
                # Bootstrapping: only value needed — skip diffusion entirely
                return None, None, value

            # Encode observation through CNN + LSTM
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                obs_features, new_buffer, new_h, new_c = self.actor_network.encode_observation(
                    scan_tensor[: self.num_agents],
                    state_tensor[: self.num_agents],
                    self.actor_buffer[-1],
                    self.actor_hidden[0],
                    self.actor_hidden[1],
                )

            # Run reverse diffusion in float32 — the 25-step iterative
            # chain accumulates rounding errors that overflow in float16.
            obs_features_f32 = obs_features.float()
            action, chain, log_prob = self.actor_network.sample_action_with_chain(
                obs_features_f32, deterministic=deterministic
            )

            # Safety: clamp to valid action range & replace any residual NaN
            action = action.clamp(
                self.actor_network.action_lo.unsqueeze(0),
                self.actor_network.action_hi.unsqueeze(0),
            )
            if torch.isnan(action).any():
                print("[WARNING] NaN in sampled action — replacing with zeros")
                action = torch.nan_to_num(action, nan=0.0)

            # Advance temporal state
            self.actor_buffer.append(new_buffer)
            self.actor_hidden = (new_h, new_c)

            # Cache obs_features for store_transition (avoids re-encoding
            # with zero hidden states during learn(), eliminating the
            # train-test temporal mismatch).
            self._last_obs_features = obs_features.float()

        return action, log_prob, value.squeeze(-1) if value.ndim > 1 else value


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
                "obs_features": self._last_obs_features,
                "critic_features": self._last_critic_features,
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

    def calculate_reward(self, next_obs):
        collisions = np.array(next_obs["collisions"][:self.num_agents])
        speeds = np.array(next_obs["linear_vels_x"][:self.num_agents])
        positions = np.stack([
            np.array(next_obs['poses_x'][:self.num_agents]),
            np.array(next_obs['poses_y'][:self.num_agents]),
        ], axis=1)
        wall_collisions = collisions == 1
        agent_collisions = collisions == 2
        rewards = np.zeros(self.num_agents)

        # --- Track position for checkpoints / laps (no progress reward) ---
        # for i in range(self.num_agents):
        #     projected_s, new_wp_idx = self._project_to_raceline(
        #         positions[i],
        #         self.last_wp_index[i],
        #         lookahead=50,
        #     )

        #     # Guard against NaN positions (e.g. from NaN actions)
        #     if np.isnan(projected_s):
        #         projected_s = self.last_cumulative_distance[i]
        #         new_wp_idx = self.last_wp_index[i]

        #     # --- Checkpoint reward (divide track into NUM_CHECKPOINTS segments) ---
        #     segment_len = self.raceline_length / self.NUM_CHECKPOINTS
        #     new_ckpt = int(projected_s / segment_len) % self.NUM_CHECKPOINTS
        #     if new_ckpt != self.last_checkpoint[i]:
        #         rewards[i] += self.CHECKPOINT_REWARD
        #         self.last_checkpoint[i] = new_ckpt

        #     self.last_cumulative_distance[i] = projected_s
        #     self.last_wp_index[i] = new_wp_idx

        # --- SPEED BONUS vs GFPP reference ---
        target_speed = self.gfpp.get_actions_batch(next_obs)
        speed_bonus = (speeds - target_speed[:self.num_agents, 1]) * self.SPEED_REWARD
        rewards += speed_bonus

        # --- LAP COMPLETION REWARD ---
        lap_counts = np.array(next_obs['lap_counts'][:self.num_agents], dtype=int)
        laps_completed = lap_counts - self._prev_lap_counts
        rewards += np.clip(laps_completed, 0, 1) * self.LAP_REWARD
        self._prev_lap_counts = lap_counts.copy()

        # --- COLLISION PENALTIES ---
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
                self._prev_lap_counts[i] = 0
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
        self._prev_lap_counts[:] = 0

    def learn(self, collisions, reward):
        """Advantage-Weighted Diffusion Regression + Diffusion Regulariser.

        Replaces the (broken) per-step importance-sampled PPO with a simpler,
        more stable objective:

            L_actor = E_t[ w(A) * ||ε_θ - ε||² ]  +  λ * E_t[ ||ε_θ - ε||² ]

        where  w(A) = exp(A / β)  weights the noise-prediction MSE by the
        normalised advantage, biasing the denoiser toward high-return actions.
        The λ term keeps pure denoising ability from degrading.

        The critic uses Huber loss with PPO-style value clipping.
        """
        print("Starting AWR-Diffusion learning...")
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
        obs_features_all = data["obs_features"]  # Cached from rollout (with LSTM context)
        critic_features_all = data["critic_features"]  # Cached critic LSTM features
        actions_all = data["action"]
        raw_advantages_all = data["raw_advantage"]  # Un-normalised for AWR weights
        advantages_all = data["advantage"]           # Normalised for diagnostics
        value_targets_all = data["value_target"]
        old_values_all = data["state_value"]

        num_timesteps = len(data)

        self.actor_network.train()
        self.critic_network.train()
        

        # Flush stale intermediate features accumulated during rollout.
        # The diffusion_utils ConditionalDenoisingMLP appends to a list on
        # every forward pass (including the K denoising steps inside
        # sample_action_with_chain).  Without this clear, the first
        # minibatch would receive ~steps*K rollout features alongside its
        # own, poisoning the dispersive loss.
        self.actor_network.denoise_net.get_intermediate_features()  # returns & clears

        K = self.num_diffusion_steps

        print(f"  Training: {self.epochs} epochs, {num_timesteps} timesteps, "
              f"AWR temp={self.awr_temperature}, diff_reg={self.diff_reg_lambda}")

        for epoch in range(self.epochs):
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_diff_loss = 0.0
            epoch_disp_loss = 0.0
            num_updates = 0

            indices = torch.randperm(num_timesteps)

            for mb_start in range(0, num_timesteps, self.minibatch_size):
                mb_end = min(mb_start + self.minibatch_size, num_timesteps)
                mb_idx = indices[mb_start:mb_end]

                obs_scan = obs_scan_all[mb_idx]
                obs_state = obs_state_all[mb_idx]
                obs_features = obs_features_all[mb_idx]
                critic_features = critic_features_all[mb_idx]
                actions = actions_all[mb_idx]
                raw_advantages = raw_advantages_all[mb_idx]
                advantages = advantages_all[mb_idx]
                value_targets = value_targets_all[mb_idx]
                old_values = old_values_all[mb_idx]
                speeds = np.array(obs_state[:, :, 0].contiguous().cpu(), dtype=np.float32)

                # Flatten agents dimension if present: [T, A, ...] → [T*A, ...]
                if obs_scan.ndim == 4:  # [T, A, 1, beams]
                    T, A = obs_scan.shape[:2]
                    obs_scan = obs_scan.reshape(T * A, *obs_scan.shape[2:])
                    obs_state = obs_state.reshape(T * A, *obs_state.shape[2:])
                    obs_features = obs_features.reshape(T * A, *obs_features.shape[2:])
                    critic_features = critic_features.reshape(T * A, *critic_features.shape[2:])
                    actions = actions.reshape(T * A, *actions.shape[2:])
                    raw_advantages = raw_advantages.reshape(T * A)
                    advantages = advantages.reshape(T * A)
                    value_targets = value_targets.reshape(T * A)
                    old_values = old_values.reshape(T * A)
                elif obs_scan.ndim == 3 and advantages.ndim == 2:
                    T, A = advantages.shape
                    obs_scan = obs_scan.reshape(T * A, *obs_scan.shape[2:]) if obs_scan.shape[0] == T else obs_scan
                    obs_state = obs_state.reshape(T * A, -1) if obs_state.shape[0] == T else obs_state
                    obs_features = obs_features.reshape(T * A, -1)
                    critic_features = critic_features.reshape(T * A, -1)
                    actions = actions.reshape(T * A, -1)
                    raw_advantages = raw_advantages.reshape(T * A)
                    advantages = advantages.reshape(T * A)
                    value_targets = value_targets.reshape(T * A)
                    old_values = old_values.reshape(T * A)

                B = obs_scan.shape[0]

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                # obs_features were cached during rollout with full LSTM
                # temporal context — no need to re-encode.  Gradients flow
                # through the denoiser only (encoder stays at BC-pretrained
                # weights, avoiding the need for TBPTT).

                # Actor: Advantage-Weighted Diffusion Regression
                actions_norm = self.actor_network.normalize_action(actions)

                # Random diffusion timestep per sample (standard DDPM training)
                t = torch.randint(0, K, (B,), device=self.device)
                noise = torch.randn_like(actions_norm)
                noisy_actions = self.actor_network.q_sample(actions_norm, t, noise=noise)
                pred_noise = self.actor_network.predict_noise(
                    noisy_actions, obs_features, t
                )

                # Per-sample noise prediction MSE: (B,)
                per_sample_mse = ((pred_noise - noise) ** 2).sum(dim=-1)

                # Advantage weights: exp(A / β) using RAW (un-normalised) advantages
                # Normalise to zero-mean unit-std BEFORE exp() so the temperature
                # parameter actually controls the sharpness rather than having
                # γ=0.999 GAE advantages (magnitude 50-200) immediately saturate
                # the exp() at the clamp ceiling.
                adv = raw_advantages.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                weights = torch.exp(adv / self.awr_temperature)
                weights = weights.clamp(max=self.awr_max_weight)
                weights = weights / (weights.mean() + 1e-8)

                # Weighted diffusion loss (advantage-weighted regression)
                awr_loss = (weights * per_sample_mse).mean()

                # Unweighted diffusion loss (regulariser — maintains denoising ability)
                diff_reg = per_sample_mse.mean()

                # Dispersive loss on intermediate denoiser features
                # The forward hook already captured features from predict_noise above
                feat_dict = self.actor_network.denoise_net.get_intermediate_features()
                if isinstance(feat_dict, list):
                    feat_dict = {i: f for i, f in enumerate(feat_dict)}
                disp_loss = torch.tensor(0.0, device=self.device)
                n_feats = 0
                for _, feats in feat_dict.items():
                    if feats.ndim > 2:
                        feats = feats.mean(dim=list(range(1, feats.ndim - 1)))
                    dl = dispersive_loss_infonce_l2(
                        feats, self.dispersive_temperature
                    )
                    if not torch.isnan(dl) and not torch.isinf(dl):
                        disp_loss = disp_loss + dl
                        n_feats += 1
                if n_feats > 0:
                    disp_loss = disp_loss / n_feats

                actor_loss = (
                    awr_loss
                    + self.diff_reg_lambda * diff_reg
                    + self.dispersive_lambda * disp_loss
                )

                # Critic: Huber loss with PPO-style value clipping
                # Use cached critic features (pre-computed with LSTM context during rollout)
                predicted_values = self.critic_network.forward_from_features(critic_features)
                if predicted_values.ndim > 1:
                    predicted_values = predicted_values.squeeze(-1)

                # PPO value clipping: prevent large value function updates
                v_clipped = old_values + (predicted_values - old_values).clamp(
                    -self.clip_epsilon, self.clip_epsilon
                )
                critic_loss_unclipped = F.smooth_l1_loss(
                    predicted_values, value_targets.detach(), reduction='none'
                )
                critic_loss_clipped = F.smooth_l1_loss(
                    v_clipped, value_targets.detach(), reduction='none'
                )
                critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

                # Backward & update
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
                epoch_diff_loss += diff_reg.item()
                epoch_disp_loss += disp_loss.item()
                num_updates += 1

                current_gen_diagnostics["loss_actor"].append(actor_loss.item())
                current_gen_diagnostics["loss_critic"].append(critic_loss.item())
                current_gen_diagnostics["loss_diffusion"].append(diff_reg.item())
                current_gen_diagnostics["loss_dispersive"].append(disp_loss.item())
                current_gen_diagnostics["adv_weight_std"].append(weights.std().item())
                current_gen_diagnostics["avg_speed"].append(float(speeds.mean()) if hasattr(speeds, 'mean') else float(speeds))

            if num_updates > 0:
                print(
                    f"  Epoch {epoch+1}/{self.epochs}: "
                    f"Actor={epoch_actor_loss/num_updates:.4f}, "
                    f"Critic={epoch_critic_loss/num_updates:.4f}, "
                    f"Diff={epoch_diff_loss/num_updates:.4f}, "
                    f"Disp={epoch_disp_loss/num_updates:.4f}, "
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
            if ax.get_legend_handles_labels()[1]:  # Only add legend if there are labeled artists
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

        # --- Per-generation reward normalisation ---
        # Normalise by this generation's reward std only (avoids cross-map
        # contamination from running statistics when maps switch frequently).
        reward_std = rewards.std().clamp(min=1e-4)
        rewards = rewards / reward_std

        timesteps = rewards.shape[0]
        num_agents = rewards.shape[1] if rewards.ndim == 2 else 1

        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(values, device=self.device)
        gae = torch.zeros(num_agents, device=self.device)

        for t in reversed(range(timesteps)):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        # Store raw advantages for AWR weighting (before normalisation)
        data.set("raw_advantage", advantages.clone())

        # Normalised advantages (for diagnostics / logging)
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
        if not os.path.exists(path):
            print(f"Warning: vision checkpoint '{path}' not found — using random init.")
            return new_encoder.to(self.device)
        checkpoint = torch.load(path, weights_only=False)
        # Try multiple prefixes: CriticNetwork uses conv_layers.*,
        # DiffusionLSTM uses vision_encoder.*, legacy uses 0.module.conv_layers.*
        prefixes = ["conv_layers.", "vision_encoder.", "0.module.conv_layers.", "0.module.vision_encoder."]
        encoder_sd = {}
        for k, v in checkpoint.items():
            if not isinstance(v, torch.Tensor):
                continue
            for prefix in prefixes:
                if k.startswith(prefix):
                    encoder_sd[k[len(prefix):]] = v
                    break
        if encoder_sd:
            # Filter to only matching shapes
            ref_sd = new_encoder.state_dict()
            filtered = {k: v for k, v in encoder_sd.items() if k in ref_sd and ref_sd[k].shape == v.shape}
            if filtered:
                new_encoder.load_state_dict(filtered, strict=False)
                print(f"Loaded {len(filtered)}/{len(ref_sd)} pre-trained vision encoder weights from '{path}'.")
            else:
                print(f"No compatible vision encoder weights found in '{path}'.")
        else:
            print(f"No vision encoder keys found in '{path}'.")
        return new_encoder.to(self.device)

    def _load_waypoints(self, map_name):
        waypoint_file = f"maps/{map_name}/{map_name}_raceline.csv"
        waypoints = np.loadtxt(waypoint_file, delimiter=";")
        waypoints_xy = waypoints[:, 1:3]
        positions = waypoints[:, 1:3]
        distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        waypoints_s = np.insert(np.cumsum(distances), 0, 0)
        raceline_length = waypoints_s[-1]
        
        self.gfpp.update_map(map_name)
        
        return waypoints_xy, waypoints_s, raceline_length

    def _obs_to_tensors(self, obs):
        scans_arr = np.array(obs["scans"])
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

        # Compute acceleration from consecutive velocities
        # (sim's linear_accel_x is always 0)
        lax = (lvx - self._prev_vels_x) / self._sim_dt
        lax = np.clip(lax, -10.0, 10.0)  # Match normalisation range
        self._prev_vels_x = lvx.copy()

        state_data = np.stack((lvx, lvy, avz, lax), axis=1)
        state_tensor = torch.from_numpy(state_data.astype(np.float32))

        return scan_tensors.to(self.device), state_tensor.to(self.device)