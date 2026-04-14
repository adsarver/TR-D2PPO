"""
D2PPO: Diffusion Policy Policy Optimization — Pure RL
=====================================================
Implementation based on: "D²PPO: Diffusion Policy Policy Optimization with
Dispersive Loss" (Zou et al., 2025) - arXiv:2508.02644

The diffusion policy models the action distribution as an iterative denoising
process (DDPM).  This agent uses **only** the PPO (Stage 2) objective — no
supervised / BC pre-training or dispersive loss.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tensordict import TensorDict
from baselines.mpc_agent import MPCAgent
from models.AuxModels import VisionEncoder
from models.CriticNetworks import CriticNetwork, Mamba2CriticNetwork
from models.DiffusionMamba2 import DiffusionMamba2


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
        num_diffusion_steps=100,
        beta_schedule="cosine",
        baseline_speed=6.0,
        # TBTT config (0 = disabled, uses shuffled minibatches)
        tbtt_length=0,
        checkpoint_every=0,
        ddim_k=5, # 1 for training, 0 for eval, 5 for fine-tune
    ):
        # --- Hyperparameters ---
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_actor = 3e-5
        self.lr_critic = 1e-4
        self.gamma = 0.99             # 0.999 needs >1k-step rollouts; 0.99 suits 256-step gens
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2       # Value-clipping range for critic
        self.max_grad_norm_actor = 0.5
        self.max_grad_norm_critic = 1.0
        self.state_dim = 4
        self.num_scan_beams = 1080
        self.lidar_fov = 4.7
        self.minibatch_size = 128
        self.epochs = 3 
        self.params = params
        self.mpc = MPCAgent(
            map_name=map_name,
            wheelbase=params['lf'] + params['lr'],
            max_steering=params['s_max'],
            num_beams=self.num_scan_beams,
            fov=self.lidar_fov,
            horizon=8,
            speed_scale=0.8,
            emergency_dist=0.8,
            speed_clamp=baseline_speed
        )

        # --- TBTT config ---
        self.tbtt_length = tbtt_length
        self.checkpoint_every = checkpoint_every

        # --- Advantage-weighted diffusion config ---
        self.num_diffusion_steps = num_diffusion_steps
        self.awr_temperature = 1.0    # β for exp(A/β) advantage weighting
        self.awr_max_weight = 20.0    # Clamp max weight for stability
        self.diff_reg_lambda = 0.1    # Keep small so AWR advantage signal dominates
        self.dispersive_lambda = 0.5  # Dispersive loss weight (reduced for Stage-2 RL)
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
        self.COLLISION_PENALTY = -8.0
        self.SPEED_REWARD = 1.5
        self.PROGRESS_REWARD = 2.0        # Per-meter forward progress along raceline
        self.AGENT_COLLISION_PENALTY = -4.0
        self.NUM_CHECKPOINTS = 10
        self._prev_lap_counts = np.zeros(self.num_agents, dtype=int)

        # --- Running reward normalisation (EMA across generations) ---
        self._reward_ema_mean = 0.0
        self._reward_ema_var = 1.0

        # --- Networks ---
        actor_encoder = self._transfer_vision(transfer[0])
        critic_encoder = self._transfer_vision(transfer[0])

        # Diffusion Policy Actor (Mamba2 temporal backbone)
        self.stride = 50
        self.actor_network = DiffusionMamba2(
            state_dim=self.state_dim,
            action_dim=2,
            encoder=actor_encoder,
            num_diffusion_steps=num_diffusion_steps,
            inference_steps=5,          # DDIM fast sampling for rollout/deploy
            obs_feature_dim=16,
            time_emb_dim=32,
            hidden_dims=(128, 128),
            beta_schedule=beta_schedule,
            d_model=16,
            d_state=16,
            d_conv=4,
            d_head=8,
            expand=2,
            memory_length=128,
            memory_stride=self.stride,
            odom_expand=32,
        ).to(self.device)

        # Register dispersive hooks on last denoiser block (same as pretraining)
        self.actor_network.denoise_net.register_dispersive_hooks("late")

        # Critic with Mamba2 temporal backbone (mirrors actor architecture)
        self.critic_network = Mamba2CriticNetwork(
            state_dim=self.state_dim,
            encoder=critic_encoder,
            d_model=16,
            d_state=16,
            d_conv=4,
            d_head=8,
            expand=2,
            memory_length=128,
            memory_stride=self.stride,
            odom_expand=32,
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

        # --- On-policy transition storage (ordered list for correct GAE) ---
        self.buffer = []
        self._last_obs_features = None  # Cached by get_action_and_value for store_transition
        self._pending_transition = None  # Deferred write for next/state_value

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

        # For computing acceleration from consecutive velocity observations
        # (the sim's linear_accel_x is always 0)
        self._prev_vels_x = np.zeros(self.num_agents)
        self._sim_dt = 0.01  # f110_gym default timestep

        # --- Deployment mode ---
        self._deploy_mode = False
        self._deploy_action_repeat = 0      # run inference every N sim steps
        self._deploy_ddim_steps = 1         # DDIM steps in deploy mode (paper: 0 intermediate = 1 call)
        self._cached_action = None          # last action for repeating
        self._action_repeat_counter = 0     # counts sim steps since last inference
        
    def clear_experience_buffer(self):
        self.buffer = []
        self._pending_transition = None

    # ------------------------------------------------------------------
    # Deployment mode  (optimised for Jetson Orin Nano ~100 Hz)
    # ------------------------------------------------------------------
    def deploy(self, action_repeat=0, ddim_steps=1, compile_model=True):
        """Switch to optimised deployment inference.

        Optimisations applied:
        1. DDIM-``ddim_steps`` instead of full 100-step DDPM.  The D2PPO
           paper uses 0 intermediate DDIM steps (``ddim_steps=1``, i.e.
           a single denoiser call that jumps directly to x_0).
        2. Action repeat — only run inference every ``action_repeat`` sim
           steps; intermediate steps reuse the cached action.
        3. Full fp16 inference (safe with ≤2 DDIM steps).
        4. ``torch.compile`` for fused kernels (optional, ~2x on Ampere+).
        5. Critic network is deleted to free VRAM.

        Call once after loading weights, before the eval / deployment loop.
        """
        self._deploy_mode = True
        self._deploy_action_repeat = max(1, action_repeat)
        self._deploy_ddim_steps = max(1, ddim_steps)
        self._cached_action = None
        self._action_repeat_counter = 0

        # Drop critic — not needed during deployment
        if hasattr(self, 'critic_network'):
            del self.critic_network
            del self.critic_optimizer
            del self.critic_scaler
            torch.cuda.empty_cache()

        self.actor_network.eval()
        self.actor_network.half()  # fp16 weights

        if compile_model:
            try:
                self.actor_network = torch.compile(
                    self.actor_network, mode="reduce-overhead")
                print("[deploy] torch.compile applied (reduce-overhead)")
            except Exception as e:
                print(f"[deploy] torch.compile unavailable: {e}")

        print(f"[deploy] DDIM-{self._deploy_ddim_steps}, "
              f"action_repeat={self._deploy_action_repeat}, fp16=True")

    # ------------------------------------------------------------------
    # Temporal state reset  (zeros the Mamba2 rolling feature buffers)
    # ------------------------------------------------------------------
    def reset_temporal_state(self, agent_idxs=None):
        """Reset internal temporal buffers on actor and critic networks.

        Args:
            agent_idxs: optional array of agent indices to reset.
                        If None, resets all agents.
        """
        self.actor_network.reset_temporal_state(agent_idxs)
        if hasattr(self, 'critic_network'):
            self.critic_network.reset_temporal_state(agent_idxs)

    # Backwards-compatible alias used by train.py, val.py, etc.
    reset_buffers = reset_temporal_state

    # ------------------------------------------------------------------
    # Critic pretraining  (run once before RL loop)
    # ------------------------------------------------------------------
    def pretrain_critic(self, env, pp_driver, num_agents_total, maps,
                        rollout_steps=512, num_rollouts=3, epochs=10,
                        lr=5e-4, batch_size=256,
                        save_demos_path="demos/critic_demos.pt", load_demos_path=None):
        """Pre-train the critic on MC returns collected by the pretrained actor.

        Runs the frozen actor for *num_rollouts* episodes on a diverse set of
        *maps*, computes discounted Monte-Carlo returns, then trains the
        critic's Mamba2 + value head via supervised MSE regression.  The vision
        encoder is frozen (already pretrained) so only the temporal and value
        layers are fitted.

        After data collection the demo tensors are saved to *save_demos_path*
        so subsequent runs can skip the expensive rollout phase.

        Should be called **once** right after agent construction, before the
        main RL loop begins.
        """
        from utils.utils import generate_start_poses, get_map_dir
        from baselines.pure_pursuit import PurePursuit

        print("\n" + "=" * 60)
        print("  CRITIC PRE-TRAINING")
        print("=" * 60)

        # Freeze the vision encoder — only train Mamba2 + projection + value head
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

        # --- Try loading pre-collected demos ---
        loaded_demos = False
        if load_demos_path and os.path.isfile(load_demos_path):
            print(f"  Loading cached demos from {load_demos_path} …")
            demos = torch.load(load_demos_path, map_location="cpu", weights_only=True)
            X_scans = demos["scans"]
            X_states = demos["states"]
            Y_returns = demos["returns"]
            loaded_demos = True
            print(f"  Loaded {X_scans.shape[0]} samples from disk.")

        if not loaded_demos:
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
                self.reset_temporal_state()
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
                        self.reset_temporal_state(stuck)
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

            # --- Save critic demos ---
            if save_demos_path:
                os.makedirs(os.path.dirname(save_demos_path) or ".", exist_ok=True)
                torch.save({
                    "scans": X_scans.cpu(),
                    "states": X_states.cpu(),
                    "returns": Y_returns.cpu(),
                }, save_demos_path)
                print(f"  Saved critic demos ({N} samples) \u2192 {save_demos_path}")

        N = X_scans.shape[0]
        print(f"  Collected {N} samples for critic pretraining."
              f"  Returns: mean={Y_returns.mean():.2f}, std={Y_returns.std():.2f}")

        # --- Supervised training (no temporal context — feedforward on single frames) ---
        # This is intentional: we can't replay temporal context from stored
        # transitions. Instead we train the projection + value head to give a
        # reasonable baseline prediction from single observations, then the
        # Mamba2 will refine this during RL training with live temporal state.
        self.critic_network.train()  # ensure train mode (get_action_and_value sets eval)
        best_loss = float('inf')
        total_batches = (N + batch_size - 1) // batch_size
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                s = X_scans[idx].to(self.device)
                st = X_states[idx].to(self.device)
                y = Y_returns[idx].to(self.device)

                # Forward: encode single frame (fresh zero buffer = no temporal context)
                _buf = self.critic_network.create_observation_buffer(s.shape[0], self.device)
                feat, _ = self.critic_network.encode_observation(s, st, obs_buffer=_buf)
                pred = self.critic_network.fc_layers(feat).squeeze(-1)
                loss = F.smooth_l1_loss(pred, y)

                pretrain_optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1.0)
                pretrain_optim.step()
                epoch_loss += loss.item()
                n_batches += 1

                # Batch progress
                if n_batches % 10 == 0 or n_batches == total_batches:
                    pct = n_batches / total_batches * 100
                    print(f"    Epoch {epoch}/{epochs}  batch {n_batches}/{total_batches}  ({pct:.0f}%)  loss={loss.item():.4f}", end='\r')

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  best={best_loss:.4f}" + ' ' * 40)

        # Unfreeze vision encoder for RL fine-tuning
        for p in self.critic_network.conv_layers.parameters():
            p.requires_grad = True

        # Re-create the optimizer so it includes all params with fresh state
        self.critic_optimizer = torch.optim.AdamW(
            self.critic_network.parameters(), lr=self.lr_critic, weight_decay=0.01,
        )
        self.critic_scaler = torch.amp.GradScaler("cuda")

        # Save pretrained critic weights
        critic_save_dir = "models/critic/pretrained"
        os.makedirs(critic_save_dir, exist_ok=True)
        critic_save_path = os.path.join(critic_save_dir, "critic_pretrained.pt")
        torch.save(self.critic_network.state_dict(), critic_save_path)
        print(f"  Saved critic weights \u2192 {critic_save_path}")

        # Reset temporal state for clean start
        self.reset_temporal_state()
        self._prev_vels_x[:] = 0

        print(f"  Critic pretraining complete.  Best loss: {best_loss:.4f}")
        print("=" * 60 + "\n")
    
    def get_action_and_value(self, scan_tensor, state_tensor, deterministic=False, store=True):
        """
        Sample an action from the diffusion policy and compute state value.
        
        Compatible with PPOAgent interface: returns (action, log_prob, value).
        
        When ``store=True`` the observation buffers are
        advanced (used during rollout collection).  When ``store=False`` we
        only need the value estimate (e.g. for bootstrapping next-state value)
        and the temporal state is left untouched.
        """
        # ----- Fast deploy path (action repeat + DDIM-few + fp16) -----
        if self._deploy_mode:
            return self._get_action_deploy(scan_tensor, state_tensor)

        self.actor_network.eval()
        self.critic_network.eval()
        value = None

        with torch.no_grad():
            # Value estimate (critic with Mamba2 temporal state)
            if not deterministic:
                critic_features = self.critic_network.encode_observation(
                    scan_tensor[: self.num_agents],
                    state_tensor[: self.num_agents],
                )
                self._last_critic_features = critic_features.float()
                value = self.critic_network.fc_layers(critic_features)

                if not store:
                    # Bootstrapping: only value needed — skip diffusion entirely
                    v = value.squeeze(-1) if value.ndim > 1 else value
                    return None, None, v

            # Encode observation through CNN + Mamba2
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                obs_features = self.actor_network.encode_observation(
                    scan_tensor[: self.num_agents],
                    state_tensor[: self.num_agents],
                )

            # Run reverse diffusion in float32 — the iterative chain
            # accumulates rounding errors that overflow in float16.
            obs_features_f32 = obs_features.float()
            if deterministic:
                # Full stochastic DDPM reverse process (same 50-step chain
                # used during training).  The per-step noise is integral to
                # reverse diffusion — it is NOT RL exploration.  Mean-only
                # DDPM collapses to ~0, and few-step DDIM is too coarse.
                B = obs_features_f32.shape[0]
                device = obs_features_f32.device
                x = torch.randn(B, self.actor_network.action_dim, device=device)
                for k in reversed(range(self.actor_network.num_diffusion_steps)):
                    t = torch.full((B,), k, device=device, dtype=torch.long)
                    x = self.actor_network.p_sample(x, obs_features_f32, t)
                action = self.actor_network.denormalize_action(x.clamp(-1.0, 1.0))
                log_prob = None
            else:
                # DDIM-5: 5 denoiser MLP calls instead of 10 DDPM steps.
                # eta=0.5 preserves stochasticity for exploration.
                # log_prob is unused by AWR, so we skip the expensive
                # inline log-prob computation of sample_action_with_chain.
                action = self.actor_network.ddim_sample_action(
                    obs_features_f32, num_steps=5, eta=0.5
                )
                log_prob = None

            # Safety: clamp to valid action range & replace any residual NaN
            action = action.clamp(
                self.actor_network.action_lo.unsqueeze(0),
                self.actor_network.action_hi.unsqueeze(0),
            )
            if torch.isnan(action).any():
                print("[WARNING] NaN in sampled action — replacing with zeros")
                action = torch.nan_to_num(action, nan=0.0)

            # Cache obs_features for store_transition (avoids re-encoding
            # with zero hidden states during learn(), eliminating the
            # train-test temporal mismatch).
            self._last_obs_features = obs_features.float()

        return action, log_prob, value.squeeze(-1) if value is not None and value.ndim > 1 else value

    # ------------------------------------------------------------------
    # Deploy-mode fast inference  (DDIM-few + action repeat + fp16)
    # ------------------------------------------------------------------
    def _get_action_deploy(self, scan_tensor, state_tensor):
        """Optimised inference for deployment (~100 Hz on Jetson Orin Nano).

        - Only runs the actor (no critic).
        - Uses DDIM with very few steps (default 2).
        - Entire pipeline in fp16 (model weights already .half()'d).
        - Action repeat: returns cached action on non-inference steps,
          but *always* updates the Mamba2 buffer so temporal context
          stays correct.
        """
        self._action_repeat_counter += 1
        need_new_action = (
            self._cached_action is None
            or self._action_repeat_counter >= self._deploy_action_repeat
        )

        with torch.no_grad():
            # Always update temporal buffer (even on repeat steps)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                obs_features = self.actor_network.encode_observation(
                    scan_tensor[: self.num_agents],
                    state_tensor[: self.num_agents],
                )

            if need_new_action:
                # DDIM with few steps, fully in fp16
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    action = self.actor_network.ddim_sample_action(
                        obs_features, num_steps=self._deploy_ddim_steps, eta=0.0
                    )
                action = action.clamp(
                    self.actor_network.action_lo.unsqueeze(0),
                    self.actor_network.action_hi.unsqueeze(0),
                )
                if torch.isnan(action).any():
                    action = torch.nan_to_num(action, nan=0.0)
                self._cached_action = action
                self._action_repeat_counter = 0

        return self._cached_action, None, None


    def store_transition(self, obs, next, action, log_prob, reward, done, value):
        done_tensor = torch.tensor(done, dtype=torch.bool).unsqueeze(-1)

        # Finalize the previous pending transition: its next/state_value
        # is this step's state_value (same obs, same critic buffer state).
        if self._pending_transition is not None:
            self._pending_transition["next", "state_value"] = value
            self.buffer.append(self._pending_transition)

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
                        "state_value": torch.zeros_like(value),  # placeholder
                        "reward": reward,
                        "done": done_tensor,
                    }
                ),
            },
            batch_size=[self.num_agents],
        )
        self._pending_transition = step_data.to(self.device)

    def finalize_rollout(self, next_obs):
        """Flush the last pending transition with a single bootstrap call."""
        if self._pending_transition is not None:
            next_scans, next_states = self._obs_to_tensors(next_obs)
            _, _, next_value = self.get_action_and_value(
                next_scans, next_states, store=False
            )
            self._pending_transition["next", "state_value"] = next_value
            self.buffer.append(self._pending_transition)
            self._pending_transition = None

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

        # --- Track progress: checkpoints + continuous forward progress ---
        for i in range(self.num_agents):
            projected_s, new_wp_idx = self._project_to_raceline(
                positions[i],
                self.last_wp_index[i],
                lookahead=50,
            )

            # Guard against NaN positions (e.g. from NaN actions)
            if np.isnan(projected_s):
                projected_s = self.last_cumulative_distance[i]
                new_wp_idx = self.last_wp_index[i]

            # --- Continuous forward-progress reward ---
            delta_s = projected_s - self.last_cumulative_distance[i]
            # Handle raceline wrap-around (positive = forward)
            if delta_s < -self.raceline_length * 0.5:
                delta_s += self.raceline_length
            elif delta_s > self.raceline_length * 0.5:
                delta_s -= self.raceline_length
            # Only reward forward progress, don't penalise backward (collision resets)
            if delta_s > 0:
                rewards[i] += delta_s * self.PROGRESS_REWARD

            # --- Checkpoint reward (divide track into NUM_CHECKPOINTS segments) ---
            segment_len = self.raceline_length / self.NUM_CHECKPOINTS
            new_ckpt = int(projected_s / segment_len) % self.NUM_CHECKPOINTS
            if new_ckpt != self.last_checkpoint[i]:
                rewards[i] += self.CHECKPOINT_REWARD
                self.last_checkpoint[i] = new_ckpt

            self.last_cumulative_distance[i] = projected_s
            self.last_wp_index[i] = new_wp_idx

        # --- SPEED BONUS vs MPC reference ---
        target_speed = self.mpc.get_actions_batch(next_obs)
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

        When ``self.tbtt_length > 0``, observations are re-encoded through
        the Mamba2 temporal backbone sequentially (TBTT) so gradients flow
        through the encoder — allowing the Mamba2 to continue learning
        temporal representations during RL fine-tuning.
        """
        print("Starting AWR-Diffusion learning...")
        print(f"  Buffer size: {len(self.buffer)}")

        # Stack in insertion order — temporal ordering is critical for GAE.
        data = torch.stack(self.buffer).contiguous()

        current_gen_diagnostics = {key: [] for key in self.diagnostic_keys}
        current_gen_diagnostics["collisions"] = [collisions]
        current_gen_diagnostics["reward"] = [reward]

        # Compute GAE
        with torch.no_grad():
            data = self._compute_gae(data)

        # Dispatch to TBTT or shuffled-minibatch learner
        if self.tbtt_length > 0:
            self._learn_tbtt(data, current_gen_diagnostics)
        else:
            self._learn_shuffled(data, current_gen_diagnostics)

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

        self.buffer.clear()
        del data
        torch.cuda.empty_cache()
        print("[D²PPO Stage 2] Learning complete.")

    # ------------------------------------------------------------------
    # Shuffled-minibatch learner (original — no encoder gradients)
    # ------------------------------------------------------------------
    def _learn_shuffled(self, data, current_gen_diagnostics):
        """Train on cached obs_features with shuffled minibatches.

        The Mamba2 encoder is frozen (uses rollout-cached features); only
        the diffusion denoiser and critic value head receive gradients.
        """
        obs_scan_all = data["observation_scan"]
        obs_state_all = data["observation_state"]
        obs_features_all = data["obs_features"]
        critic_features_all = data["critic_features"]
        actions_all = data["action"]
        raw_advantages_all = data["raw_advantage"]
        advantages_all = data["advantage"]
        value_targets_all = data["value_target"]
        old_values_all = data["state_value"]

        num_timesteps = len(data)

        self.actor_network.train()
        self.critic_network.train()

        self.actor_network.denoise_net.get_intermediate_features()  # flush stale

        K = self.num_diffusion_steps

        print(f"  Training (shuffled): {self.epochs} epochs, {num_timesteps} timesteps, "
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

                # obs_features were cached during rollout with full Mamba2
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
                # Use cached critic features (pre-computed with Mamba2 context during rollout)
                predicted_values = self.critic_network.forward_from_features(critic_features)
                if predicted_values.ndim > 1:
                    predicted_values = predicted_values.squeeze(-1)

                # Normalise value targets per-minibatch so the critic always
                # regresses to a unit-scale distribution.  This makes the loss
                # magnitude independent of the reward scale and prevents the
                # value clipping from choking the gradient.
                vt_mean = value_targets.mean().detach()
                vt_std  = value_targets.std().clamp(min=1e-4).detach()
                vt_norm = (value_targets.detach() - vt_mean) / vt_std
                ov_norm = (old_values.detach()    - vt_mean) / vt_std
                pv_norm = (predicted_values        - vt_mean) / vt_std

                # PPO value clipping on the normalised predictions
                v_clipped = ov_norm + (pv_norm - ov_norm).clamp(
                    -self.clip_epsilon, self.clip_epsilon
                )
                critic_loss_unclipped = F.smooth_l1_loss(
                    pv_norm, vt_norm, reduction='none'
                )
                critic_loss_clipped = F.smooth_l1_loss(
                    v_clipped, vt_norm, reduction='none'
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

    # ------------------------------------------------------------------
    # TBTT learner — re-encodes through Mamba2 with temporal context
    # ------------------------------------------------------------------
    def _learn_tbtt(self, data, current_gen_diagnostics):
        """TBTT variant: re-encode observations through Mamba2 with gradients.

        Processes the rollout buffer sequentially so the Mamba2 temporal
        backbone receives gradient signal, allowing it to continue learning
        temporal representations (buffer utilisation) during RL fine-tuning.

        Hidden / buffer state is detached at TBTT chunk boundaries to bound
        the backward graph.  Optional activation checkpointing
        (``self.checkpoint_every > 0``) trades recomputation for memory.
        """
        obs_scan_all = data["observation_scan"].to(self.device)     # (T, A, 1, beams)
        obs_state_all = data["observation_state"].to(self.device)   # (T, A, state_dim)
        actions_all = data["action"].to(self.device)                # (T, A, 2)
        raw_advantages_all = data["raw_advantage"].to(self.device)  # (T, A)
        value_targets_all = data["value_target"].to(self.device)    # (T, A)
        old_values_all = data["state_value"].to(self.device)        # (T, A)

        T = len(data)
        A = self.num_agents
        K = self.num_diffusion_steps
        tbtt_len = self.tbtt_length
        use_ckpt = self.checkpoint_every > 0
        num_chunks = (T + tbtt_len - 1) // tbtt_len

        # Pre-compute full-buffer statistics for stable normalisation
        raw_adv_flat = raw_advantages_all.flatten()
        adv_mean = raw_adv_flat.mean().detach()
        adv_std = raw_adv_flat.std().clamp(min=1e-8).detach()
        vt_flat = value_targets_all.flatten()
        vt_mean_g = vt_flat.mean().detach()
        vt_std_g = vt_flat.std().clamp(min=1e-4).detach()

        # Flush stale dispersive features from rollout
        self.actor_network.denoise_net.get_intermediate_features()

        self.actor_network.train()
        self.critic_network.train()

        # ── Segment function for activation checkpointing ────────────
        def _rl_segment_fn(actor_buf, critic_buf,
                           scans_seg, states_seg, actions_seg,
                           raw_adv_seg, vt_seg, ov_seg):
            """Process a temporal segment through both encoders + losses."""
            seg_len = scans_seg.shape[0]
            s_actor = torch.tensor(0.0, device=scans_seg.device)
            s_critic = torch.tensor(0.0, device=scans_seg.device)
            s_diff = torch.tensor(0.0, device=scans_seg.device)
            s_disp = torch.tensor(0.0, device=scans_seg.device)
            s_wstd = torch.tensor(0.0, device=scans_seg.device)

            for i in range(seg_len):
                with torch.amp.autocast("cuda"):
                    obs_feat, actor_buf = self.actor_network.encode_observation(
                        scans_seg[i], states_seg[i], actor_buf)
                    crit_feat, critic_buf = self.critic_network.encode_observation(
                        scans_seg[i], states_seg[i], critic_buf)

                obs_feat_f = obs_feat.float()
                crit_feat_f = crit_feat.float()

                # --- Actor AWR loss ---
                act_norm = self.actor_network.normalize_action(actions_seg[i])
                t_d = torch.randint(0, K, (act_norm.shape[0],),
                                    device=scans_seg.device)
                noise = torch.randn_like(act_norm)
                noisy = self.actor_network.q_sample(act_norm, t_d, noise=noise)
                pred = self.actor_network.predict_noise(noisy, obs_feat_f, t_d)
                mse = ((pred - noise) ** 2).sum(dim=-1)

                adv = (raw_adv_seg[i] - adv_mean) / adv_std
                w = torch.exp(adv / self.awr_temperature).clamp(
                    max=self.awr_max_weight)
                w = w / (w.mean() + 1e-8)

                awr = (w * mse).mean()
                dr = mse.mean()

                feat_dict = self.actor_network.denoise_net\
                    .get_intermediate_features()
                if isinstance(feat_dict, list):
                    feat_dict = {j: f for j, f in enumerate(feat_dict)}
                dl = torch.tensor(0.0, device=scans_seg.device)
                nf = 0
                for _, feats in feat_dict.items():
                    if feats.ndim > 2:
                        feats = feats.mean(
                            dim=list(range(1, feats.ndim - 1)))
                    d = dispersive_loss_infonce_l2(
                        feats, self.dispersive_temperature)
                    if not torch.isnan(d) and not torch.isinf(d):
                        dl = dl + d
                        nf += 1
                if nf > 0:
                    dl = dl / nf

                s_actor = s_actor + (
                    awr + self.diff_reg_lambda * dr
                    + self.dispersive_lambda * dl)
                s_diff = s_diff + dr
                s_disp = s_disp + dl
                s_wstd = s_wstd + w.std()

                # --- Critic value loss ---
                pv = self.critic_network.forward_from_features(crit_feat_f)
                if pv.ndim > 1:
                    pv = pv.squeeze(-1)
                pv_n = (pv - vt_mean_g) / vt_std_g
                vt_n = (vt_seg[i].detach() - vt_mean_g) / vt_std_g
                ov_n = (ov_seg[i].detach() - vt_mean_g) / vt_std_g

                vc = ov_n + (pv_n - ov_n).clamp(
                    -self.clip_epsilon, self.clip_epsilon)
                cl_u = F.smooth_l1_loss(pv_n, vt_n, reduction='none')
                cl_c = F.smooth_l1_loss(vc, vt_n, reduction='none')
                s_critic = s_critic + torch.max(cl_u, cl_c).mean()

            return (s_actor, s_critic, s_diff, s_disp, s_wstd,
                    actor_buf, critic_buf)

        print(f"  Training (TBTT): {self.epochs} epochs, {T} timesteps, "
              f"chunk={tbtt_len}, {num_chunks} chunks/epoch, "
              f"ckpt_every={self.checkpoint_every}")

        for epoch in range(self.epochs):
            # Fresh temporal state each epoch
            actor_obs_buffer = self.actor_network.create_observation_buffer(
                A, self.device)
            critic_obs_buffer = self.critic_network.create_observation_buffer(
                A, self.device)

            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_diff_loss = 0.0
            epoch_disp_loss = 0.0
            num_updates = 0

            for chunk_start in range(0, T, tbtt_len):
                chunk_end = min(chunk_start + tbtt_len, T)
                chunk_len = chunk_end - chunk_start

                # Detach temporal state at chunk boundary (truncated BPTT)
                actor_obs_buffer = actor_obs_buffer.detach().requires_grad_()
                critic_obs_buffer = critic_obs_buffer.detach().requires_grad_()

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                chunk_actor = torch.tensor(0.0, device=self.device)
                chunk_critic = torch.tensor(0.0, device=self.device)
                chunk_diff = torch.tensor(0.0, device=self.device)
                chunk_disp = torch.tensor(0.0, device=self.device)
                chunk_wstd = torch.tensor(0.0, device=self.device)

                if use_ckpt:
                    # --- Activation-checkpointed path ---
                    for seg_start in range(chunk_start, chunk_end,
                                           self.checkpoint_every):
                        seg_end = min(seg_start + self.checkpoint_every,
                                      chunk_end)
                        with torch.amp.autocast("cuda"):
                            (sa, sc, sd, sp, sw,
                             actor_obs_buffer,
                             critic_obs_buffer) = torch_checkpoint(
                                _rl_segment_fn,
                                actor_obs_buffer, critic_obs_buffer,
                                obs_scan_all[seg_start:seg_end],
                                obs_state_all[seg_start:seg_end],
                                actions_all[seg_start:seg_end],
                                raw_advantages_all[seg_start:seg_end],
                                value_targets_all[seg_start:seg_end],
                                old_values_all[seg_start:seg_end],
                                use_reentrant=False,
                            )
                        chunk_actor = chunk_actor + sa
                        chunk_critic = chunk_critic + sc
                        chunk_diff = chunk_diff + sd
                        chunk_disp = chunk_disp + sp
                        chunk_wstd = chunk_wstd + sw
                else:
                    # --- Standard (non-checkpointed) path ---
                    with torch.amp.autocast("cuda"):
                        for t_idx in range(chunk_start, chunk_end):
                            obs_feat, actor_obs_buffer = \
                                self.actor_network.encode_observation(
                                    obs_scan_all[t_idx],
                                    obs_state_all[t_idx],
                                    actor_obs_buffer,
                                )
                            crit_feat, critic_obs_buffer = \
                                self.critic_network.encode_observation(
                                    obs_scan_all[t_idx],
                                    obs_state_all[t_idx],
                                    critic_obs_buffer,
                                )

                            obs_feat_f = obs_feat.float()
                            crit_feat_f = crit_feat.float()

                            # Actor AWR loss
                            act_norm = self.actor_network.normalize_action(
                                actions_all[t_idx])
                            t_d = torch.randint(
                                0, K, (A,), device=self.device)
                            noise = torch.randn_like(act_norm)
                            noisy = self.actor_network.q_sample(
                                act_norm, t_d, noise=noise)
                            pred = self.actor_network.predict_noise(
                                noisy, obs_feat_f, t_d)
                            mse = ((pred - noise) ** 2).sum(dim=-1)

                            adv = (raw_advantages_all[t_idx] - adv_mean) \
                                / adv_std
                            w = torch.exp(
                                adv / self.awr_temperature
                            ).clamp(max=self.awr_max_weight)
                            w = w / (w.mean() + 1e-8)

                            chunk_actor = chunk_actor + (
                                (w * mse).mean()
                                + self.diff_reg_lambda * mse.mean()
                            )
                            chunk_diff = chunk_diff + mse.mean()
                            chunk_wstd = chunk_wstd + w.std()

                            # Dispersive loss
                            feat_dict = self.actor_network.denoise_net\
                                .get_intermediate_features()
                            if isinstance(feat_dict, list):
                                feat_dict = {
                                    j: f for j, f in enumerate(feat_dict)}
                            dl = torch.tensor(0.0, device=self.device)
                            nf = 0
                            for _, feats in feat_dict.items():
                                if feats.ndim > 2:
                                    feats = feats.mean(
                                        dim=list(range(
                                            1, feats.ndim - 1)))
                                d = dispersive_loss_infonce_l2(
                                    feats, self.dispersive_temperature)
                                if not (torch.isnan(d) or torch.isinf(d)):
                                    dl = dl + d
                                    nf += 1
                            if nf > 0:
                                dl = dl / nf
                            chunk_actor = chunk_actor + (
                                self.dispersive_lambda * dl)
                            chunk_disp = chunk_disp + dl

                            # Critic value loss
                            pv = self.critic_network\
                                .forward_from_features(crit_feat_f)
                            if pv.ndim > 1:
                                pv = pv.squeeze(-1)
                            pv_n = (pv - vt_mean_g) / vt_std_g
                            vt_n = (value_targets_all[t_idx].detach()
                                    - vt_mean_g) / vt_std_g
                            ov_n = (old_values_all[t_idx].detach()
                                    - vt_mean_g) / vt_std_g
                            vc = ov_n + (pv_n - ov_n).clamp(
                                -self.clip_epsilon, self.clip_epsilon)
                            cl_u = F.smooth_l1_loss(
                                pv_n, vt_n, reduction='none')
                            cl_c = F.smooth_l1_loss(
                                vc, vt_n, reduction='none')
                            chunk_critic = chunk_critic + torch.max(
                                cl_u, cl_c).mean()

                # Average over timesteps in chunk
                avg_actor = chunk_actor / chunk_len
                avg_critic = chunk_critic / chunk_len

                # Backward & update — actor first, then free its graph
                # before building the critic graph.  This halves peak
                # activation memory vs. holding both simultaneously.
                self.actor_scaler.scale(avg_actor).backward(
                    retain_graph=False)
                self.actor_scaler.unscale_(self.actor_optimizer)
                nn.utils.clip_grad_norm_(
                    self.actor_network.parameters(),
                    self.max_grad_norm_actor,
                )
                self.actor_scaler.step(self.actor_optimizer)
                self.actor_scaler.update()
                avg_actor_val = avg_actor.item()
                del avg_actor, chunk_actor  # free actor graph memory

                self.critic_scaler.scale(avg_critic).backward()
                self.critic_scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    self.critic_network.parameters(),
                    self.max_grad_norm_critic,
                )
                self.critic_scaler.step(self.critic_optimizer)
                self.critic_scaler.update()
                avg_critic_val = avg_critic.item()
                avg_diff_val = (chunk_diff / chunk_len).item()
                avg_disp_val = (chunk_disp / chunk_len).item()
                avg_wstd_val = (chunk_wstd / chunk_len).item()
                del avg_critic, chunk_critic  # free critic graph memory

                # Diagnostics
                epoch_actor_loss += avg_actor_val
                epoch_critic_loss += avg_critic_val
                epoch_diff_loss += avg_diff_val
                epoch_disp_loss += avg_disp_val
                num_updates += 1

                current_gen_diagnostics["loss_actor"].append(
                    avg_actor_val)
                current_gen_diagnostics["loss_critic"].append(
                    avg_critic_val)
                current_gen_diagnostics["loss_diffusion"].append(
                    avg_diff_val)
                current_gen_diagnostics["loss_dispersive"].append(
                    avg_disp_val)
                current_gen_diagnostics["adv_weight_std"].append(
                    avg_wstd_val)
                current_gen_diagnostics["avg_speed"].append(
                    float(obs_state_all[chunk_start:chunk_end, :, 0]
                          .mean().cpu()))

            if num_updates > 0:
                print(
                    f"  Epoch {epoch+1}/{self.epochs}: "
                    f"Actor={epoch_actor_loss/num_updates:.4f}, "
                    f"Critic={epoch_critic_loss/num_updates:.4f}, "
                    f"Diff={epoch_diff_loss/num_updates:.4f}, "
                    f"Disp={epoch_disp_loss/num_updates:.4f}, "
                )

            torch.cuda.empty_cache()

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

        # Running reward normalisation (EMA) — smooths across map transitions
        # instead of per-generation std which spikes at boundaries.
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item()
        self._reward_ema_mean = 0.99 * self._reward_ema_mean + 0.01 * batch_mean
        self._reward_ema_var = 0.99 * self._reward_ema_var + 0.01 * batch_var
        r_std = max(self._reward_ema_var ** 0.5, 1e-4)
        rewards = (rewards - self._reward_ema_mean) / r_std

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

        # Strip common wrapper prefixes: torch.compile → "_orig_mod.",
        # DataParallel/legacy → "0.module."
        state_dict = {}
        for k, v in state_dict_raw.items():
            if not isinstance(v, torch.Tensor):
                continue
            clean_k = k
            for prefix in ("_orig_mod.", "0.module."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            state_dict[clean_k] = v
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
        # DiffusionMamba2 uses vision_encoder.*, legacy uses 0.module.conv_layers.*,
        # torch.compile wraps with _orig_mod.*
        prefixes = [
            "conv_layers.", "vision_encoder.",
            "0.module.conv_layers.", "0.module.vision_encoder.",
            "_orig_mod.conv_layers.", "_orig_mod.vision_encoder.",
        ]
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
        
        self.mpc.update_map(map_name)
        
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