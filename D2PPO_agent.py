"""
D2PPO: Diffusion Policy Policy Optimization (Stage 2 — PPO fine-tune)
=====================================================================
Implementation based on: "D²PPO: Diffusion Policy Policy Optimization with
Dispersive Loss" (Zou et al., 2025) — arXiv:2508.02644

The diffusion policy models the action distribution as an iterative denoising
process (DDPM).  This agent implements the RL fine-tuning stage, which
assumes the actor has been BC-pretrained with dispersive loss (see
``weight_initializer.py``).

Losses applied during ``learn()``:
  * Actor: PPO-clipped diffusion objective on a stochastic DDIM chain
           (or Advantage-Weighted Regression fallback)
         + diffusion MSE regulariser (λ = ``diff_reg_lambda``)
         + dispersive InfoNCE-L2 regulariser (λ = ``dispersive_lambda``,
           enabled via ``use_dispersive_in_rl``)
  * Critic: Huber loss with PPO-style value clipping on normalised targets
"""

import copy
import math
import os
import multiprocessing as _mp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from baselines.mpc_agent import MPCAgent
from models.AuxModels import VisionEncoder
from models.CriticNetworks import CriticNetwork, Mamba2CriticNetwork
from models.DiffusionMamba2 import DiffusionMamba2
from utils.diffusion_utils import extract, dispersive_loss_infonce_l2
from utils.sim_config import D2PPO_STATE_DIM, LIDAR_BEAMS, LIDAR_FOV


class D2PPOAgent:
    """
    D²PPO Agent: Diffusion Policy Policy Optimization.

    Follows the same interface as PPOAgent for compatibility with the F1Tenth
    training loop (train.py), but replaces the Gaussian policy with a diffusion
    policy.  Pure RL — no supervised / BC pre-training.
    """

    @staticmethod
    def _critic_env_worker(conn, map_name, num_agents_total, num_agents_ai,
                           collision_reset_threshold, num_beams, lidar_fov, params):
        """Subprocess: owns one gym env + MPC driver for background agents.

        Protocol (after init):
          recv actions (np array) → env.step → send obs_payload
          recv None               → shutdown
        """
        import gym as _gym
        from utils.utils import generate_start_poses, get_map_dir

        env = _gym.make(
            "f110_gym:f110-v0",
            map=get_map_dir(map_name) + f"/{map_name}_map",
            num_agents=num_agents_total,
            num_beams=num_beams,
            fov=lidar_fov,
            params=params,
        )

        pp_driver = MPCAgent(
            map_name=map_name,
            wheelbase=params['lf'] + params['lr'],
            max_steering=params['s_max'],
            num_beams=num_beams,
            fov=lidar_fov,
            horizon=8,
            speed_scale=0.8,
            emergency_dist=0.8,
            speed_clamp=6.0,
        )

        poses = generate_start_poses(map_name, num_agents_total)
        obs, _, _, _ = env.reset(poses=poses)

        collision_timers = np.zeros(num_agents_total, dtype=np.int32)

        def _extract_obs(o):
            """Return numpy arrays the main process needs."""
            return {
                "scans": np.array(o["scans"], dtype=np.float32),
                "linear_vels_x": np.array(o["linear_vels_x"], dtype=np.float32),
                "linear_vels_y": np.array(o["linear_vels_y"], dtype=np.float32),
                "ang_vels_z": np.array(o["ang_vels_z"], dtype=np.float32),
                "poses_x": np.array(o["poses_x"], dtype=np.float32),
                "poses_y": np.array(o["poses_y"], dtype=np.float32),
                "poses_theta": np.array(o["poses_theta"], dtype=np.float32),
                "collisions": np.array(o["collisions"], dtype=np.float32),
                "lap_counts": np.array(o["lap_counts"], dtype=np.float32),
            }

        # Send initial obs + poses
        conn.send(("init", _extract_obs(obs), poses))

        while True:
            msg = conn.recv()
            if msg is None:
                break

            ai_actions = msg  # (num_agents_ai, 2)

            # PP/MPC for background agents
            pp_act = pp_driver.get_actions_batch(obs).astype(np.float32)
            actions = np.vstack((ai_actions, pp_act[num_agents_ai:])) \
                if ai_actions.shape[0] < num_agents_total else ai_actions

            next_obs, _, _, _ = env.step(actions)

            # Stuck detection
            cols = np.array(next_obs['collisions'][:num_agents_total])
            vels = np.array(next_obs['linear_vels_x'][:num_agents_total])
            collision_timers[(cols == 1) | ((vels < 0.1) & (vels > -0.1))] += 1
            collision_timers[cols == 0] = 0
            stuck = np.where(
                collision_timers >= collision_reset_threshold)[0]

            new_poses = None
            if len(stuck) > 0:
                cur_poses = np.stack([
                    next_obs['poses_x'], next_obs['poses_y'],
                    next_obs['poses_theta']], axis=1)
                new_poses = generate_start_poses(
                    map_name, num_agents_total, agent_poses=cur_poses)
                next_obs, _, _, _ = env.reset(
                    poses=new_poses, agent_idxs=stuck)
                collision_timers[stuck] = 0

            obs = next_obs
            conn.send(("step", _extract_obs(obs), stuck, new_poses))

        env.close()
        conn.close()

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
        # TBTT config (0 = disabled, uses shuffled minibatches)
        tbtt_length=512,
        ddim_k=5, # 1 for training, 0 for eval, 5 for fine-tune
    ):
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr_actor = 1e-5
        self.lr_critic = 5e-5
        self.gamma = 0.999
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.max_grad_norm_actor = 0.5
        self.max_grad_norm_critic = 1.0
        self.state_dim = D2PPO_STATE_DIM
        self.num_scan_beams = LIDAR_BEAMS
        self.lidar_fov = LIDAR_FOV
        self.minibatch_size = 128
        self.epochs = 4
        self.params = params

        self.tbtt_length = tbtt_length

        self.num_diffusion_steps = num_diffusion_steps
        self.awr_temperature = 0.5
        self.awr_max_weight = 10.0
        self.diff_reg_lambda = 0.02
        self.dispersive_lambda = 0.1
        self.dispersive_temperature = 0.5
        self.use_dispersive_in_rl = True
        self.dispersive_coef_rl = 0.03

        self.ppo_clip_coef = 0.1
        self.use_ppo_diffusion = True
        self.kl_target = 0.5
        self.kl_early_stop = 1.0
        self._old_denoise_net = None
        self._last_denoising_chain = None
        self._last_ddim_schedule = None

        self._ddim_rl_steps = 10
        self._ddim_rl_eta = 0.5

        self.waypoints_xy, self.waypoints_s, self.raceline_length = self._load_waypoints(map_name)
        self.last_cumulative_distance = np.zeros(self.num_agents)
        self.last_wp_index = np.zeros(self.num_agents, dtype=np.int32)
        self.start_s = np.zeros(self.num_agents)
        self.current_lap_count = np.zeros(self.num_agents, dtype=int)
        self.last_checkpoint = np.zeros(self.num_agents, dtype=int)

        self.LAP_REWARD = 10.0
        self.CHECKPOINT_REWARD = 1.0
        self.COLLISION_PENALTY = -5.0
        self.PROGRESS_REWARD = 1.0
        self.AGENT_COLLISION_PENALTY = -1.0
        self.NUM_CHECKPOINTS = 10
        self.STEER_RATE_PENALTY = -0.5
        self.STEER_ABS_PENALTY = 0.0
        self.SPEED_BONUS = 0.008
        self.SPEED_BONUS_FLOOR = 3.0
        self.SPEED_BONUS_CAP = 10.0
        self.SPEED_BONUS_STEER_GATE = 0.2
        self._prev_lap_counts = np.zeros(self.num_agents, dtype=int)
        self._was_colliding_wall = np.zeros(self.num_agents, dtype=bool)
        self._was_colliding_agent = np.zeros(self.num_agents, dtype=bool)
        self._last_steer = np.zeros(self.num_agents, dtype=np.float64)

        self._reward_ema_mean = 0.0
        self._reward_ema_var = 1.0

        actor_encoder = self._transfer_vision(transfer[0])
        critic_encoder = self._transfer_vision(transfer[0])

        self.actor_network = DiffusionMamba2(
            state_dim=self.state_dim,
            action_dim=2,
            encoder=actor_encoder,
            num_diffusion_steps=num_diffusion_steps,
            inference_steps=ddim_k,          # DDIM fast sampling for rollout/deploy
            obs_feature_dim=256,
            time_emb_dim=32,
            hidden_dims=(256, 256),
            beta_schedule=beta_schedule,
            d_model=256,
            d_state=128,
            d_conv=4,
            d_head=32,
            expand=2,
            odom_expand=64,
        ).to(self.device)

        self.actor_network.denoise_net.register_dispersive_hooks("late")

        self.critic_network = Mamba2CriticNetwork(
            state_dim=self.state_dim,
            encoder=critic_encoder,
            d_model=256,
            d_state=128,
            d_conv=4,
            d_head=32,
            expand=2,
            odom_expand=64,
        ).to(self.device)

        self.actor_network = self._transfer_weights(transfer[0], self.actor_network)
        self.critic_network = self._transfer_weights(transfer[1], self.critic_network)

        actor_params = sum(p.numel() for p in self.actor_network.parameters())
        actor_train = sum(p.numel() for p in self.actor_network.parameters() if p.requires_grad)
        critic_params = sum(p.numel() for p in self.critic_network.parameters())
        critic_train = sum(p.numel() for p in self.critic_network.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"  D²PPO Agent Initialized")
        print(f"{'='*60}")
        print(f"  Device:          {self.device}")
        print(f"  Num AI agents:   {self.num_agents}")
        print(f"  TBTT length:     {self.tbtt_length}")
        print(f"  Diffusion steps: {self.num_diffusion_steps}")
        print(f"  Actor params:    {actor_params:,} ({actor_train:,} trainable)")
        print(f"  Critic params:   {critic_params:,} ({critic_train:,} trainable)")
        print(f"  Total params:    {actor_params + critic_params:,}")
        print(f"  LR actor:        {self.lr_actor}")
        print(f"  LR critic:       {self.lr_critic}")
        print(f"  Gamma:           {self.gamma}  GAE lambda: {self.gae_lambda}")
        print(f"  AWR temp:        {self.awr_temperature}  max weight: {self.awr_max_weight}")
        print(f"  Diff reg lambda: {self.diff_reg_lambda}")
        print(f"  PPO diffusion:   {self.use_ppo_diffusion}  clip={self.ppo_clip_coef}")
        print(f"  DDIM RL rollout: steps={self._ddim_rl_steps}  eta={self._ddim_rl_eta}")
        print(f"  Dispersive:      lambda={self.dispersive_lambda}  temp={self.dispersive_temperature}  rl={self.use_dispersive_in_rl} (coef_rl={self.dispersive_coef_rl})")
        print(f"{'='*60}\n")

        self.actor_optimizer = optim.AdamW(
            self.actor_network.parameters(), lr=self.lr_actor, weight_decay=0
        )
        self.critic_optimizer = optim.AdamW(
            self.critic_network.parameters(), lr=self.lr_critic, weight_decay=0
        )
        self.buffer = []
        self._last_obs_features = None  # Cached by get_action_and_value for store_transition
        self._pending_transition = None  # Deferred write for next/state_value

        self.POST_WARMUP_RAMP_GENS = int(os.getenv("TR_POST_WARMUP_RAMP_GENS", "2"))
        self._post_warmup_remaining = 0
        self._was_critic_only_last_gen = False

        self.plot_save_path = os.getenv("TR_PLOT_PATH", "plots/d2ppo_training_diagnostics.png")
        plot_dir = os.path.dirname(self.plot_save_path)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        self.diagnostic_keys = [
            "loss_actor", "loss_critic", "loss_diffusion",
            "loss_dispersive", "adv_weight_std", "raw_adv_mean", "raw_adv_pos_frac",
            "actor_grad_norm", "critic_grad_norm", "approx_kl", "actor_skips",
            "clipfrac", "actor_early_stops",
            "collisions", "reward", "avg_speed",
            "dist_per_collision",
            "laps_per_collision",   # track-length-normalised survival
            "speed_ratio",          # avg_speed / raceline_mean_speed (1.0 = expert pace)
            "progress_score",       # laps_per_collision * speed_ratio (combined KPI)
            "action_velocity_mean",  # mean of action[:, 1] across rollout buffer (m/s)
            "action_velocity_min",   # minimum velocity command in the rollout
            "action_steer_abs_mean", # mean |action[:, 0]| (steering aggressiveness)
        ]
        self.diagnostics_history = {key: [] for key in self.diagnostic_keys}
        # Per-generation string/scalar metadata (parallel to diagnostics_history;
        # one dict per generation, indexed by generation_counter - max_len + 1).
        # Used to log focus_map, geometric difficulty, and per-map running
        # dist/coll EMA for offline analysis.  Strings cannot be stored in the
        # arity-3 diagnostic system, so they live here.
        self.per_gen_meta = []
        # Per-map running dist/coll exponential moving average (real maps only;
        # procedural tracks are one-shot so EMA is meaningless).  Used as an
        # empirical difficulty signal and for curriculum analysis.
        self.map_dpc_ema = {}
        self._dpc_ema_alpha = 0.3  # smoothing factor: ~3-gen half-life
        # Latest map_type from set_gen_meta (consumed by _compute_gae for
        # procedural-track reward clamping — bounds value-target magnitude
        # on never-before-seen tracks where one lap-bonus or chained
        # checkpoint can otherwise blow up the critic).
        self._current_map_type = "real"
        self.generation_counter = 0



        self._deploy_mode = False
        self._deploy_action_repeat = 0      # run inference every N sim steps
        self._deploy_ddim_steps = 1         # DDIM steps in deploy mode (paper: 0 intermediate = 1 call)
        self._cached_action = None          # last action for repeating
        self._action_repeat_counter = 0     # counts sim steps since last inference

    def clear_experience_buffer(self):
        self.buffer = []
        self._pending_transition = None

    def reset_reward_ema(self):
        """Reset EMA reward statistics — call after map transitions so the
        normalisation adapts immediately to the new reward distribution."""
        self._reward_ema_mean = 0.0
        self._reward_ema_var = 1.0

    def reset_optimizers(self):
        """Re-create AdamW optimizers from scratch, discarding momentum/variance
        moving averages.  Call on curriculum transitions (between focus maps)
        so stale Adam moments from one map's loss surface don't poison the
        next map's updates.  Mitigates the ``catastrophic-forgetting + KL-
        freeze trap`` observed when the actor's running gradient statistics
        diverge from the new map's distribution."""
        self.actor_optimizer = optim.AdamW(
            self.actor_network.parameters(), lr=self.lr_actor, weight_decay=0
        )
        self.critic_optimizer = optim.AdamW(
            self.critic_network.parameters(), lr=self.lr_critic, weight_decay=0
        )
        print("  [Optimizer reset] AdamW state cleared (actor + critic)")

    def set_gen_meta(self, **kwargs):
        """Record per-generation metadata (focus_map, difficulty, map_type, …)
        for offline analysis.  Call once per generation after the diagnostics
        for that generation have been appended.  Pads with empty dicts for
        warmup gens that may have already been logged.

        Also updates the per-map running dist/coll EMA when a numeric
        ``dist_per_collision`` is supplied alongside ``focus_map`` and
        ``map_type='real'`` (procedural tracks are one-shot).
        """
        # Pad to align with diagnostics_history (warmup gens record diagnostics
        # but might not have called this method).
        max_len = max(
            (len(v) for v in self.diagnostics_history.values()), default=0
        )
        while len(self.per_gen_meta) < max_len - 1:
            self.per_gen_meta.append({})
        self.per_gen_meta.append(dict(kwargs))

        # Update EMA for real maps only.
        focus = kwargs.get("focus_map")
        dpc = kwargs.get("dist_per_collision")
        mtype = kwargs.get("map_type", "real")
        # Stash for _compute_gae (procedural reward clamp).
        self._current_map_type = mtype
        if focus and dpc is not None and mtype == "real":
            try:
                dpc_f = float(dpc)
                if math.isfinite(dpc_f):
                    prev = self.map_dpc_ema.get(focus)
                    if prev is None:
                        self.map_dpc_ema[focus] = dpc_f
                    else:
                        a = self._dpc_ema_alpha
                        self.map_dpc_ema[focus] = (1 - a) * prev + a * dpc_f
                    # Stamp the EMA back onto this generation's meta so the
                    # CSV captures it.
                    self.per_gen_meta[-1]["map_dpc_ema"] = self.map_dpc_ema[focus]
            except (TypeError, ValueError):
                pass

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
        self._deploy_action_repeat = max(0, action_repeat)
        self._deploy_ddim_steps = max(1, ddim_steps)
        self._cached_action = None
        self._action_repeat_counter = 0

        if hasattr(self, 'critic_network'):
            del self.critic_network
            del self.critic_optimizer
            torch.cuda.empty_cache()

        self.actor_network.eval()

        if compile_model:
            try:
                self.actor_network = torch.compile(
                    self.actor_network, mode="reduce-overhead")
                print("[deploy] torch.compile applied (reduce-overhead)")
            except Exception as e:
                print(f"[deploy] torch.compile unavailable: {e}")

        print(f"[deploy] DDIM-{self._deploy_ddim_steps}, "
              f"action_repeat={self._deploy_action_repeat}")

    def reset_temporal_state(self, agent_idxs=None):
        """Reset internal temporal buffers on actor and critic networks.

        Args:
            agent_idxs: optional array of agent indices to reset.
                        If None, resets all agents.
        """
        self.actor_network.reset_temporal_state(agent_idxs)
        if hasattr(self, 'critic_network'):
            self.critic_network.reset_temporal_state(agent_idxs)

    reset_buffers = reset_temporal_state

    def pretrain_critic(self, env, pp_driver, num_agents_total, maps,
                        rollout_steps=512, num_rollouts=3, epochs=10,
                        lr=1e-4, batch_size=256,
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
            lr=lr, weight_decay=0,
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
            # --- Parallel multi-env collection ---
            # Run up to `concurrent_envs` gym envs on separate CPU cores,
            # batching the actor forward pass on GPU each step.
            concurrent_envs = min(num_rollouts, 16)
            all_maps = [maps[i % len(maps)] for i in range(num_rollouts)]

            # Process maps in batches of `concurrent_envs`
            for batch_start in range(0, len(all_maps), concurrent_envs):
                batch_maps = all_maps[batch_start:batch_start + concurrent_envs]
                E = len(batch_maps)
                print(f"\n  Batch {batch_start // concurrent_envs + 1}: "
                      f"Collecting {E} parallel rollouts on {batch_maps}")

                # Per-env reward tracking state
                env_wp_xy = []
                env_wp_s = []
                env_rl = []
                env_last_cum_dist = []
                env_last_wp_idx = []
                env_last_ckpt = []
                env_prev_laps = []

                # Spawn worker processes (fork to avoid re-importing modules)
                ctx = _mp.get_context("fork")
                parent_conns = []
                workers = []
                init_poses_list = []

                for e, mname in enumerate(batch_maps):
                    parent_conn, child_conn = ctx.Pipe()
                    p = ctx.Process(
                        target=D2PPOAgent._critic_env_worker,
                        args=(child_conn, mname, num_agents_total,
                              self.num_agents, 32,
                              self.num_scan_beams, self.lidar_fov, self.params),
                        daemon=True,
                    )
                    p.start()
                    child_conn.close()
                    parent_conns.append(parent_conn)
                    workers.append(p)

                # Receive initial observations from all workers
                all_obs_data = []
                for e, conn in enumerate(parent_conns):
                    tag, obs_data, poses = conn.recv()
                    all_obs_data.append(obs_data)
                    init_poses_list.append(poses)

                    # Load waypoints (inline to avoid mpc side-effect)
                    wf = f"maps/{batch_maps[e]}/{batch_maps[e]}_raceline.csv"
                    _wp = np.loadtxt(wf, delimiter=";")
                    wp_xy = _wp[:, 1:3]
                    _d = np.sqrt(np.sum(np.diff(wp_xy, axis=0) ** 2, axis=1))
                    wp_s = np.insert(np.cumsum(_d), 0, 0)
                    rl = wp_s[-1]
                    env_wp_xy.append(wp_xy)
                    env_wp_s.append(wp_s)
                    env_rl.append(rl)
                    # Init per-env progress trackers
                    cum_dist = np.zeros(self.num_agents)
                    wp_idx = np.zeros(self.num_agents, dtype=np.int32)
                    for i in range(self.num_agents):
                        dists = np.linalg.norm(wp_xy - poses[i, :2], axis=1)
                        closest = np.argmin(dists)
                        cum_dist[i] = wp_s[closest]
                        wp_idx[i] = closest
                    env_last_cum_dist.append(cum_dist)
                    env_last_wp_idx.append(wp_idx)
                    env_last_ckpt.append(np.zeros(self.num_agents, dtype=int))
                    env_prev_laps.append(np.zeros(self.num_agents, dtype=int))

                # Force re-allocation of internal state for the batch
                self.actor_network._conv_state = None
                self.actor_network._ssm_state = None

                # Per-env rollout storage
                env_rollout_scans = [[] for _ in range(E)]
                env_rollout_states = [[] for _ in range(E)]
                env_rollout_rewards = [[] for _ in range(E)]
                env_rollout_dones = [[] for _ in range(E)]

                for step in range(rollout_steps):
                    # --- Build batched scan/state tensors for all envs ---
                    scan_list = []
                    state_list = []
                    for e, obs_d in enumerate(all_obs_data):
                        scans = obs_d["scans"][:self.num_agents]
                        scan_t = torch.from_numpy(
                            scans.astype(np.float32)).unsqueeze(1)
                        lvx = obs_d["linear_vels_x"][:self.num_agents]
                        lvy = obs_d["linear_vels_y"][:self.num_agents]
                        avz = obs_d["ang_vels_z"][:self.num_agents]
                        state_data = np.stack((lvx, lvy, avz), axis=1)
                        state_t = torch.from_numpy(
                            state_data.astype(np.float32))
                        scan_list.append(scan_t)
                        state_list.append(state_t)

                    batch_scans = torch.cat(scan_list, dim=0).to(self.device)
                    batch_states = torch.cat(
                        state_list, dim=0).to(self.device)

                    with torch.no_grad():
                        obs_features = self.actor_network.encode_observation(
                            batch_scans, batch_states)
                        obs_f32 = obs_features.float()
                        action = self.actor_network.ddim_sample_action(
                            obs_f32, num_steps=5, eta=0.5)

                    action_np = action.cpu().numpy()

                    # Send actions to all workers (they step in parallel)
                    for e in range(E):
                        off = e * self.num_agents
                        parent_conns[e].send(
                            action_np[off:off + self.num_agents])

                    # Receive results from all workers
                    for e in range(E):
                        tag, next_obs_d, stuck, new_poses = \
                            parent_conns[e].recv()

                        # Store pre-step scans/states
                        env_rollout_scans[e].append(scan_list[e].clone())
                        env_rollout_states[e].append(state_list[e].clone())

                        # Calculate rewards (per-env waypoint state)
                        collisions = next_obs_d["collisions"][:self.num_agents]
                        positions = np.stack([
                            next_obs_d["poses_x"][:self.num_agents],
                            next_obs_d["poses_y"][:self.num_agents],
                        ], axis=1)
                        lap_counts = next_obs_d[
                            "lap_counts"][:self.num_agents].astype(int)

                        rewards = np.zeros(self.num_agents)
                        wp_xy = env_wp_xy[e]
                        wp_s = env_wp_s[e]
                        rl_len = env_rl[e]
                        segment_len = rl_len / self.NUM_CHECKPOINTS
                        wp_count = len(wp_xy)

                        for i in range(self.num_agents):
                            start_idx = env_last_wp_idx[e][i]
                            search_indices = np.arange(
                                start_idx, start_idx + 50) % wp_count
                            dists = np.linalg.norm(
                                wp_xy[search_indices] - positions[i],
                                axis=1)
                            closest_local = np.argmin(dists)
                            closest_global = search_indices[closest_local]
                            prev_idx = (
                                closest_global - 1 + wp_count) % wp_count
                            V = wp_xy[closest_global] - wp_xy[prev_idx]
                            V_len_sq = np.dot(V, V)
                            W = positions[i] - wp_xy[prev_idx]
                            L = (np.dot(W, V) / V_len_sq
                                 if V_len_sq > 1e-6 else 0.0)
                            s_prev = wp_s[prev_idx]
                            s_curr = wp_s[closest_global]
                            seg_dist = s_curr - s_prev
                            if seg_dist < 0:
                                seg_dist += rl_len
                            projected_s = s_prev + L * seg_dist

                            if np.isnan(projected_s):
                                projected_s = env_last_cum_dist[e][i]
                                closest_global = env_last_wp_idx[e][i]

                            delta_s = projected_s - env_last_cum_dist[e][i]
                            if delta_s < -rl_len * 0.5:
                                delta_s += rl_len
                            elif delta_s > rl_len * 0.5:
                                delta_s -= rl_len
                            if delta_s > 0:
                                delta_s /= self.raceline_length
                                rewards[i] += delta_s * self.PROGRESS_REWARD

                            new_ckpt = (int(projected_s / segment_len)
                                        % self.NUM_CHECKPOINTS)
                            if new_ckpt != env_last_ckpt[e][i]:
                                rewards[i] += self.CHECKPOINT_REWARD
                                env_last_ckpt[e][i] = new_ckpt

                            env_last_cum_dist[e][i] = projected_s
                            env_last_wp_idx[e][i] = closest_global

                        laps_completed = lap_counts - env_prev_laps[e]
                        rewards += (np.clip(laps_completed, 0, 1)
                                    * self.LAP_REWARD)
                        env_prev_laps[e] = lap_counts.copy()

                        rewards += ((collisions == 1)
                                    * self.COLLISION_PENALTY)
                        rewards += ((collisions == 2)
                                    * self.AGENT_COLLISION_PENALTY)

                        rew_t = torch.from_numpy(
                            rewards.astype(np.float32)).unsqueeze(-1)
                        env_rollout_rewards[e].append(rew_t)

                        done_t = torch.zeros(
                            self.num_agents, dtype=torch.float32)
                        if len(stuck) > 0:
                            ai_stuck = stuck[stuck < self.num_agents]
                            if len(ai_stuck) > 0:
                                done_t[ai_stuck] = 1.0
                                global_idxs = (ai_stuck
                                               + e * self.num_agents)
                                self.actor_network.reset_temporal_state(
                                    global_idxs)
                                if new_poses is not None:
                                    for idx in ai_stuck:
                                        d = np.linalg.norm(
                                            wp_xy - new_poses[idx, :2],
                                            axis=1)
                                        c = np.argmin(d)
                                        env_last_cum_dist[e][idx] = wp_s[c]
                                        env_last_wp_idx[e][idx] = c
                                        env_last_ckpt[e][idx] = 0
                                        env_prev_laps[e][idx] = 0
                        env_rollout_dones[e].append(done_t)

                        all_obs_data[e] = next_obs_d

                    if (step + 1) % 500 == 0:
                        print(f"    step {step + 1}/{rollout_steps}",
                              end='\r')
                        torch.cuda.synchronize()

                print()

                # Shutdown workers for this batch
                for conn in parent_conns:
                    conn.send(None)
                    conn.close()
                for p in workers:
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()

                # Restore actor temporal state
                self.actor_network._conv_state = None
                self.actor_network._ssm_state = None

                # Process each env's rollout into returns
                for e in range(E):
                    if not env_rollout_rewards[e]:
                        continue
                    T = len(env_rollout_rewards[e])
                    rewards_arr = torch.stack(
                        env_rollout_rewards[e]).squeeze(-1)
                    dones_arr = torch.stack(
                        env_rollout_dones[e]).squeeze(-1)
                    r_mean = rewards_arr.mean()
                    r_std = rewards_arr.std().clamp(min=1e-4)
                    rewards_arr = (rewards_arr - r_mean) / r_std

                    returns = torch.zeros_like(rewards_arr)
                    G = torch.zeros(self.num_agents)
                    for t in reversed(range(T)):
                        G = (rewards_arr[t]
                             + self.gamma * G * (1.0 - dones_arr[t]))
                        returns[t] = G

                    scans_flat = torch.cat(env_rollout_scans[e], dim=0)
                    states_flat = torch.cat(env_rollout_states[e], dim=0)
                    returns_flat = returns.reshape(-1)

                    all_scans.append(scans_flat)
                    all_states.append(states_flat)
                    all_returns.append(returns_flat)

                    print(f"  \u2713 {batch_maps[e]}: {T} steps, "
                          f"{scans_flat.shape[0]} samples")

            # --- Combine all rollout data ---
            X_scans = torch.cat(all_scans, dim=0).to(self.device)
            X_states = torch.cat(all_states, dim=0).to(self.device)
            Y_returns = torch.cat(all_returns, dim=0).to(self.device)

            # --- Save critic demos ---
            N = X_scans.shape[0]
            if save_demos_path:
                os.makedirs(os.path.dirname(save_demos_path) or ".", exist_ok=True)
                torch.save({
                    "scans": X_scans.cpu(),
                    "states": X_states.cpu(),
                    "returns": Y_returns.cpu(),
                }, save_demos_path)
                print(f"  Saved critic demos ({N} samples) \u2192 {save_demos_path}")

        N = X_scans.shape[0]
        X_scans = X_scans.to(self.device)
        X_states = X_states.to(self.device)
        Y_returns = Y_returns.to(self.device)
        print(f"  Collected {N} samples for critic pretraining."
              f"  Returns: mean={Y_returns.mean():.2f}, std={Y_returns.std():.2f}")

        # Normalize returns to zero-mean unit-variance so critic targets
        # are in a learnable range (matches RL-time EMA normalisation).
        ret_mean = Y_returns.mean()
        ret_std = Y_returns.std().clamp(min=1e-4)
        Y_returns = (Y_returns - ret_mean) / ret_std
        print(f"  Normalized returns: mean={Y_returns.mean():.4f}, std={Y_returns.std():.4f}")

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

                # Forward: encode single frame (fresh zero state = no temporal context)
                _conv, _ssm = self.critic_network.allocate_state(s.shape[0], self.device)
                feat, _, _ = self.critic_network.encode_observation(s, st, conv_state=_conv, ssm_state=_ssm)
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
            self.critic_network.parameters(), lr=self.lr_critic, weight_decay=0,
        )
        # Save pretrained critic weights
        critic_save_dir = "models/critic/pretrained"
        os.makedirs(critic_save_dir, exist_ok=True)
        critic_save_path = os.path.join(critic_save_dir, "critic_pretrained.pt")
        torch.save(self.critic_network.state_dict(), critic_save_path)
        print(f"  Saved critic weights \u2192 {critic_save_path}")

        # Reset temporal state for clean start
        self.reset_temporal_state()

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
            obs_features = self.actor_network.encode_observation(
                scan_tensor[: self.num_agents],
                state_tensor[: self.num_agents],
            )

            # obs_features are already float32 (no autocast)
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
                if self.use_ppo_diffusion:
                    # DDIM reverse process with inline log-prob and chain
                    # storage — D2PPO uses DDIM (not full DDPM) for RL.
                    # eta > 0 gives stochastic transitions with tractable
                    # Gaussian log-probs for PPO importance ratios.
                    action, chain, log_prob, ddim_sched = \
                        self.actor_network.ddim_sample_action_with_chain(
                            obs_features_f32,
                            num_steps=self._ddim_rl_steps,
                            eta=self._ddim_rl_eta)
                    # Store chain in normalised [-1,1] space for learning.
                    # chain is a list of (B, action_dim) tensors, length S+1.
                    # Stack into (S+1, B, action_dim) and cache.
                    self._last_denoising_chain = torch.stack(chain, dim=0)
                    self._last_ddim_schedule = ddim_sched
                else:
                    # DDIM-5: fast sampling for legacy AWR mode.
                    action = self.actor_network.ddim_sample_action(
                        obs_features_f32, num_steps=5, eta=0.5
                    )
                    log_prob = None
                    self._last_denoising_chain = None
                    self._last_ddim_schedule = None

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
            obs_features = self.actor_network.encode_observation(
                scan_tensor[: self.num_agents],
                state_tensor[: self.num_agents],
            )

            if need_new_action:
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

        # Map metadata stored per-step so map-agnostic metrics (laps_per_collision,
        # speed_ratio, progress_score) work correctly when MAPS_PER_GEN > 1.
        _track_len = float(getattr(self, "raceline_length", 1.0)) or 1.0
        _race_v    = float(getattr(self, "raceline_mean_speed", 6.0)) or 6.0
        track_len_t = torch.full((self.num_agents,), _track_len,
                                 dtype=torch.float32, device=value.device)
        raceline_v_t = torch.full((self.num_agents,), _race_v,
                                  dtype=torch.float32, device=value.device)

        td_dict = {
                "observation_scan": obs[0],
                "observation_state": obs[1],
                "obs_features": self._last_obs_features,
                "critic_features": self._last_critic_features,
                "action": action,
                "action_log_prob": log_prob,
                "state_value": value,
                "track_len": track_len_t,
                "raceline_v": raceline_v_t,
                "next": TensorDict(
                    {
                        "state_value": torch.zeros_like(value),  # placeholder
                        "reward": reward,
                        "done": done_tensor,
                    }
                ),
        }
        # Store denoising chain for D2PPO (shape: A, S+1, action_dim) where S = DDIM steps
        if self._last_denoising_chain is not None:
            # _last_denoising_chain is (S+1, A, action_dim); transpose for TensorDict batch dim
            td_dict["denoising_chain"] = self._last_denoising_chain.permute(1, 0, 2)

        step_data = TensorDict(td_dict, batch_size=[self.num_agents])
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

    def calculate_reward(self, next_obs, action=None):
        collisions = np.array(next_obs["collisions"][:self.num_agents])
        speeds = np.array(next_obs["linear_vels_x"][:self.num_agents])
        positions = np.stack([
            np.array(next_obs['poses_x'][:self.num_agents]),
            np.array(next_obs['poses_y'][:self.num_agents]),
        ], axis=1)
        wall_collisions = collisions == 1
        agent_collisions = collisions == 2
        rewards = np.zeros(self.num_agents)

        progress_r = np.zeros(self.num_agents)
        checkpoint_r = np.zeros(self.num_agents)
        steer_r = np.zeros(self.num_agents)
        steer_abs_r = np.zeros(self.num_agents)
        speed_r = np.zeros(self.num_agents)

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
            # Only reward forward progress, don't penalise backward (collision resets).
            # Speed-scaled bonus: reward per meter grows with instantaneous speed,
            # so the policy cannot maximise return by driving slowly and safely.
            # Shape: (0.5 + v/v_ref) -> 0.5x at v=0, 1.5x at expert pace, 2.5x at 2x pace.
            if delta_s > 0:
                progress_r[i] = delta_s * self.PROGRESS_REWARD
                rewards[i] += progress_r[i]

            # --- Checkpoint reward (divide track into NUM_CHECKPOINTS segments) ---
            segment_len = self.raceline_length / self.NUM_CHECKPOINTS
            new_ckpt = int(projected_s / segment_len) % self.NUM_CHECKPOINTS
            if new_ckpt != self.last_checkpoint[i]:
                checkpoint_r[i] = self.CHECKPOINT_REWARD
                rewards[i] += checkpoint_r[i]
                self.last_checkpoint[i] = new_ckpt

            self.last_cumulative_distance[i] = projected_s
            self.last_wp_index[i] = new_wp_idx

        # --- LAP COMPLETION REWARD ---
        lap_counts = np.array(next_obs['lap_counts'][:self.num_agents], dtype=int)
        laps_completed = lap_counts - self._prev_lap_counts
        lap_r = np.clip(laps_completed, 0, 1) * self.LAP_REWARD
        rewards += lap_r
        self._prev_lap_counts = lap_counts.copy()

        # --- COLLISION PENALTIES (one-shot: only on NEW collision events) ---
        new_wall = wall_collisions & ~self._was_colliding_wall
        new_agent = agent_collisions & ~self._was_colliding_agent
        wall_r = new_wall.astype(np.float64) * self.COLLISION_PENALTY
        agent_r = new_agent.astype(np.float64) * self.AGENT_COLLISION_PENALTY
        self._was_colliding_wall = wall_collisions.copy()
        self._was_colliding_agent = agent_collisions.copy()
        rewards += wall_r
        rewards += agent_r

        # --- STEER-RATE (wiggle) + STEER-ABS PENALTY + SPEED-FLOOR BONUS ---
        # All three shape the policy toward smoother, faster driving.
        # Skipped for any agent that just collided this step so the
        # collision signal stays clean.
        if action is not None and (
            self.STEER_RATE_PENALTY != 0.0 or self.STEER_ABS_PENALTY != 0.0
        ):
            try:
                act_arr = np.asarray(action)[:self.num_agents]
                if act_arr.ndim >= 2 and act_arr.shape[-1] >= 2:
                    steer_chunk = act_arr[..., 0].astype(np.float64)
                    if steer_chunk.ndim > 1:
                        # Action chunk: |steer| averaged over the chunk
                        # so within-chunk magnitude is also penalised.
                        steer_now = steer_chunk[:, -1]
                        steer_abs_mean = np.mean(np.abs(steer_chunk), axis=1)
                    else:
                        steer_now = steer_chunk
                        steer_abs_mean = np.abs(steer_chunk)
                    delta_steer = np.abs(steer_now - self._last_steer)
                    not_collided = ~(wall_collisions | agent_collisions)
                    not_collided_f = not_collided.astype(np.float64)
                    steer_r = (
                        self.STEER_RATE_PENALTY
                        * delta_steer
                        * not_collided_f
                    )
                    steer_abs_r = (
                        self.STEER_ABS_PENALTY
                        * steer_abs_mean
                        * not_collided_f
                    )
                    rewards += steer_r
                    rewards += steer_abs_r
                    self._last_steer = steer_now.copy()
            except (TypeError, ValueError, IndexError):
                pass

        if self.SPEED_BONUS > 0.0:
            v = np.clip(speeds.astype(np.float64), 0.0, self.SPEED_BONUS_CAP)
            above = np.clip(v - self.SPEED_BONUS_FLOOR, 0.0, None)
            forward = progress_r > 0.0
            not_collided = ~(wall_collisions | agent_collisions)
            # Curvature gate: |self._last_steer| was just updated to the
            # current step's steer (or stays 0 if action is None).  Bonus
            # ramps linearly from 1 at |steer|=0 down to 0 at the gate.
            gate = np.clip(
                1.0 - np.abs(self._last_steer) / max(self.SPEED_BONUS_STEER_GATE, 1e-6),
                0.0,
                1.0,
            )
            speed_r = (
                self.SPEED_BONUS
                * above
                * gate
                * forward.astype(np.float64)
                * not_collided.astype(np.float64)
            )
            rewards += speed_r

        rewards_tensor = torch.from_numpy(rewards.astype(np.float32)).unsqueeze(-1)
        avg_reward = rewards.mean()

        # Per-component means (across agents) for diagnostics
        self._reward_components = {
            "progress": float(progress_r.mean()),
            "checkpoint": float(checkpoint_r.mean()),
            "lap": float(lap_r.mean()),
            "wall_col": float(wall_r.mean()),
            "agent_col": float(agent_r.mean()),
            "steer_rate": float(steer_r.mean()),
            "steer_abs": float(steer_abs_r.mean()),
            "speed_bonus": float(speed_r.mean()),
        }

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
                self._was_colliding_wall[i] = False
                self._was_colliding_agent[i] = False
                self._last_steer[i] = 0.0
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
        self._was_colliding_wall[:] = False
        self._was_colliding_agent[:] = False
        self._last_steer[:] = 0.0

    def learn(self, collisions, reward, critic_only=False):
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

        Args:
            critic_only: If True, only update the critic (skip actor gradients).
                         Used for warmup after map transitions.
        """
        print("Starting PPO-Diffusion learning..." if self.use_ppo_diffusion else "Starting AWR-Diffusion learning...")
        print(f"  Buffer size: {len(self.buffer)}")

        if len(self.buffer) < 4:
            print("  Skipping learning — buffer too short (early collision).")
            self.generation_counter += 1
            for key in self.diagnostic_keys:
                self.diagnostics_history[key].append((np.nan, np.nan, np.nan))
            if self.generation_counter > 0 and (self.generation_counter % 5 == 0):
                self._plot_historical_diagnostics()
            self.buffer.clear()
            return

        # Stack in insertion order — temporal ordering is critical for GAE.
        data = torch.stack(self.buffer).contiguous()

        current_gen_diagnostics = {key: [] for key in self.diagnostic_keys}
        current_gen_diagnostics["collisions"] = [collisions]
        current_gen_diagnostics["reward"] = [reward]

        # Distance-per-collision: estimate total forward distance from speeds.
        # observation_state[:, 0] is linear_vel_x (m/s, pre-normalization).
        speeds = data.get("observation_state")[:, :, 0].clamp(min=0.0)  # (T, N_agents)
        total_distance = (speeds * 0.01).sum().item()  # metres (dt = 0.01s)
        dpc = total_distance / max(collisions, 1)
        current_gen_diagnostics["dist_per_collision"] = [dpc]

        # --- Map-agnostic metrics ---
        # Compute per-step using the track_len / raceline_v that were active
        # at each transition (tagged in store_transition). This makes the
        # metrics correct even when MAPS_PER_GEN > 1 rotates maps mid-gen.
        try:
            track_len_per_step = data.get("track_len")          # (T, A)
            raceline_v_per_step = data.get("raceline_v")        # (T, A)
            # laps contributed by each (t, agent) sample = speed*dt / track_len_at_step
            step_dist = speeds * 0.01                            # (T, A)
            step_laps = (step_dist / track_len_per_step.clamp(min=1e-6)).sum().item()
            lpc = step_laps / max(collisions, 1)
            current_gen_diagnostics["laps_per_collision"] = [lpc]

            # speed_ratio: distance-weighted mean of (speed / raceline_v_at_step)
            ratio_per_step = speeds / raceline_v_per_step.clamp(min=1e-6)
            sr = float(ratio_per_step.mean().item())
            current_gen_diagnostics["speed_ratio"] = [sr]

            current_gen_diagnostics["progress_score"] = [lpc * sr]
        except KeyError:
            # Old buffers (resumed checkpoints) lack the metadata.
            current_gen_diagnostics["laps_per_collision"] = [np.nan]
            current_gen_diagnostics["speed_ratio"] = [np.nan]
            current_gen_diagnostics["progress_score"] = [np.nan]

        # --- Action-distribution sanity ---
        # Surface the policy's *commanded* velocity (not just realised speed)
        # so a "drive-slow" attractor shows up immediately rather than after
        # hundreds of generations of degrading dist_per_collision.  Action
        # layout: [steer, velocity].
        try:
            actions_t = data.get("action")  # (T, A, 2)
            if actions_t is not None and actions_t.numel() > 0:
                vel = actions_t[..., 1].float()
                steer = actions_t[..., 0].float()
                current_gen_diagnostics["action_velocity_mean"] = [
                    float(vel.mean().item())]
                current_gen_diagnostics["action_velocity_min"] = [
                    float(vel.min().item())]
                current_gen_diagnostics["action_steer_abs_mean"] = [
                    float(steer.abs().mean().item())]
            else:
                current_gen_diagnostics["action_velocity_mean"] = [np.nan]
                current_gen_diagnostics["action_velocity_min"] = [np.nan]
                current_gen_diagnostics["action_steer_abs_mean"] = [np.nan]
        except Exception:
            current_gen_diagnostics["action_velocity_mean"] = [np.nan]
            current_gen_diagnostics["action_velocity_min"] = [np.nan]
            current_gen_diagnostics["action_steer_abs_mean"] = [np.nan]

        # Compute GAE
        with torch.no_grad():
            data = self._compute_gae(data)

        # Snapshot the denoising MLP for PPO importance ratio
        if self.use_ppo_diffusion:
            self._old_denoise_net = copy.deepcopy(
                self.actor_network.denoise_net
            )
            self._old_denoise_net.eval()
            for p in self._old_denoise_net.parameters():
                p.requires_grad = False

        # --- Post-warmup actor LR ramp ---
        # If the previous gen was a critic-only warmup and this one is not,
        # start a multi-gen LR ramp on the actor.
        if self._was_critic_only_last_gen and not critic_only:
            self._post_warmup_remaining = self.POST_WARMUP_RAMP_GENS
            print(f"  [post-warmup] actor LR ramp: {self.POST_WARMUP_RAMP_GENS} gens")
        self._was_critic_only_last_gen = critic_only

        _orig_actor_lr = None
        if (not critic_only) and self._post_warmup_remaining > 0:
            # Schedule: gen N -> 0.25x, gen N-1 -> 0.5x, ..., 1.0x at end
            ramp_idx = self.POST_WARMUP_RAMP_GENS - self._post_warmup_remaining
            scale = 0.25 * (2 ** ramp_idx)  # 0.25, 0.5, 1.0, ...
            scale = min(scale, 1.0)
            _orig_actor_lr = [g['lr'] for g in self.actor_optimizer.param_groups]
            for g in self.actor_optimizer.param_groups:
                g['lr'] = g['lr'] * scale
            print(f"  [post-warmup] actor LR scaled x{scale:.2f} "
                  f"({self._post_warmup_remaining} gen(s) remaining)")
            self._post_warmup_remaining -= 1

        # Dispatch to TBTT or shuffled-minibatch learner
        if self.tbtt_length > 0:
            self._learn_tbtt(data, current_gen_diagnostics,
                             critic_only=critic_only)
        else:
            self._learn_shuffled(data, current_gen_diagnostics,
                                 critic_only=critic_only)

        # Restore actor LR after post-warmup-scaled gen
        if _orig_actor_lr is not None:
            for g, lr in zip(self.actor_optimizer.param_groups, _orig_actor_lr):
                g['lr'] = lr

        # --- Post-training bookkeeping ---
        self.generation_counter += 1
        # During critic warmup, only the critic/diffusion/dispersive losses
        # are meaningful — everything else (rewards, collisions, dpc, speed,
        # actor losses/grads/kl/skips, advantage stats) is frozen or noise.
        _warmup_report_keys = {"loss_critic", "loss_diffusion", "loss_dispersive"}
        for key in self.diagnostic_keys:
            values = current_gen_diagnostics.get(key)
            if critic_only and key not in _warmup_report_keys:
                # Append NaN placeholder so generation indices stay aligned
                self.diagnostics_history[key].append((np.nan, np.nan, np.nan))
            elif values:
                avg_val = np.mean(values)
                min_val = np.min(values)
                max_val = np.max(values)
                self.diagnostics_history[key].append((avg_val, min_val, max_val))
            else:
                self.diagnostics_history[key].append((np.nan, np.nan, np.nan))

        # Throttle plotting — it's cheap early but grows O(gens) over time.
        if self.generation_counter > 0:
            self._plot_historical_diagnostics()

        self.buffer.clear()
        del data
        # Free the old denoising MLP snapshot
        if self._old_denoise_net is not None:
            del self._old_denoise_net
            self._old_denoise_net = None
        torch.cuda.empty_cache()
        print("[D²PPO Stage 2] Learning complete.")

    # ------------------------------------------------------------------
    # Shuffled-minibatch learner (original — no encoder gradients)
    # ------------------------------------------------------------------
    def _learn_shuffled(self, data, current_gen_diagnostics, critic_only=False):
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
                raw_adv_mean = adv.mean()
                raw_adv_pos_frac = (adv > 0).float().mean()
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
                feat_list = self.actor_network.denoise_net.get_intermediate_features()
                disp_loss = torch.tensor(0.0, device=self.device)
                n_feats = 0
                for feats in feat_list:
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

                disp_weight = self.dispersive_coef_rl if self.use_dispersive_in_rl else 0.0
                actor_loss = (
                    awr_loss
                    + self.diff_reg_lambda * diff_reg
                    + disp_weight * disp_loss
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
                actor_loss.backward()
                actor_gn = nn.utils.clip_grad_norm_(
                    self.actor_network.parameters(), self.max_grad_norm_actor
                )
                if torch.isfinite(actor_gn):
                    self.actor_optimizer.step()
                else:
                    # Non-finite grads would poison the weights via Adam state.
                    # Drop this minibatch's actor update.
                    self.actor_optimizer.zero_grad(set_to_none=True)

                critic_loss.backward()
                critic_gn = nn.utils.clip_grad_norm_(
                    self.critic_network.parameters(), self.max_grad_norm_critic
                )
                self.critic_optimizer.step()

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
                current_gen_diagnostics["raw_adv_mean"].append(raw_adv_mean.item())
                current_gen_diagnostics["raw_adv_pos_frac"].append(raw_adv_pos_frac.item())
                current_gen_diagnostics["actor_grad_norm"].append(float(actor_gn))
                current_gen_diagnostics["critic_grad_norm"].append(float(critic_gn))
                current_gen_diagnostics["approx_kl"].append(float("nan"))
                current_gen_diagnostics["actor_skips"].append(0.0)
                current_gen_diagnostics["avg_speed"].append(float(speeds.mean()) if hasattr(speeds, 'mean') else float(speeds))

            if num_updates > 0:
                avg_a = epoch_actor_loss / num_updates
                avg_d = epoch_diff_loss / num_updates
                avg_dp = epoch_disp_loss / num_updates
                _dw = self.dispersive_coef_rl if self.use_dispersive_in_rl else 0.0
                awr_comp = avg_a - self.diff_reg_lambda * avg_d - _dw * avg_dp
                disp_comp = abs(_dw * avg_dp)
                print(
                    f"  Epoch {epoch+1}/{self.epochs}: "
                    f"Actor={avg_a:.4f}, "
                    f"Critic={epoch_critic_loss/num_updates:.4f}, "
                    f"Diff={avg_d:.4f}, "
                    f"Disp={avg_dp:.4f} (rl={'ON' if self.use_dispersive_in_rl else 'OFF'}), "
                )
                if epoch == 0:
                    print(f"    Gradient budget: AWR={awr_comp:.4f}  "
                          f"Dispersive={disp_comp:.4f}  "
                          f"ratio={disp_comp/max(abs(awr_comp), 1e-8):.1f}x")

            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # TBTT learner — re-encodes through Mamba2 with temporal context
    # ------------------------------------------------------------------
    def _learn_tbtt(self, data, current_gen_diagnostics, critic_only=False):
        """TBTT variant: re-encode observations through Mamba2 with gradients.

        Processes the rollout buffer sequentially so the Mamba2 temporal
        backbone receives gradient signal through its recurrent SSM state,
        allowing it to learn temporal representations during RL fine-tuning.

        SSM state ``(conv_state, ssm_state)`` is detached at TBTT chunk
        boundaries to bound the backward graph while preserving temporal
        context (values carry forward, only gradients are truncated).

        Args:
            critic_only: If True, only update the critic (skip actor backward/step).
        """
        obs_scan_all = data["observation_scan"].to(self.device)     # (T, A, 1, beams)
        obs_state_all = data["observation_state"].to(self.device)   # (T, A, state_dim)
        actions_all = data["action"].to(self.device)                # (T, A, 2)
        raw_advantages_all = data["raw_advantage"].to(self.device)  # (T, A)
        value_targets_all = data["value_target"].to(self.device)    # (T, A)
        old_values_all = data["state_value"].to(self.device)        # (T, A)
        dones_all = data.get(("next", "done")).float().to(self.device)  # (T, A) or (T, A, 1)
        if dones_all.ndim == 3:
            dones_all = dones_all.squeeze(-1)                       # (T, A)

        # Denoising chain for D2PPO: (T, A, S+1, action_dim) — DDIM chain
        has_chain = "denoising_chain" in data.keys()
        if has_chain:
            chains_all = data["denoising_chain"].to(self.device)    # (T, A, S+1, 2)

        # DDIM schedule is fixed across the rollout (stored on agent)
        ddim_schedule = self.actor_network._ddim_timestep_schedule(self._ddim_rl_steps)
        S_ddim = len(ddim_schedule)  # number of DDIM steps

        T = len(data)
        A = self.num_agents
        K = self.num_diffusion_steps
        tbtt_len = self.tbtt_length
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

        mode_str = "PPO-Diffusion" if self.use_ppo_diffusion else "AWR"
        print(f"  Training (TBTT, {mode_str}): {self.epochs} epochs, {T} timesteps, "
              f"chunk={tbtt_len}, {num_chunks} chunks/epoch")

        for epoch in range(self.epochs):
            # Fresh recurrent state each epoch
            actor_conv, actor_ssm = self.actor_network.allocate_state(A, self.device)
            critic_conv, critic_ssm = self.critic_network.allocate_state(A, self.device)

            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            epoch_diff_loss = 0.0
            epoch_disp_loss = 0.0
            num_updates = 0

            # --- KL-adaptive early stop ---
            # Once the running-mean chunk KL on this epoch exceeds
            # `self.kl_early_stop`, freeze further actor updates for the
            # remainder of this epoch. Critic still updates.
            epoch_kl_running = 0.0
            epoch_kl_count = 0
            early_stopped_this_epoch = False

            for chunk_start in range(0, T, tbtt_len):
                chunk_end = min(chunk_start + tbtt_len, T)
                chunk_len = chunk_end - chunk_start

                # Detach SSM state at chunk boundary (truncated BPTT)
                actor_conv = actor_conv.detach().clone()
                actor_ssm = actor_ssm.detach().clone()
                critic_conv = critic_conv.detach().clone()
                critic_ssm = critic_ssm.detach().clone()

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                chunk_actor = torch.tensor(0.0, device=self.device)
                chunk_critic = torch.tensor(0.0, device=self.device)
                chunk_diff = torch.tensor(0.0, device=self.device)
                chunk_disp = torch.tensor(0.0, device=self.device)
                chunk_wstd = torch.tensor(0.0, device=self.device)
                chunk_kl = torch.tensor(0.0, device=self.device)
                chunk_clipfrac = torch.tensor(0.0, device=self.device)
                chunk_raw_adv_mean = torch.tensor(0.0, device=self.device)
                chunk_raw_adv_pos_frac = torch.tensor(0.0, device=self.device)

                for t_idx in range(chunk_start, chunk_end):
                    # Zero SSM state for agents that were reset
                    # at the previous step (done[t-1] == 1 means
                    # observation at t is from a fresh episode).
                    if t_idx > 0:
                        done_mask = dones_all[t_idx - 1]  # (A,)
                        if done_mask.any():
                            reset_idx = done_mask.nonzero(
                                as_tuple=False).squeeze(-1)
                            actor_conv = actor_conv.clone()
                            actor_ssm = actor_ssm.clone()
                            critic_conv = critic_conv.clone()
                            critic_ssm = critic_ssm.clone()
                            actor_conv[reset_idx] = 0.0
                            actor_ssm[reset_idx] = 0.0
                            critic_conv[reset_idx] = 0.0
                            critic_ssm[reset_idx] = 0.0

                    obs_feat, actor_conv, actor_ssm = \
                        self.actor_network.encode_observation(
                            obs_scan_all[t_idx],
                            obs_state_all[t_idx],
                            actor_conv, actor_ssm,
                        )
                    crit_feat, critic_conv, critic_ssm = \
                        self.critic_network.encode_observation(
                            obs_scan_all[t_idx],
                            obs_state_all[t_idx],
                            critic_conv, critic_ssm,
                        )
                    obs_feat_f = obs_feat.float()
                    crit_feat_f = crit_feat.float()

                    # Actor policy loss (PPO or AWR)
                    act_norm = self.actor_network.normalize_action(
                        actions_all[t_idx])
                    # Diffusion MSE loss for regularisation (all modes)
                    t_d_mse = torch.randint(
                        0, K, (A,), device=self.device)
                    noise = torch.randn_like(act_norm)
                    noisy = self.actor_network.q_sample(
                        act_norm, t_d_mse, noise=noise)
                    pred = self.actor_network.predict_noise(
                        noisy, obs_feat_f, t_d_mse)
                    mse = ((pred - noise) ** 2).sum(dim=-1)

                    adv = (raw_advantages_all[t_idx] - adv_mean) \
                        / adv_std
                    raw_adv_t = raw_advantages_all[t_idx].detach()
                    chunk_raw_adv_mean = chunk_raw_adv_mean + raw_adv_t.mean()
                    chunk_raw_adv_pos_frac = chunk_raw_adv_pos_frac + (raw_adv_t > 0).float().mean()

                    if self.use_ppo_diffusion and has_chain:
                        # --- DDIM chain-based D2PPO clipped objective ---
                        # Use the stored DDIM denoising chain from rollout.
                        # chains_all[t_idx] shape: (A, S+1, action_dim)
                        # chain[i, 0, :] = pure noise (x_T)
                        # chain[i, S, :] = final action (x_0)
                        chain_t = chains_all[t_idx]  # (A, S+1, 2)

                        # Average PPO loss over multiple random DDIM steps.
                        # With S=5, we sample from steps [0..S-1] to get
                        # transition pairs (chain[s], chain[s+1]).
                        n_k_samples = min(S_ddim, 5)
                        ppo_accum = torch.tensor(0.0, device=self.device)
                        ratio_accum = torch.tensor(0.0, device=self.device)
                        kl_accum = torch.tensor(0.0, device=self.device)
                        clipfrac_accum = torch.tensor(0.0, device=self.device)
                        for _ki in range(n_k_samples):
                            # Random DDIM step index: s in [0, S-1]
                            # Only use steps with eta > 0 and t_curr > 0
                            # (the last step is often deterministic)
                            s_idx = torch.randint(0, S_ddim, (1,)).item()
                            t_curr = ddim_schedule[s_idx]

                            # Skip deterministic steps (t_curr == 0 or eta == 0)
                            if t_curr == 0:
                                continue

                            t_next = ddim_schedule[s_idx + 1] if s_idx + 1 < S_ddim else None

                            x_curr = chain_t[:, s_idx]      # (A, 2)
                            x_prev = chain_t[:, s_idx + 1]  # (A, 2)

                            log_p_new = self.actor_network.compute_ddim_log_prob(
                                x_prev, x_curr, obs_feat_f,
                                t_curr, t_next, eta=self._ddim_rl_eta)
                            with torch.no_grad():
                                log_p_old = self.actor_network.compute_ddim_log_prob_with(
                                    x_prev, x_curr, obs_feat_f,
                                    t_curr, t_next, self._old_denoise_net,
                                    eta=self._ddim_rl_eta)

                            # Relaxed log-ratio clamp: ratio ∈ [e^-2, e^2] ≈ [0.14, 7.4].
                            # PPO's ratio clip (ppo_clip_coef) inside the max() does the
                            # actual trust region; the log_ratio clamp only prevents
                            # numerical blow-ups. Combined with the KL-skip guard below
                            # this prevents the gen-7 runaway we saw previously.
                            log_ratio = (log_p_new - log_p_old).clamp(-2.0, 2.0)
                            ratio = log_ratio.exp()

                            pg1 = -adv * ratio
                            pg2 = -adv * torch.clamp(
                                ratio,
                                1.0 - self.ppo_clip_coef,
                                1.0 + self.ppo_clip_coef)
                            ppo_accum = ppo_accum + torch.max(pg1, pg2).mean()
                            ratio_accum = ratio_accum + ratio.std().detach()
                            # Track mean |log_ratio| (≈ policy-drift indicator).
                            # If it's already large, further actor steps will
                            # only make things worse.
                            kl_accum = kl_accum + log_ratio.detach().abs().mean()
                            # Fraction of samples where the ratio hit the PPO clip
                            # (diagnostic for how tight the trust region is).
                            clipfrac_accum = clipfrac_accum + (
                                (ratio.detach() - 1.0).abs() > self.ppo_clip_coef
                            ).float().mean()

                        ppo_loss = ppo_accum / max(n_k_samples, 1)
                        avg_ratio_std = ratio_accum / max(n_k_samples, 1)
                        avg_kl = kl_accum / max(n_k_samples, 1)
                        avg_clipfrac = clipfrac_accum / max(n_k_samples, 1)

                        # Diffusion reg (BC) keeps denoising quality
                        chunk_actor = chunk_actor + (
                            ppo_loss
                            + self.diff_reg_lambda * mse.mean()
                        )
                        chunk_wstd = chunk_wstd + avg_ratio_std
                        chunk_kl = chunk_kl + avg_kl
                        chunk_clipfrac = chunk_clipfrac + avg_clipfrac
                    else:
                        # --- Legacy AWR ---
                        w = torch.exp(
                            adv / self.awr_temperature
                        ).clamp(max=self.awr_max_weight)
                        w = w / (w.mean() + 1e-8)
                        chunk_actor = chunk_actor + (
                            (w * mse).mean()
                            + self.diff_reg_lambda * mse.mean()
                        )
                        chunk_wstd = chunk_wstd + w.std()

                    chunk_diff = chunk_diff + mse.mean()

                    # Dispersive loss
                    feat_list = self.actor_network.denoise_net\
                        .get_intermediate_features()
                    dl = torch.tensor(0.0, device=self.device)
                    nf = 0
                    for feats in feat_list:
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
                    disp_w = self.dispersive_coef_rl if self.use_dispersive_in_rl else 0.0
                    chunk_actor = chunk_actor + (disp_w * dl)
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
                avg_kl_chunk = (chunk_kl / chunk_len).item()
                avg_clipfrac_chunk = (chunk_clipfrac / chunk_len).item()

                # KL-based actor skip: if mean |log_ratio| on this chunk
                # already exceeds kl_target, the policy has drifted far
                # from the old snapshot — further steps will only make
                # things worse (classic PPO trust-region violation).
                kl_exceeded = avg_kl_chunk > self.kl_target

                # KL-adaptive early stop: once the running-mean chunk KL
                # on this epoch crosses `kl_early_stop`, freeze actor
                # updates for the rest of the epoch (critic still trains).
                if not critic_only and self.use_ppo_diffusion:
                    epoch_kl_running += avg_kl_chunk
                    epoch_kl_count += 1
                    running_mean_kl = epoch_kl_running / max(epoch_kl_count, 1)
                    if (not early_stopped_this_epoch
                            and running_mean_kl > self.kl_early_stop):
                        early_stopped_this_epoch = True
                        print(
                            f"  [KL early-stop] epoch {epoch+1} chunk "
                            f"{chunk_start//tbtt_len + 1}: running KL="
                            f"{running_mean_kl:.4f} > {self.kl_early_stop:.3f}"
                            f" — freezing actor for remainder of epoch"
                        )

                # Backward & update — actor first, then free its graph
                # before building the critic graph.
                actor_skip = critic_only or kl_exceeded or early_stopped_this_epoch
                if not actor_skip:
                    avg_actor.backward(retain_graph=False)
                    actor_gn = nn.utils.clip_grad_norm_(
                        self.actor_network.parameters(),
                        self.max_grad_norm_actor,
                    )
                    if torch.isfinite(actor_gn):
                        self.actor_optimizer.step()
                        skipped = 0.0
                    else:
                        # NaN/Inf grads would poison Adam moments.  Drop the
                        # update; track as an actor skip so it shows up in
                        # diagnostics rather than silently corrupting weights.
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        skipped = 1.0
                else:
                    actor_gn = torch.tensor(float("nan"))
                    # skipped=1 for per-chunk KL violation; early stop tracked separately.
                    skipped = 1.0 if kl_exceeded else 0.0
                avg_actor_val = avg_actor.item()
                del avg_actor, chunk_actor

                avg_critic.backward()
                critic_gn = nn.utils.clip_grad_norm_(
                    self.critic_network.parameters(),
                    self.max_grad_norm_critic,
                )
                if torch.isfinite(critic_gn):
                    self.critic_optimizer.step()
                else:
                    self.critic_optimizer.zero_grad(set_to_none=True)
                avg_critic_val = avg_critic.item()
                avg_diff_val = (chunk_diff / chunk_len).item()
                avg_disp_val = (chunk_disp / chunk_len).item()
                avg_wstd_val = (chunk_wstd / chunk_len).item()
                avg_raw_adv_mean_val = (chunk_raw_adv_mean / chunk_len).item()
                avg_raw_adv_pos_frac_val = (chunk_raw_adv_pos_frac / chunk_len).item()
                del avg_critic, chunk_critic

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
                current_gen_diagnostics["raw_adv_mean"].append(
                    avg_raw_adv_mean_val)
                current_gen_diagnostics["raw_adv_pos_frac"].append(
                    avg_raw_adv_pos_frac_val)
                current_gen_diagnostics["actor_grad_norm"].append(
                    float(actor_gn))
                current_gen_diagnostics["critic_grad_norm"].append(
                    float(critic_gn))
                current_gen_diagnostics["approx_kl"].append(avg_kl_chunk)
                current_gen_diagnostics["actor_skips"].append(skipped)
                current_gen_diagnostics["clipfrac"].append(avg_clipfrac_chunk)
                current_gen_diagnostics["actor_early_stops"].append(
                    1.0 if early_stopped_this_epoch else 0.0)
                current_gen_diagnostics["avg_speed"].append(
                    float(obs_state_all[chunk_start:chunk_end, :, 0]
                          .mean().cpu()))

            if num_updates > 0:
                avg_a = epoch_actor_loss / num_updates
                avg_d = epoch_diff_loss / num_updates
                avg_dp = epoch_disp_loss / num_updates
                _dw = self.dispersive_coef_rl if self.use_dispersive_in_rl else 0.0
                rl_comp = avg_a - self.diff_reg_lambda * avg_d - _dw * avg_dp
                disp_comp = abs(_dw * avg_dp)
                rl_label = "PPO" if self.use_ppo_diffusion else "AWR"
                print(
                    f"  Epoch {epoch+1}/{self.epochs}: "
                    f"Actor={avg_a:.4f}, "
                    f"Critic={epoch_critic_loss/num_updates:.4f}, "
                    f"Diff={avg_d:.4f}, "
                    f"Disp={avg_dp:.4f} (rl={'ON' if self.use_dispersive_in_rl else 'OFF'}), "
                )
                if epoch == 0:
                    diff_comp = self.diff_reg_lambda * avg_d
                    print(f"    Gradient budget: {rl_label}={rl_comp:.4f}  "
                          f"DiffReg={diff_comp:.4f}  "
                          f"Dispersive={disp_comp:.4f}  "
                          f"ratio(diff/rl)={diff_comp/max(abs(rl_comp), 1e-8):.1f}x")

            torch.cuda.empty_cache()

        # --- Per-generation skip summary ---
        # Counts how often the actor was frozen (KL trust-region hit). Use
        # the values just appended this generation by the chunk loop above.
        try:
            skips_arr = current_gen_diagnostics.get("actor_skips", [])
            estop_arr = current_gen_diagnostics.get("actor_early_stops", [])
            kl_arr    = current_gen_diagnostics.get("approx_kl", [])
            n_chunks = len(skips_arr)
            if n_chunks > 0:
                kl_skipped = int(sum(1 for s in skips_arr if s and s == s))  # NaN-safe
                early_stopped = int(sum(1 for e in estop_arr if e and e == e))
                kl_vals = [k for k in kl_arr if k == k]  # drop NaNs
                kl_mean = (sum(kl_vals) / len(kl_vals)) if kl_vals else float('nan')
                kl_max  = max(kl_vals) if kl_vals else float('nan')
                print(
                    f"  [actor-skip summary] kl_skip={kl_skipped}/{n_chunks} "
                    f"early_stop={early_stopped}/{n_chunks}  "
                    f"approx_kl mean={kl_mean:.4f} max={kl_max:.4f}  "
                    f"(target={self.kl_target:.2f}, early_stop_thr={self.kl_early_stop:.2f})"
                )
        except Exception as _e:
            print(f"  [actor-skip summary] (skipped: {_e})")

    # ------------------------------------------------------------------
    # Full checkpoint save / load
    # ------------------------------------------------------------------
    def save_checkpoint(self, path, generation=None, best_reward=None):
        """Save a full training checkpoint (weights + optimizer + diagnostics).

        Args:
            path: file path for the checkpoint (e.g. 'models/checkpoint.pt')
            generation: current generation counter (optional, for resuming)
            best_reward: best average reward so far (optional, for resuming)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Unwrap torch.compile if present
        actor_sd = (self.actor_network._orig_mod.state_dict()
                    if hasattr(self.actor_network, '_orig_mod')
                    else self.actor_network.state_dict())
        critic_sd = (self.critic_network._orig_mod.state_dict()
                     if hasattr(self.critic_network, '_orig_mod')
                     else self.critic_network.state_dict())

        checkpoint = {
            "actor_state_dict": actor_sd,
            "critic_state_dict": critic_sd,
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "generation": generation if generation is not None else self.generation_counter,
            "best_reward": best_reward,
            "diagnostics_history": self.diagnostics_history,
            "per_gen_meta": self.per_gen_meta,
            "map_dpc_ema": self.map_dpc_ema,
            "reward_ema_mean": self._reward_ema_mean,
            "reward_ema_var": self._reward_ema_var,
        }
        torch.save(checkpoint, path)
        print(f"[checkpoint] Saved → {path}  (gen={checkpoint['generation']})")

    def load_checkpoint(self, path):
        """Load a full training checkpoint and restore all state.

        Args:
            path: file path to the checkpoint

        Returns:
            dict with 'generation' and 'best_reward' for the training loop
            to resume from, or None if the file doesn't exist.
        """
        if not os.path.isfile(path):
            print(f"[checkpoint] No checkpoint found at {path}")
            return None

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Restore model weights (handle torch.compile wrapper)
        target_actor = (self.actor_network._orig_mod
                        if hasattr(self.actor_network, '_orig_mod')
                        else self.actor_network)
        target_critic = (self.critic_network._orig_mod
                         if hasattr(self.critic_network, '_orig_mod')
                         else self.critic_network)

        target_actor.load_state_dict(checkpoint["actor_state_dict"])
        target_critic.load_state_dict(checkpoint["critic_state_dict"])

        # Restore optimizer state
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        # Restore diagnostics / counters
        self.generation_counter = checkpoint.get("generation", 0)
        if "diagnostics_history" in checkpoint:
            self.diagnostics_history = checkpoint["diagnostics_history"]
            # Backfill keys that were added after the checkpoint was saved
            # so downstream plotting / indexing doesn't KeyError.
            for key in self.diagnostic_keys:
                if key not in self.diagnostics_history:
                    self.diagnostics_history[key] = []
        if "reward_ema_mean" in checkpoint:
            self._reward_ema_mean = checkpoint["reward_ema_mean"]
            self._reward_ema_var = checkpoint["reward_ema_var"]
        if "per_gen_meta" in checkpoint:
            self.per_gen_meta = checkpoint["per_gen_meta"]
        if "map_dpc_ema" in checkpoint:
            self.map_dpc_ema = checkpoint["map_dpc_ema"]

        gen = checkpoint.get("generation", 0)
        best = checkpoint.get("best_reward")
        print(f"[checkpoint] Loaded ← {path}  (gen={gen}, best_reward={best})")
        return {"generation": gen, "best_reward": best}

    def save_weights(self, actor_path, critic_path):
        """Save only model weights (lightweight, for best-model snapshots)."""
        os.makedirs(os.path.dirname(actor_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(critic_path) or ".", exist_ok=True)

        actor_sd = (self.actor_network._orig_mod.state_dict()
                    if hasattr(self.actor_network, '_orig_mod')
                    else self.actor_network.state_dict())
        critic_sd = (self.critic_network._orig_mod.state_dict()
                     if hasattr(self.critic_network, '_orig_mod')
                     else self.critic_network.state_dict())

        torch.save(actor_sd, actor_path)
        torch.save(critic_sd, critic_path)

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
            # x_axis length must match the number of recorded entries,
            # which may be fewer than generation_counter during warmup.
            n_entries = values_np.shape[0]
            x_axis = np.arange(self.generation_counter - n_entries + 1,
                               self.generation_counter + 1)
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
        self._dump_diagnostics_csv()

    def _dump_diagnostics_csv(self):
        """Dump diagnostics_history to CSV alongside the plot.
        Each row = one generation; each diagnostic_key produces _avg/_min/_max columns
        (or a single column for scalar entries)."""
        try:
            import csv
            csv_path = os.path.splitext(self.plot_save_path)[0] + ".csv"
            keys = [k for k in self.diagnostic_keys
                    if k in self.diagnostics_history and self.diagnostics_history[k]]
            if not keys:
                return
            # Determine max length and per-key arity (scalar vs [avg,min,max])
            max_len = max(len(self.diagnostics_history[k]) for k in keys)
            arity = {}
            for k in keys:
                v = self.diagnostics_history[k]
                if v and hasattr(v[0], "__len__"):
                    arity[k] = 3
                else:
                    arity[k] = 1
            header = ["generation"]
            for k in keys:
                if arity[k] == 3:
                    header += [f"{k}_avg", f"{k}_min", f"{k}_max"]
                else:
                    header.append(k)
            # Per-generation metadata columns (focus_map, difficulty, …).
            # Discover the union of meta keys actually used; emit a stable
            # ordering with the most useful columns first.
            preferred_meta = [
                "focus_map", "map_type", "difficulty",
                "raceline_length", "map_dpc_ema",
                "progress_per_step", "checkpoint_per_step",
                "wall_col_per_step", "agent_col_per_step", "lap_per_step",
                "steer_rate_per_step", "steer_abs_per_step", "speed_bonus_per_step",
            ]
            meta_keys_seen = set()
            for m in self.per_gen_meta:
                if isinstance(m, dict):
                    meta_keys_seen.update(m.keys())
            extra_meta = [k for k in preferred_meta if k in meta_keys_seen]
            extra_meta += sorted(k for k in meta_keys_seen if k not in preferred_meta and k != "dist_per_collision")
            header += extra_meta
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                start_gen = self.generation_counter - max_len + 1
                for i in range(max_len):
                    row = [start_gen + i]
                    for k in keys:
                        hist = self.diagnostics_history[k]
                        # Align to end (warmup pads at the front)
                        offset = max_len - len(hist)
                        if i < offset:
                            row += [""] * arity[k]
                        else:
                            entry = hist[i - offset]
                            if arity[k] == 3:
                                try:
                                    row += [float(entry[0]), float(entry[1]), float(entry[2])]
                                except Exception:
                                    row += ["", "", ""]
                            else:
                                try:
                                    row.append(float(entry))
                                except Exception:
                                    row.append("")
                    # Append per-gen meta columns (empty if no entry recorded).
                    meta = self.per_gen_meta[i] if i < len(self.per_gen_meta) else {}
                    if not isinstance(meta, dict):
                        meta = {}
                    for mk in extra_meta:
                        v = meta.get(mk, "")
                        if isinstance(v, float):
                            row.append(v)
                        else:
                            row.append(str(v) if v != "" else "")
                    w.writerow(row)
        except Exception as e:
            print(f"Error saving diagnostics CSV: {e}")

    def _compute_gae(self, data):
        rewards = data.get(("next", "reward")).to(self.device)
        dones = data.get(("next", "done")).float().to(self.device)
        values = data.get("state_value").to(self.device)
        next_values = data.get(("next", "state_value")).to(self.device)

        # Normalise to exactly 2-D (T, A)
        def to_2d(t):
            if t.ndim == 3 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            return t

        rewards = to_2d(rewards)
        dones = to_2d(dones)
        values = to_2d(values)
        next_values = to_2d(next_values)

        # Procedural-track reward clamp: never-seen tracks produce wide
        # reward distributions (rare lap bonus +10, chained checkpoints)
        # that shock the critic and have caused actor_grad_norm=NaN
        # transients during multi-procedural-gen blocks.  Clamping to ±5
        # (= |COLLISION_PENALTY|) bounds value targets without removing
        # the collision signal.  Real maps are unaffected.
        if getattr(self, "_current_map_type", "real") == "procedural":
            rewards = rewards.clamp(min=-5.0, max=5.0)

        # Running reward normalisation (EMA) — smooths across map transitions
        # instead of per-generation std which spikes at boundaries.
        # decay=0.97 (half-life ~23 updates) tracks regime changes from
        # MAPS_PER_GEN rotation without over-compressing advantage std.
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item()
        self._reward_ema_mean = 0.97 * self._reward_ema_mean + 0.03 * batch_mean
        self._reward_ema_var = 0.97 * self._reward_ema_var + 0.03 * batch_var
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

    # ------------------------------------------------------------------
    # Critic pretraining: single-map rollout (extracted for retry logic)
    # ------------------------------------------------------------------
    def _pretrain_single_rollout(self, env, pp_driver, map_name, num_agents_total,
                                  rollout_steps, generate_start_poses, get_map_dir):
        """Run one rollout on a single map for critic pretraining.

        Returns (rollout_scans, rollout_states, rollout_rewards) lists.
        Raises RuntimeError/OSError on CUDA kernel failures so the caller
        can retry.
        """
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

        collision_timers = np.zeros(num_agents_total, dtype=np.int32)
        rollout_scans = []
        rollout_states = []
        rollout_rewards = []
        rollout_dones = []

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
                ai_stuck = stuck[stuck < self.num_agents]
                if len(ai_stuck) > 0:
                    self.reset_temporal_state(ai_stuck)
                    self.reset_progress_trackers(initial_poses_xy=new_poses[:, :2], agent_idxs=ai_stuck)
                collision_timers[stuck] = 0

            # Track per-agent done flags for MC return masking
            done_t = torch.zeros(self.num_agents, dtype=torch.float32)
            if len(stuck) > 0:
                ai_done = stuck[stuck < self.num_agents]
                if len(ai_done) > 0:
                    done_t[ai_done] = 1.0

            # Store per-agent data (scans/states only for AI agents)
            rollout_scans.append(scan_t[:self.num_agents].cpu())
            rollout_states.append(state_t[:self.num_agents].cpu())
            rollout_rewards.append(rew_t[:self.num_agents].cpu())
            rollout_dones.append(done_t)

            obs = next_obs
            if (step + 1) % 100 == 0:
                print(f"    step {step + 1}/{rollout_steps}", end='\r')
            # Periodic CUDA sync to catch async kernel errors early
            if (step + 1) % 500 == 0:
                torch.cuda.synchronize()

        print()
        return rollout_scans, rollout_states, rollout_rewards, rollout_dones

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

        # Remap CNN layer indices when checkpoint used BatchNorm (stride-4
        # numbering: 0,4,8,12,16) but current VisionEncoder uses stride-3
        # (0,3,6,9,12).  Only applies when the direct key is missing.
        _cnn_index_remap = {4: 3, 5: 4, 8: 6, 9: 7, 12: 9, 13: 10, 16: 12, 17: 13}
        remapped = {}
        for k, v in list(state_dict.items()):
            if k.startswith("conv_layers.conv_layers."):
                parts = k.split(".")
                idx = int(parts[2])
                if idx in _cnn_index_remap:
                    new_k = f"conv_layers.conv_layers.{_cnn_index_remap[idx]}.{parts[3]}"
                    if new_k not in state_dict:
                        remapped[new_k] = v
        state_dict.update(remapped)
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

        # vx_mps (col 5) — used for map-agnostic speed normalisation.
        if waypoints.shape[1] > 5:
            self.raceline_mean_speed = float(np.mean(waypoints[:, 5]))
        else:
            self.raceline_mean_speed = 6.0  # fallback

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

        state_data = np.stack((lvx, lvy, avz), axis=1)
        state_tensor = torch.from_numpy(state_data.astype(np.float32))

        return scan_tensors.to(self.device), state_tensor.to(self.device)