import os

import gc
import time
import gym
import numpy as np
from D2PPO_agent import D2PPOAgent as PPOAgent
from baselines.pure_pursuit import PurePursuit
from utils.utils import *
from utils.sim_config import LIDAR_BEAMS, LIDAR_FOV, SIM_PARAMS
from track_generator import TrackGenerator
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import random
import shutil
torch.manual_seed(42)


def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_str(name, default):
    value = os.getenv(name)
    return value if value not in (None, "") else default


params_dict = SIM_PARAMS.copy()

NUM_AGENTS_AI = 3
NUM_AGENTS_PP = 8
NUM_AGENTS = NUM_AGENTS_AI + NUM_AGENTS_PP
EASY_MAPS = ["Hockenheim", "Monza", "Melbourne", "BrandsHatch"]
MEDIUM_MAPS = ["Sakhir", "SaoPaulo", "Budapest", "Silverstone"]
HARD_MAPS = ["Zandvoort", "MoscowRaceway", "Sochi",]
TOTAL_TIMESTEPS = _env_int("TR_TOTAL_TIMESTEPS", 12_000_000)
STEPS_PER_GENERATION = _env_int("TR_STEPS_PER_GENERATION", 2048)
INITIAL_POSES = None
CURRENT_MAP = _env_str("TR_START_MAP", "Zandvoort")
PATIENCE = _env_int("TR_PATIENCE", 200)  # Early stopping patience
GEN_PER_MAP = _env_int("TR_GEN_PER_MAP", 16)
MAX_GENERATIONS = _env_int("TR_MAX_GENERATIONS", 0)
SKIP_RESUME = os.getenv("TR_SKIP_RESUME", "0") == "1"
DISABLE_HELDOUT_EVAL = os.getenv("TR_DISABLE_HELDOUT_EVAL", "0") == "1"
RESET_OPT_ON_MAP_SWITCH = os.getenv("TR_RESET_OPT_ON_MAP_SWITCH", "1") == "1"
HELDOUT_MAPS = ["Spa", "YasMarina", "Spielberg"]
HELDOUT_EVAL_EVERY = _env_int("TR_HELDOUT_EVAL_EVERY", 12)  # Generations between held-out evaluations
MIN_SELECTION_SPEED = _env_float("TR_MIN_SELECTION_SPEED", 4.5)
MIN_SELECTION_SPEED_RATIO = _env_float("TR_MIN_SELECTION_SPEED_RATIO", 0.85)
MIN_SELECTION_PROGRESS_PER_STEP = _env_float(
    "TR_MIN_SELECTION_PROGRESS_PER_STEP", 0.055
)


def compute_selection_score(
    dist_per_collision,
    avg_speed,
    raceline_mean_speed,
    progress_per_step=None,
):
    """Checkpoint-selection score that rejects slow/cautious collapse.

    Distance-per-collision alone can select a model that simply slows down.
    This score keeps DPC, but multiplies it by pace and softly gates models
    whose speed/progress have dropped below useful racing behavior.
    """
    speed_ratio = avg_speed / max(float(raceline_mean_speed), 1e-6)
    score = float(dist_per_collision) * speed_ratio

    gate = 1.0
    min_speed = max(
        float(MIN_SELECTION_SPEED),
        float(MIN_SELECTION_SPEED_RATIO) * float(raceline_mean_speed),
    )
    if avg_speed < min_speed:
        gate *= max(avg_speed, 0.0) / max(min_speed, 1e-6)
    if progress_per_step is not None and progress_per_step < MIN_SELECTION_PROGRESS_PER_STEP:
        gate *= max(progress_per_step, 0.0) / max(MIN_SELECTION_PROGRESS_PER_STEP, 1e-6)

    return score * gate, speed_ratio, gate

def get_curriculum_map_pool(generation, selector=None):
    """Returns the appropriate map pool based on training progress.
    Held-out maps are excluded so generalization eval is meaningful."""
    pool = EASY_MAPS + MEDIUM_MAPS + HARD_MAPS
    return [m for m in pool if m not in HELDOUT_MAPS]


def compute_geometric_difficulty(waypoints_xy, raceline_length=None):
    """Estimate a geometric difficulty score for a track from its raceline.

    Works for both real and procedural maps — only needs the raceline polyline.
    Combines two signals:
      * Mean absolute curvature of the raceline (rad/m).  Tighter, more
        frequent turns → higher.
      * Track length (longer tracks demand more sustained policy stability).

    The polyline is resampled to uniform 1 m spacing before differentiating
    so the result is robust to non-uniform raceline point spacing (which
    otherwise produces giant spikes wherever consecutive points are close).

    Returns a non-negative scalar; higher = harder.  Roughly normalised so
    typical real-world circuits land in [0.5, 3.0].
    """
    try:
        pts = np.asarray(waypoints_xy, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 6 or pts.shape[1] < 2:
            return float("nan")
        # Cumulative arc-length along the (closed) polyline.
        diffs = np.diff(pts, axis=0, append=pts[:1])
        seg = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])  # length len(pts)+1
        total_len = float(s[-1])
        if not np.isfinite(total_len) or total_len < 10.0:
            return float("nan")
        # Resample to uniform 1 m spacing for stable finite differences.
        ds = 1.0
        n = max(int(total_len / ds), 8)
        s_uni = np.linspace(0.0, total_len, n, endpoint=False)
        # pts is closed if last≈first; build a closed series for interp.
        pts_closed = np.vstack([pts, pts[:1]])
        x_u = np.interp(s_uni, s, pts_closed[:, 0])
        y_u = np.interp(s_uni, s, pts_closed[:, 1])
        # Heading per uniform segment, then |Δheading| / ds = curvature.
        dx = np.diff(x_u, append=x_u[:1])
        dy = np.diff(y_u, append=y_u[:1])
        headings = np.arctan2(dy, dx)
        dtheta = np.diff(headings, append=headings[:1])
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        curvature = np.abs(dtheta) / ds  # rad/m
        mean_curv = float(np.mean(curvature))
        if raceline_length is None or raceline_length <= 0:
            raceline_length = total_len
        # Length factor: long tracks slightly harder; saturates above ~1500m.
        length_factor = min(float(raceline_length) / 1000.0, 1.5)
        # Curvature factor: 0.05 rad/m ≈ 20m radius → factor ≈ 1.0.
        curv_factor = mean_curv / 0.05
        return float(curv_factor + 0.5 * length_factor)
    except Exception:
        return float("nan")

# --- Track Generator ---
track_gen = TrackGenerator(
    min_track_length=50,
    max_track_length=500,
    min_turns=10,
    max_turns=50,
    min_track_width=0.8,
    max_track_width=2.0,
    min_turn_radius=5.0,
    seed=None,  # Random every time
)
_last_generated_track = None

env = gym.make(
    "f110_gym:f110-v0",
    map=get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map",
    num_agents=NUM_AGENTS,
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    params=params_dict
)
INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
obs, timestep, _, _ = env.reset(poses=INITIAL_POSES)
env.render(mode="human")


ORIGINAL_WEIGHT = "models/actor/pretrained/masked_test/actor_masked_final.pt"
CRITIC_WEIGHT = "models/critic/critic_best.pt"

agent = PPOAgent(
    num_agents=NUM_AGENTS_AI,
    map_name=CURRENT_MAP,
    steps=STEPS_PER_GENERATION,
    params=params_dict,
    transfer=[ORIGINAL_WEIGHT, CRITIC_WEIGHT],
    tbtt_length=512,
)
pp_driver = PurePursuit(
    map_name=CURRENT_MAP,
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    max_speed=7.0,
    min_speed=1.5,
)

# torch.compile stays disabled for Mamba2 recurrent step() stability.
ALL_PRETRAIN_MAPS = EASY_MAPS + MEDIUM_MAPS
if CRITIC_WEIGHT is None:
    agent.pretrain_critic(
        env=env,
        pp_driver=pp_driver,
        num_agents_total=NUM_AGENTS,
        maps=ALL_PRETRAIN_MAPS,
        rollout_steps=8000,
        num_rollouts=len(ALL_PRETRAIN_MAPS),
        epochs=300,
        lr=3e-4,
        batch_size=1024,
        save_demos_path="demos/critic_demos.pt",
    )
    CRITIC_WEIGHT = "models/critic/pretrained/critic_pretrained.pt"

def _validate_procedural_track(
    map_name,
    n_steps=40,
    crash_window=10,
    min_progress_m=0.5,
):
    """Smoke-test a freshly generated track with pure-pursuit.

    Returns True iff every AI agent survives the first ``crash_window``
    steps and at least one AI agent makes forward progress within
    ``n_steps``.  This rejects tracks that spawn agents into walls (the
    dominant cause of dead procedural generations) without paying for a
    full rollout.

    The env is restored to the previous map's state via the caller's
    subsequent ``env.update_map`` + ``env.reset`` sequence, so this
    function only needs to leave ``CURRENT_MAP`` consistent.
    """
    try:
        env.update_map(get_map_dir(map_name) + f"/{map_name}_map", ".png")
        pp_driver.update_map(map_name)
        poses = generate_start_poses(map_name, NUM_AGENTS)
        obs, _, _, _ = env.reset(poses=poses)
        reset_cols = np.asarray(obs["collisions"][:NUM_AGENTS_AI])
        if np.any(reset_cols == 1):
            print(f"  [validate] {map_name}: wall collision on reset")
            return False
        start_x = np.array(obs["poses_x"][:NUM_AGENTS_AI], dtype=np.float64)
        start_y = np.array(obs["poses_y"][:NUM_AGENTS_AI], dtype=np.float64)
        for step in range(n_steps):
            acts = np.array(
                [pp_driver.get_action(obs, agent_idx=j)
                 for j in range(NUM_AGENTS)],
                dtype=np.float32,
            )
            obs, _, _, _ = env.step(acts)
            cols = np.asarray(obs["collisions"][:NUM_AGENTS_AI])
            if step < crash_window and np.any(cols == 1):
                print(f"  [validate] {map_name}: wall collision at step {step + 1}")
                return False
        # At least one AI agent must move far enough to show the map is drivable.
        # With the simulator's 0.01s timestep and validation min-speed of 1.5 m/s,
        # 40 steps only covers ~0.6m before controller/dynamics effects.
        end_x = np.array(obs["poses_x"][:NUM_AGENTS_AI], dtype=np.float64)
        end_y = np.array(obs["poses_y"][:NUM_AGENTS_AI], dtype=np.float64)
        max_disp = float(np.max(np.hypot(end_x - start_x, end_y - start_y)))
        if max_disp <= min_progress_m:
            print(
                f"  [validate] {map_name}: insufficient motion "
                f"({max_disp:.2f}m <= {min_progress_m:.2f}m)"
            )
            return False
        return True
    except Exception as exc:
        print(f"  [validate] exception: {exc}")
        return False


def _switch_to_new_track(gen_label="init"):
    """Generate a new random track, switch env/agents to it, and clean up the old one."""
    global CURRENT_MAP, _last_generated_track, INITIAL_POSES
    # Clean up previous generated track
    if _last_generated_track is not None:
        old_dir = os.path.join("maps", _last_generated_track)
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)
        _last_generated_track = None

    # Pure-pursuit smoke-test rejects bad spawns before they cost a full
    # generation of zero-gradient rollouts.  Up to MAX_VALIDATION_TRIES
    # regenerations; on persistent failure, fall back to a real map.
    MAX_VALIDATION_TRIES = 4
    track_name = None
    for attempt in range(MAX_VALIDATION_TRIES):
        candidate = f"gen_track_{gen_label}_{attempt}" if attempt else f"gen_track_{gen_label}"
        try:
            track_gen.generate(candidate)
        except RuntimeError as e:
            print(f"  [validate] generate failed ({e}) attempt {attempt+1}")
            continue
        if _validate_procedural_track(candidate):
            track_name = candidate
            break
        # Reject: tear down and try again
        print(f"  [validate] rejecting {candidate}")
        bad_dir = os.path.join("maps", candidate)
        if os.path.isdir(bad_dir):
            shutil.rmtree(bad_dir, ignore_errors=True)

    if track_name is None:
        print("  [validate] no viable procedural track after "
              f"{MAX_VALIDATION_TRIES} tries — falling back to real map")
        available = EASY_MAPS + MEDIUM_MAPS + HARD_MAPS
        CURRENT_MAP = random.choice(available)
        _last_generated_track = None
    else:
        CURRENT_MAP = track_name
        _last_generated_track = track_name

    INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
    env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
    wp_xy, wp_s, rl = agent._load_waypoints(CURRENT_MAP)
    agent.waypoints_xy, agent.waypoints_s, agent.raceline_length = wp_xy, wp_s, rl
    pp_driver.update_map(CURRENT_MAP)

    agent.clear_experience_buffer()

    # Procedurally generated track has a fresh loss surface — reset Adam
    # state so prior-map momentum doesn't bias the first updates here.
    if RESET_OPT_ON_MAP_SWITCH:
        agent.reset_optimizers()

    # Force cleanup after map switch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Rescale rewards for the new map while keeping rollout length fixed.
    _update_steps_and_buffer(rl)

    return INITIAL_POSES


def _switch_to_real_map(map_name):
    """Switch the environment to an existing real map (no procedural generation)."""
    global CURRENT_MAP, _last_generated_track, INITIAL_POSES
    _prev_map = CURRENT_MAP
    # Clean up previous generated track
    if _last_generated_track is not None:
        old_dir = os.path.join("maps", _last_generated_track)
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)
        _last_generated_track = None

    CURRENT_MAP = map_name
    INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
    env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
    wp_xy, wp_s, rl = agent._load_waypoints(CURRENT_MAP)
    agent.waypoints_xy, agent.waypoints_s, agent.raceline_length = wp_xy, wp_s, rl
    pp_driver.update_map(CURRENT_MAP)
    agent.clear_experience_buffer()
    agent.reset_reward_ema()  # Reward distribution changes with map; re-anchor EMA
    # Curriculum transition: discard Adam momentum/variance from the previous
    # map's loss surface so it doesn't poison the next map's PPO updates.
    # Skip when the map didn't actually change (e.g. initial bootstrap or
    # resume from checkpoint with the same focus map).
    if _prev_map != map_name and RESET_OPT_ON_MAP_SWITCH:
        agent.reset_optimizers()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _update_steps_and_buffer(rl)
    return INITIAL_POSES


def _switch_map_keep_buffer(map_name):
    """Mid-generation map switch (Option B): same as _switch_to_real_map but
    preserves the rollout buffer so the PPO minibatch sees heterogeneous maps.

    Does NOT clear buffer, NOT reset reward EMA (keep normalisation stable
    across the generation), and does NOT trigger critic warmup."""
    global CURRENT_MAP, INITIAL_POSES
    CURRENT_MAP = map_name
    INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
    env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
    wp_xy, wp_s, rl = agent._load_waypoints(CURRENT_MAP)
    agent.waypoints_xy, agent.waypoints_s, agent.raceline_length = wp_xy, wp_s, rl
    agent.PROGRESS_REWARD = 500.0 / max(rl, 1.0)  # Per-map reward scaling
    pp_driver.update_map(CURRENT_MAP)
    return INITIAL_POSES


def _switch_to_eval_map(map_name):
    """Switch maps for held-out evaluation without touching training state.

    Unlike ``_switch_to_real_map``, this does not clear replay data, reset
    reward EMA, reset optimizers, or resize the training rollout buffer.
    Held-out eval should select checkpoints, not mutate the next PPO update.
    """
    global CURRENT_MAP, INITIAL_POSES
    CURRENT_MAP = map_name
    INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
    env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
    wp_xy, wp_s, rl = agent._load_waypoints(CURRENT_MAP)
    agent.waypoints_xy, agent.waypoints_s, agent.raceline_length = wp_xy, wp_s, rl
    pp_driver.update_map(CURRENT_MAP)
    return INITIAL_POSES


CRITIC_WARMUP_GENS = _env_int("TR_CRITIC_WARMUP_GENS", 4)
MAPS_PER_GEN = _env_int("TR_MAPS_PER_GEN", 3)
agent.mixed_map_generation = MAPS_PER_GEN > 1


def _update_steps_and_buffer(raceline_length):
    """Normalize progress reward for a map and reset temporal state."""
    agent.PROGRESS_REWARD = 500.0 / max(raceline_length, 1.0)

    agent.reset_buffers()
    print(f"  Steps/gen={STEPS_PER_GENERATION}, tbtt={agent.tbtt_length} "
          f"(track={raceline_length:.1f}m, progress_r={agent.PROGRESS_REWARD:.3f}/m)")


_switch_to_real_map(CURRENT_MAP)
agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
obs, _, _, _ = env.reset(poses=INITIAL_POSES)

print(f"Starting training on {agent.device} for {TOTAL_TIMESTEPS} timesteps...")

best_dist_per_collision = -float('inf')
best_per_map = {}  # {map_name: best composite score on that map}
best_generalist_score = -float('inf')
best_generalist_dpc = -float('inf')
patience = 0
_critic_warmup_remaining = 0

CHECKPOINT_PATH = _env_str("TR_CHECKPOINT_PATH", "models/checkpoint.pt")
RESUME_CHECKPOINT_PATH = _env_str("TR_RESUME_CHECKPOINT", CHECKPOINT_PATH)
print(f"[startup] transfer actor={ORIGINAL_WEIGHT}, critic={CRITIC_WEIGHT}")
print(f"[startup] resume_checkpoint={RESUME_CHECKPOINT_PATH}, save_checkpoint={CHECKPOINT_PATH}, "
    f"resume={'off' if SKIP_RESUME else 'on'}")
resumed = None if SKIP_RESUME else agent.load_checkpoint(RESUME_CHECKPOINT_PATH)
start_gen = 0
if resumed is not None:
    start_gen = resumed["generation"]
    if resumed["best_reward"] is not None:
        best_dist_per_collision = resumed["best_reward"]
    print(
        f"Resuming from generation {start_gen}, "
        f"best_dist_per_collision={best_dist_per_collision:.3f}"
    )

collision_timers = np.zeros(NUM_AGENTS, dtype=np.int32)
COLLISION_RESET_THRESHOLD = 20

total_steps_done = start_gen * STEPS_PER_GENERATION
gen = start_gen
focus_map = CURRENT_MAP
next_focus_is_generated = True
force_initial_generated_switch = True
while total_steps_done < TOTAL_TIMESTEPS and (MAX_GENERATIONS <= 0 or gen < MAX_GENERATIONS):
    collisions = 0
    gen += 1

    if force_initial_generated_switch:
        _startup_mode = "Resumed" if start_gen > 0 else "Starting"
        print(
            f"[curriculum] {_startup_mode} at generation {start_gen}; "
            f"switching to a generated track before generation {gen}."
        )
        _switch_to_new_track(gen)
        focus_map = CURRENT_MAP
        next_focus_is_generated = False
        obs, _, _, _ = env.reset(poses=INITIAL_POSES)
        agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
        agent.reset_buffers()
        collision_timers[:] = 0
        _critic_warmup_remaining = CRITIC_WARMUP_GENS
        force_initial_generated_switch = False

    print(f"--- Generation {gen} focus={focus_map}  "
          f"(steps={STEPS_PER_GENERATION}, "
          f"total={total_steps_done}/{TOTAL_TIMESTEPS}) ---")
    total_reward_this_gen = []
    ego_reward_this_gen = []
    total_distance_this_gen = 0.0
    current_gen_time = 0.0
    reward_component_sums = {}  # running sums for per-component diagnostics

    # Rotate through extra maps for heterogeneous PPO minibatches.
    if MAPS_PER_GEN > 1 and not _last_generated_track:
        _pool = get_curriculum_map_pool(gen)
        if focus_map not in _pool:
            focus_map = random.choice(_pool)
        if CURRENT_MAP != focus_map:
            _switch_map_keep_buffer(focus_map)
            obs, _, _, _ = env.reset(poses=INITIAL_POSES)
            agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
            agent.reset_buffers()
            collision_timers[:] = 0
        _gen_maps = [focus_map]
        _extras = [m for m in _pool if m != focus_map]
        random.shuffle(_extras)
        _gen_maps += _extras[:MAPS_PER_GEN - 1]
        _segment_steps = STEPS_PER_GENERATION // MAPS_PER_GEN
        _switch_boundaries = set(
            _segment_steps * (i + 1) for i in range(MAPS_PER_GEN - 1)
        )
        _map_rotation_idx = 0
        print(f"  [multi-map] gen maps: {_gen_maps} "
              f"(segment={_segment_steps} steps each)")
    else:
        _gen_maps = [CURRENT_MAP]
        _switch_boundaries = set()

    for step in range(STEPS_PER_GENERATION):
        if step in _switch_boundaries:
            _map_rotation_idx += 1
            _new_map = _gen_maps[_map_rotation_idx]
            print(f"\n  [multi-map] step {step}: switching to {_new_map}")
            _switch_map_keep_buffer(_new_map)
            obs, _, _, _ = env.reset(poses=INITIAL_POSES)
            agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
            agent.reset_buffers()
            collision_timers[:] = 0
            pp_driver.update_map(CURRENT_MAP)

        timer = time.time()
        done_np = np.zeros(NUM_AGENTS, dtype=np.int32)

        scan_tensors, state_tensor = agent._obs_to_tensors(obs)
        action_tensor, log_prob_tensor, value_tensor = agent.get_action_and_value(
            scan_tensors, state_tensor
        )

        action_np = action_tensor.cpu().numpy()

        if action_np.shape[0] < NUM_AGENTS:
            # Fill in the remaining agents with Pure Pursuit actions
            n_pp = NUM_AGENTS - action_np.shape[0]
            pp_start = action_np.shape[0]
            pp_acts = np.array([pp_driver.get_action(obs, agent_idx=pp_start + j)
                                for j in range(n_pp)], dtype=np.float32)
            action_np = np.vstack((action_np, pp_acts))

        next_obs, timestep, _, _ = env.step(action_np)

        # Track forward distance travelled by AI agents (meters)
        ai_speeds = np.clip(next_obs['linear_vels_x'][:NUM_AGENTS_AI], 0.0, None)
        total_distance_this_gen += float(np.sum(ai_speeds) * float(max(timestep, 0.0)))

        # Calculate Reward
        rewards_list, avg_reward = agent.calculate_reward(next_obs, action=action_np)

        # Accumulate per-component reward diagnostics
        for k, v in agent._reward_components.items():
            reward_component_sums[k] = reward_component_sums.get(k, 0.0) + v

        # Update collision timers
        current_collisions = np.array(next_obs['collisions'][:NUM_AGENTS])
        collision_timers[(current_collisions == 1)] += 1
        collision_timers[current_collisions == 0] = 0

        agents_to_reset = np.where(collision_timers >= COLLISION_RESET_THRESHOLD)[0]

        if len(agents_to_reset) > 0:
            # Generate new poses for stuck agents
            poses = np.array([[x, y, theta] for x, y, theta in zip(
                next_obs['poses_x'], next_obs['poses_y'], next_obs['poses_theta']
            )])
            INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS, agent_poses=poses)

            # Reset the environment for stuck agents
            next_obs, _, _, _ = env.reset(poses=INITIAL_POSES, agent_idxs=agents_to_reset)

            # Reset agent buffers and trackers
            ai_agents_to_reset = agents_to_reset[agents_to_reset < NUM_AGENTS_AI]
            if len(ai_agents_to_reset) > 0:
                agent.reset_buffers(ai_agents_to_reset)
            agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2], agent_idxs=agents_to_reset)

            # Reset collision timers for these agents
            collision_timers[agents_to_reset] = 0

            # Count these as collision exits
            collisions += len(agents_to_reset[agents_to_reset < NUM_AGENTS_AI])

            done_np[agents_to_reset] = 1

        total_reward_this_gen.append(avg_reward)
        ego_reward_this_gen.append(rewards_list[0])

        # Calculate time
        current_gen_time += timestep

        # Store Experience
        agent.store_transition(
            obs=[scan_tensors, state_tensor],
            next=next_obs,
            action=action_tensor,
            log_prob=log_prob_tensor,
            reward=rewards_list,
            done=done_np[:NUM_AGENTS_AI],
            value=value_tensor,
        )

        done_np = np.zeros(NUM_AGENTS, dtype=np.int32)

        # Periodic CUDA sync: surfaces async kernel errors as catchable
        # RuntimeError instead of delayed SIGILL/segfault
        if (step + 1) % 500 == 0:
            torch.cuda.synchronize()

        if (step + 1) % 10 == 0 or step == STEPS_PER_GENERATION - 1:
            print(f"{step+1}/{STEPS_PER_GENERATION}: \
Collisions: {collisions}, \
Max vel: {np.max(next_obs['linear_vels_x'][:NUM_AGENTS_AI]):.1f} m/s, \
Max actor_vel: {torch.max(action_tensor[:,1]).item():.1f} m/s, \
Ego Speed: {next_obs['linear_vels_x'][0]:.2f} \
Avg Reward: {sum(total_reward_this_gen) / (step + 1):.3f} \
S/s: {1 / (time.time() - timer):.1f}", end='\r')

        obs = next_obs

    total_steps_done += STEPS_PER_GENERATION
    print() # Finish the carriage return line

    # Print per-component reward breakdown
    n = max(len(total_reward_this_gen), 1)
    comp_str = " | ".join(f"{k}: {v/n:+.3f}" for k, v in reward_component_sums.items())
    print(f"  Reward breakdown (avg/step): {comp_str}")

    current_physics_time = 0.0

    agent.finalize_rollout(obs)

    reward_avg = sum(total_reward_this_gen) / len(total_reward_this_gen)
    current_avg_ego_reward = sum(ego_reward_this_gen) / len(total_reward_this_gen)
    dist_per_collision = total_distance_this_gen / max(collisions, 1)
    avg_ai_speed = total_distance_this_gen / max(current_gen_time * NUM_AGENTS_AI, 1e-6)
    progress_per_step = float(reward_component_sums.get("progress", 0.0)) / float(n)
    selection_score, selection_speed_ratio, selection_gate = compute_selection_score(
        dist_per_collision=dist_per_collision,
        avg_speed=avg_ai_speed,
        raceline_mean_speed=getattr(agent, "raceline_mean_speed", 6.0),
        progress_per_step=progress_per_step,
    )

    # Score multi-map generations against the focus map, not the final segment.
    _score_map = focus_map if MAPS_PER_GEN > 1 else CURRENT_MAP
    map_best = best_per_map.get(_score_map, -float('inf'))
    is_improvement = (
        selection_score > map_best
        or (selection_score >= map_best and collisions == 0)
    )
    if is_improvement:
        agent.save_weights("models/actor/actor_best.pt", "models/critic/critic_best.pt")
        best_per_map[_score_map] = selection_score
        if dist_per_collision > best_dist_per_collision:
            best_dist_per_collision = dist_per_collision
        print(
            f"New best on {_score_map}: "
            f"score={selection_score:.3f} "
            f"(dpc={dist_per_collision:.1f}, speed={avg_ai_speed:.2f}, "
            f"speed_ratio={selection_speed_ratio:.2f}, gate={selection_gate:.2f}, "
            f"distance={total_distance_this_gen:.1f} m, collisions={collisions})"
        )
        patience = 0
    elif gen % 100 == 0:
        # Keep key name `best_reward` for backward-compatible checkpoints.
        agent.save_checkpoint(
            CHECKPOINT_PATH,
            generation=gen,
            best_reward=best_dist_per_collision,
        )
        agent.save_weights(f"models/actor/checkpoint/actor_gen_{gen}.pt",
                           f"models/critic/checkpoint/critic_gen_{gen}.pt")
        print(f"Checkpoint saved at generation {gen}.")
        patience += 1
    else:
        patience += 1
        print(
            f"No improvement on {_score_map}: "
            f"score={selection_score:.3f} vs {map_best:.3f} "
            f"(dpc={dist_per_collision:.1f}, speed={avg_ai_speed:.2f}, "
            f"gate={selection_gate:.2f}). "
            f"Patience: {patience}"
        )

    critic_only = _critic_warmup_remaining > 0
    if critic_only:
        _critic_warmup_remaining -= 1
        print(f"  [Critic warmup] {_critic_warmup_remaining} gens remaining — skipping actor update")

    _is_procedural = bool(_last_generated_track) or str(focus_map).startswith("gen_track")
    _map_type = "procedural" if _is_procedural else "real"
    try:
        _diff = compute_geometric_difficulty(
            getattr(agent, "waypoints_xy", None),
            getattr(agent, "raceline_length", None),
        )
    except Exception:
        _diff = float("nan")
    agent.set_gen_meta(
        focus_map=focus_map,
        map_type=_map_type,
        difficulty=_diff,
        raceline_length=float(getattr(agent, "raceline_length", 0.0) or 0.0),
        dist_per_collision=dist_per_collision,
        progress_per_step=progress_per_step,
        checkpoint_per_step=float(reward_component_sums.get("checkpoint", 0.0)) / float(n),
        wall_col_per_step=float(reward_component_sums.get("wall_col", 0.0)) / float(n),
        agent_col_per_step=float(reward_component_sums.get("agent_col", 0.0)) / float(n),
        lap_per_step=float(reward_component_sums.get("lap", 0.0)) / float(n),
        steer_rate_per_step=float(reward_component_sums.get("steer_rate", 0.0)) / float(n),
        steer_abs_per_step=float(reward_component_sums.get("steer_abs", 0.0)) / float(n),
        speed_bonus_per_step=float(reward_component_sums.get("speed_bonus", 0.0)) / float(n),
    )

    agent.learn(collisions, reward_avg, critic_only=critic_only)

    if gen % GEN_PER_MAP == 0:
        if (not DISABLE_HELDOUT_EVAL) and gen % HELDOUT_EVAL_EVERY == 0 and len(HELDOUT_MAPS) > 0:
            eval_results = {}
            eval_scores = {}
            print(f"\n[generalist-eval] Evaluating on held-out maps: {HELDOUT_MAPS}")
            for eval_map in HELDOUT_MAPS:
                _switch_to_eval_map(eval_map)
                eval_obs, _, _, _ = env.reset(poses=INITIAL_POSES)
                agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
                agent.reset_buffers()
                collision_timers[:] = 0
                eval_steps = min(STEPS_PER_GENERATION, 2000)  # cap eval cost
                eval_coll = 0
                eval_dist = 0.0
                eval_time = 0.0
                for _ in range(eval_steps):
                    scan_t, state_t = agent._obs_to_tensors(eval_obs)
                    with torch.no_grad():
                        act_t, _, _ = agent.get_action_and_value(
                            scan_t, state_t, deterministic=True
                        )
                    act_np = act_t.cpu().numpy()
                    if act_np.shape[0] < NUM_AGENTS:
                        n_pp = NUM_AGENTS - act_np.shape[0]
                        pp_start = act_np.shape[0]
                        pp_acts = np.array([pp_driver.get_action(eval_obs, agent_idx=pp_start + j)
                                            for j in range(n_pp)], dtype=np.float32)
                        act_np = np.vstack((act_np, pp_acts))
                    eval_obs, ets, _, _ = env.step(act_np)
                    eval_time += float(max(ets, 0.0))
                    ai_sp = np.clip(eval_obs['linear_vels_x'][:NUM_AGENTS_AI], 0.0, None)
                    eval_dist += float(np.sum(ai_sp) * float(max(ets, 0.0)))
                    cur_coll = np.array(eval_obs['collisions'][:NUM_AGENTS])
                    collision_timers[(cur_coll == 1)] += 1
                    collision_timers[cur_coll == 0] = 0
                    stuck = np.where(collision_timers >= COLLISION_RESET_THRESHOLD)[0]
                    if len(stuck) > 0:
                        poses = np.array([[x, y, th] for x, y, th in zip(
                            eval_obs['poses_x'], eval_obs['poses_y'], eval_obs['poses_theta'])])
                        INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS, agent_poses=poses)
                        eval_obs, _, _, _ = env.reset(poses=INITIAL_POSES, agent_idxs=stuck)
                        ai_stuck = stuck[stuck < NUM_AGENTS_AI]
                        if len(ai_stuck) > 0:
                            agent.reset_buffers(ai_stuck)
                        agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2], agent_idxs=stuck)
                        collision_timers[stuck] = 0
                        eval_coll += len(ai_stuck)
                eval_dpc = eval_dist / max(eval_coll, 1)
                eval_avg_speed = eval_dist / max(eval_time * NUM_AGENTS_AI, 1e-6)
                eval_score, eval_speed_ratio, eval_gate = compute_selection_score(
                    dist_per_collision=eval_dpc,
                    avg_speed=eval_avg_speed,
                    raceline_mean_speed=getattr(agent, "raceline_mean_speed", 6.0),
                    progress_per_step=None,
                )
                eval_results[eval_map] = eval_dpc
                eval_scores[eval_map] = eval_score
                print(
                    f"  {eval_map}: score={eval_score:.1f}, "
                    f"dpc={eval_dpc:.1f} m/coll, speed={eval_avg_speed:.2f}, "
                    f"speed_ratio={eval_speed_ratio:.2f}, gate={eval_gate:.2f} "
                    f"(dist={eval_dist:.0f}, coll={eval_coll})"
                )
            mean_dpc = sum(eval_results.values()) / len(eval_results)
            mean_score = sum(eval_scores.values()) / len(eval_scores)
            print(
                f"[generalist-eval] mean held-out: "
                f"score={mean_score:.1f}, dpc={mean_dpc:.1f} m/coll"
            )
            if mean_score > best_generalist_score:
                best_generalist_score = mean_score
                best_generalist_dpc = mean_dpc
                agent.save_weights("models/actor/actor_generalist.pt",
                                   "models/critic/critic_generalist.pt")
                print(
                    f"[generalist-eval] NEW generalist best: "
                    f"score={mean_score:.1f}, dpc={mean_dpc:.1f} m/coll"
                )
            agent.clear_experience_buffer()

        if next_focus_is_generated:
            _switch_to_new_track(gen)
            focus_map = CURRENT_MAP  # Generated track: disables Option B rotation
        else:
            pool = get_curriculum_map_pool(gen)
            _switch_to_real_map(random.choice(pool))
            focus_map = CURRENT_MAP
        next_focus_is_generated = not next_focus_is_generated

        print(f"Gen {gen}: New focus map \u2192 {CURRENT_MAP}  "
              f"(steps/gen={STEPS_PER_GENERATION})")
        agent.last_cumulative_distance = np.zeros(NUM_AGENTS_AI)
        agent.last_wp_index = np.zeros(NUM_AGENTS_AI, dtype=np.int32)
        obs, _, _, _ = env.reset(poses=INITIAL_POSES)
        agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
        agent.reset_buffers()
        # Don't reset reward EMA — let it adapt smoothly across maps
        collision_timers[:] = 0
        _critic_warmup_remaining = CRITIC_WARMUP_GENS
        # Reset patience so new-map generations aren't compared against prior map's streak
        patience = 0
        prev_best = best_per_map.get(CURRENT_MAP, None)
        if prev_best is not None:
            print(f"  [map-best] {CURRENT_MAP} previous best score: {prev_best:.3f}")
        else:
            print(f"  [map-best] {CURRENT_MAP} first visit")


agent.save_checkpoint(CHECKPOINT_PATH, generation=gen, best_reward=best_dist_per_collision)
agent.save_weights("models/actor/checkpoint/actor_gen_FINAL.pt",
                   "models/critic/checkpoint/critic_gen_FINAL.pt")
print(f"Final checkpoint saved at generation {gen}.")

# Clean up last generated track
if _last_generated_track is not None:
    old_dir = os.path.join("maps", _last_generated_track)
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir, ignore_errors=True)

# --- END OF TRAINING ---
env.close()
print("Training complete.")