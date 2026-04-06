import time
import gym
import numpy as np
from D2PPO_agent import D2PPOAgent as PPOAgent
# from utils.control_handler import ControlHandler
from baselines.mpc_agent import MPCAgent
from baselines.pure_pursuit import PurePursuit
from utils.utils import *
from track_generator import TrackGenerator
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import random
import shutil

params_dict = {'mu': 1.0489,
               'C_Sf': 4.718,
               'C_Sr': 5.4562,
               'lf': 0.15875,
               'lr': 0.17145,
               'h': 0.074,
               'm': 3.74,
               'I': 0.04712,
               's_min': -0.34,
               's_max': 0.34,
               'sv_min': -3.2,
               'sv_max': 3.2,
               'v_switch':7.319,
               'a_max': 9.51,
               'v_min': -5.0,
               'v_max': 20.0,
               'width': 0.31,
               'length': 0.58
               }

# --- Main Training Parameters ---
NUM_AGENTS_AI = 6
NUM_AGENTS_PP = 4
NUM_AGENTS = NUM_AGENTS_AI + NUM_AGENTS_PP
EASY_MAPS = ["Hockenheim", "Monza", "Melbourne", "BrandsHatch"]
MEDIUM_MAPS = ["Oschersleben", "Sakhir", "Sepang", "SaoPaulo", "Budapest", "Catalunya", "Silverstone"]
HARD_MAPS = ["Zandvoort", "MoscowRaceway", "Nuerburgring", "Sochi",]
TOTAL_TIMESTEPS = 12_000_000
STEPS_PER_GENERATION = 2048  # Initial default; overwritten per-map to 10x track length
LIDAR_BEAMS = 1080  # Default is 1080
LIDAR_FOV = 4.7   # Default is 4.7 radians (approx 270 deg)
INITIAL_POSES = None # Generated later
CURRENT_MAP = "Catalunya" # Starting map, used for pretraining
PATIENCE = 200  # Early stopping patience
GEN_PER_MAP = 16  # Generations per track (let Mamba2 buffer fill for localization)

def get_curriculum_map_pool(generation, selector=None):
    """Returns the appropriate map pool based on training progress."""
    return EASY_MAPS + MEDIUM_MAPS + HARD_MAPS  # Phase 3: Full curriculum

# --- Track Generator ---
track_gen = TrackGenerator(
    min_track_length=50,
    max_track_length=600,
    min_turns=6,
    max_turns=35,
    min_track_width=0.5,
    max_track_width=2.0,
    min_turn_radius=3.0,
    seed=None,  # Random every time
)
_last_generated_track = None  # Track cleanup bookkeeping

# -- Environment Setup ---
env = gym.make(
    "f110_gym:f110-v0",
    map=get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map",
    num_agents=NUM_AGENTS,
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    params=params_dict
)
# --- Reset Environment ---
INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)    
obs, timestep, _, _ = env.reset(poses=INITIAL_POSES)
env.render(mode="human") # Render first to create the window/renderer

# --- Agent Setup ---

ORIGINAL_WEIGHT = "models/actor/pretrained/actor_pretrained.pt"
CRITIC_WEIGHT = "critic_gen_36.pt"

agent = PPOAgent(
    num_agents=NUM_AGENTS_AI, 
    map_name=CURRENT_MAP,
    steps=STEPS_PER_GENERATION,
    params=params_dict,
    transfer=[ORIGINAL_WEIGHT, CRITIC_WEIGHT],
    baseline_speed=10.0,
    tbtt_length=64,                    # TBTT chunk length (detach temporal state every N steps)
    checkpoint_every=32,               # Activation checkpointing every N steps within each chunk
)
pp_driver = MPCAgent(
    map_name=CURRENT_MAP,
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    horizon=8,
    speed_scale=0.8,
    emergency_dist=0.8,
    speed_clamp=7.5
)

if hasattr(torch, 'compile'):
    agent.actor_network = torch.compile(agent.actor_network)
    agent.critic_network = torch.compile(agent.critic_network)

# --- Critic Pretraining (live rollouts with real reward function) ---
ALL_PRETRAIN_MAPS = EASY_MAPS + MEDIUM_MAPS
if CRITIC_WEIGHT is None:
    agent.pretrain_critic(
        env=env,
        pp_driver=agent.mpc,
        num_agents_total=NUM_AGENTS,
        maps=ALL_PRETRAIN_MAPS,
        rollout_steps=8000,
        num_rollouts=len(ALL_PRETRAIN_MAPS),
        epochs=2,
        lr=5e-4,
        batch_size=256,
        # load_demos_path="demos/critic_demos.pt",
    )
    torch.save(agent.critic_network.state_dict(), f"models/critic/pretrained/critic_pretrained.pt")

# Generate a fresh random track to start training on
def _switch_to_new_track(gen_label="init"):
    """Generate a new random track, switch env/agents to it, and clean up the old one."""
    global CURRENT_MAP, _last_generated_track, INITIAL_POSES
    # Clean up previous generated track
    if _last_generated_track is not None:
        old_dir = os.path.join("maps", _last_generated_track)
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)
        _last_generated_track = None

    track_name = f"gen_track_{gen_label}"
    try:
        track_gen.generate(track_name)
        CURRENT_MAP = track_name
        _last_generated_track = track_name
    except RuntimeError as e:
        # Fallback to a random real map if generation fails
        print(f"Track generation failed ({e}), falling back to real map")
        available = EASY_MAPS + MEDIUM_MAPS + HARD_MAPS
        CURRENT_MAP = random.choice(available)
        _last_generated_track = None

    INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
    env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
    wp_xy, wp_s, rl = agent._load_waypoints(CURRENT_MAP)
    agent.waypoints_xy, agent.waypoints_s, agent.raceline_length = wp_xy, wp_s, rl
    pp_driver.update_map(CURRENT_MAP)

    agent.clear_experience_buffer()

    # Scale steps & buffer to 10× track length (meters → sim steps)
    _update_steps_and_buffer(rl)

    return INITIAL_POSES


def _update_steps_and_buffer(raceline_length):
    """Set STEPS_PER_GENERATION = 10 * track_length and resize
    actor/critic Mamba2 buffers + TBTT length to match.
    memory_length = 10 * track_length / stride  so the buffer
    covers the full time window via strided insertion."""
    global STEPS_PER_GENERATION
    new_steps = int(10 * raceline_length)
    new_steps = max(new_steps, 512)  # safety floor
    STEPS_PER_GENERATION = new_steps

    # Access _orig_mod to handle torch.compile wrapper
    actor_mod = getattr(agent.actor_network, '_orig_mod', agent.actor_network)
    critic_mod = getattr(agent.critic_network, '_orig_mod', agent.critic_network)
    # memory_length = total_window / stride
    stride = agent.stride
    mem_len = max(new_steps // stride, 16)  # safety floor
    actor_mod.memory_length = mem_len
    critic_mod.memory_length = mem_len
    agent.reset_buffers()
    print(f"  Steps/gen={new_steps}, buffer={mem_len} (stride={stride}, "
          f"track={raceline_length:.1f}m)")


_switch_to_new_track("init")
agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
obs, _, _, _ = env.reset(poses=INITIAL_POSES)

print(f"Starting training on {agent.device} for {TOTAL_TIMESTEPS} timesteps...")

best_avg_reward = -float('inf')
patience = 0

collision_timers = np.zeros(NUM_AGENTS, dtype=np.int32)
COLLISION_RESET_THRESHOLD = 32

total_steps_done = 0
gen = 0
while total_steps_done < TOTAL_TIMESTEPS:
    collisions = 0
    gen += 1
    print(f"\n--- Generation {gen} {CURRENT_MAP}  "
          f"(steps={STEPS_PER_GENERATION}, "
          f"total={total_steps_done}/{TOTAL_TIMESTEPS}) ---")
    total_reward_this_gen = []
    ego_reward_this_gen = []
    current_gen_time = 0.0
    
    for step in range(STEPS_PER_GENERATION):
        timer = time.time()
        done_np = np.zeros(NUM_AGENTS, dtype=np.int32)
        
        # env.render(mode="human")
        
        # Get Action from Agent
        scan_tensors, state_tensor = agent._obs_to_tensors(obs)
        action_tensor, log_prob_tensor, value_tensor = agent.get_action_and_value(
            scan_tensors, state_tensor
        )
                
        # Convert to NumPy for the Gym environment
        action_np = action_tensor.cpu().numpy()
        
        if action_np.shape[0] < NUM_AGENTS:
            # Fill in the remaining agents with PP actions
            pp_drive_action = pp_driver.get_actions_batch(obs)
            pp_drive_action = pp_drive_action.astype(np.float32)
            
            # Combine PP and Agent actions
            action_np = np.vstack((action_np, pp_drive_action[action_np.shape[0]:]))
        
        # Step the Environment
        next_obs, timestep, _, _ = env.step(action_np)
        
        # Calculate Reward
        rewards_list, avg_reward = agent.calculate_reward(next_obs)
        
        # Update collision timers
        current_collisions = np.array(next_obs['collisions'][:NUM_AGENTS])
        current_velocities = np.array(next_obs['linear_vels_x'][:NUM_AGENTS])
        collision_timers[(current_collisions == 1) | ((current_velocities < 0.1) & (current_velocities > -0.1))] += 1  # Increment for agents in collision
        collision_timers[current_collisions == 0] = 0  # Reset for agents not in collision
        
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
            agent.reset_buffers(agents_to_reset)
            agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2], agent_idxs=agents_to_reset)
            
            # Reset collision timers for these agents
            collision_timers[agents_to_reset] = 0
            
            # Count these as collision exits
            collisions += len(agents_to_reset[agents_to_reset < NUM_AGENTS_AI])
            
            done_np[agents_to_reset] = 1  # Mark these agents as done for this step
        
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
    current_physics_time = 0.0
    
    # --- END OF GENERATION ---
    # Flush the last pending transition with a bootstrap value estimate
    agent.finalize_rollout(obs)
    
    reward_avg = sum(total_reward_this_gen) / len(total_reward_this_gen)
    current_avg_ego_reward = sum(ego_reward_this_gen) / len(total_reward_this_gen)

    if reward_avg > best_avg_reward:
        torch.save(agent.actor_network.state_dict(), f"models/actor/best/actor_gen_{gen}.pt")
        torch.save(agent.critic_network.state_dict(), f"models/critic/best/critic_gen_{gen}.pt")
        best_avg_reward = reward_avg
        print(f"New best model saved with average reward: {reward_avg:.3f}")
        patience = 0
    elif gen % 100 == 0:
        torch.save(agent.actor_network.state_dict(), f"models/actor/checkpoint/actor_gen_{gen}.pt")
        torch.save(agent.critic_network.state_dict(), f"models/critic/checkpoint/critic_gen_{gen}.pt")
        print(f"Checkpoint saved at generation {gen}.")
        patience += 1
    else:
        patience += 1
        print(f"No improvement in average reward {reward_avg:.3f} vs {best_avg_reward:.3f}. Patience: {patience}")
        
    agent.learn(collisions, reward_avg)
    
        
    # if patience >= PATIENCE:
    #     print("Early stopping triggered due to no improvement.")
    #     break
    
    # --- Switch to a new random track every GEN_PER_MAP generations ---
    if gen % GEN_PER_MAP == 0:
        _switch_to_new_track(gen)
        print(f"Gen {gen}: New track \u2192 {CURRENT_MAP}  "
              f"(steps/gen={STEPS_PER_GENERATION})")
        agent.last_cumulative_distance = np.zeros(NUM_AGENTS_AI)
        agent.last_wp_index = np.zeros(NUM_AGENTS_AI, dtype=np.int32)
        env.reset(poses=INITIAL_POSES)
        agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
        agent.reset_buffers()
        

torch.save(agent.actor_network.state_dict(), f"models/actor/checkpoint/actor_gen_FINAL.pt")
torch.save(agent.critic_network.state_dict(), f"models/critic/checkpoint/critic_gen_FINAL.pt")
print(f"Checkpoint saved at final generation.")

# Clean up last generated track
if _last_generated_track is not None:
    old_dir = os.path.join("maps", _last_generated_track)
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir, ignore_errors=True)
        
# --- END OF TRAINING ---
env.close()
print("Training complete.")