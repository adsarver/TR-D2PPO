import time
import gym
import numpy as np
from D2PPO_agent import D2PPOAgent as PPOAgent
# from utils.control_handler import ControlHandler
from baselines.pure_pursuit import PurePursuit
from utils.utils import *
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import random

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
NUM_AGENTS_AI = 8
NUM_AGENTS_PP = 4
NUM_AGENTS = NUM_AGENTS_AI + NUM_AGENTS_PP
EASY_MAPS = ["Hockenheim", "Monza", "Melbourne", "BrandsHatch"]
MEDIUM_MAPS = ["Oschersleben", "Sakhir", "Sepang", "SaoPaulo", "Budapest", "Catalunya", "Silverstone"]
HARD_MAPS = ["Zandvoort", "MoscowRaceway", "Austin", "Nuerburgring", "Spa", "YasMarina", "Sochi",]
TOTAL_TIMESTEPS = 12_000_000
STEPS_PER_GENERATION = 1024
LIDAR_BEAMS = 1080  # Default is 1080
LIDAR_FOV = 4.7   # Default is 4.7 radians (approx 270 deg)
INITIAL_POSES = None # Generated later
CURRENT_MAP = "Sepang" # Starting map, used for pretraining
PATIENCE = 200  # Early stopping patience
GEN_PER_MAP = 1

def get_curriculum_map_pool(generation, selector=None):
    """Returns the appropriate map pool based on training progress."""
    return EASY_MAPS + MEDIUM_MAPS + HARD_MAPS  # Phase 3: Full curriculum

# -- Environment Setup ---
env = gym.make(
    "f110_gym:f110-v0",
    map=get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map",
    num_agents=NUM_AGENTS,
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    params=params_dict
)

# --- Agent Setup ---
num_generations = TOTAL_TIMESTEPS // STEPS_PER_GENERATION

ORIGINAL_WEIGHT = "models/actor/pretrained/actor_pretrained.pt"
ACTOR_CHECKPOINT = f"models/actor/best/actor_gen_93.pt"
CRITIC_CHECKPOINT = f"models/critic/best/critic_gen_93.pt"

agent = PPOAgent(
    num_agents=NUM_AGENTS_AI, 
    map_name=CURRENT_MAP,
    steps=STEPS_PER_GENERATION,
    params=params_dict,
    transfer=[ORIGINAL_WEIGHT, None]
)
pp_driver = PurePursuit(
    map_name=CURRENT_MAP,
    lookahead_points=20,
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    max_speed=5.0,
    min_speed=1.5
)

if hasattr(torch, 'compile'):
    agent.actor_network = torch.compile(agent.actor_network)
    agent.critic_network = torch.compile(agent.critic_network)

STEPS_PER_GENERATION = int((agent.raceline_length / 7) * 201)  # Adjust steps based on raceline length and desired time per generation
agent.update_buffer_size(STEPS_PER_GENERATION)
print(f"Adjusted steps per generation to {STEPS_PER_GENERATION} based on raceline length.")

# --- Reset Environment ---
current_physics_time = 0.0

INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)    
obs, timestep, _, _ = env.reset(poses=INITIAL_POSES)
agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])

print(f"Starting training on {agent.device} for {TOTAL_TIMESTEPS} timesteps...")

# Render first to create the window/renderer
env.render(mode="human")

best_avg_reward = -float('inf')
patience = 0

collision_timers = np.zeros(NUM_AGENTS, dtype=np.int32)
COLLISION_RESET_THRESHOLD = 32

for gen in range(num_generations):
    collisions = 0
    print(f"\n--- Generation {gen+1} / {num_generations} {CURRENT_MAP} ---")
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
    
    print() # Finish the carriage return line
    current_physics_time = 0.0
    
    # --- END OF GENERATION ---
    reward_avg = sum(total_reward_this_gen) / len(total_reward_this_gen)
    current_avg_ego_reward = sum(ego_reward_this_gen) / len(total_reward_this_gen)

    
    if reward_avg > best_avg_reward:
        torch.save(agent.actor_network.state_dict(), f"models/actor/best/actor_gen_{gen+1}.pt")
        torch.save(agent.critic_network.state_dict(), f"models/critic/best/critic_gen_{gen+1}.pt")
        best_avg_reward = reward_avg
        print(f"New best model saved with average reward: {reward_avg:.3f}")
        patience = 0
    elif (gen + 1) % 100 == 0:
        torch.save(agent.actor_network.state_dict(), f"models/actor/checkpoint/actor_gen_{gen+1}.pt")
        torch.save(agent.critic_network.state_dict(), f"models/critic/checkpoint/critic_gen_{gen+1}.pt")
        print(f"Checkpoint saved at generation {gen+1}.")
        patience += 1
    else:
        patience += 1
        print(f"No improvement in average reward {reward_avg:.3f} vs {best_avg_reward:.3f}. Patience: {patience}")
        
    agent.learn(collisions, reward_avg)
    
        
    # if patience >= PATIENCE:
    #     print("Early stopping triggered due to no improvement.")
    #     break
    
    if (gen + 1) % GEN_PER_MAP == 0:
        available_maps = get_curriculum_map_pool(gen+1)
        next_map = CURRENT_MAP
        while next_map == CURRENT_MAP:
            next_map = random.choice(available_maps)
        CURRENT_MAP = next_map
        print(f"Gen {gen+1}: Map={CURRENT_MAP}, Pool size={len(available_maps)}")
        
        INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
        env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
        
        waypoints_xy, waypoints_s, raceline_length = agent._load_waypoints(CURRENT_MAP)
        agent.waypoints_xy = waypoints_xy
        agent.waypoints_s = waypoints_s
        agent.raceline_length = raceline_length
        agent.last_cumulative_distance = np.zeros(NUM_AGENTS_AI) 
        agent.last_wp_index = np.zeros(NUM_AGENTS_AI, dtype=np.int32)
        
        sps = 1 / (time.time() - timer)
        sps = sps if sps > 1 else 100
        STEPS_PER_GENERATION = int((raceline_length / 7) * sps)
        agent.update_buffer_size(STEPS_PER_GENERATION)
        print(f"Adjusted steps per generation to {STEPS_PER_GENERATION} based on raceline length.")

        env.reset(poses=INITIAL_POSES)
        agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
        agent.reset_buffers()
        

torch.save(agent.actor_network.state_dict(), f"models/actor/checkpoint/actor_gen_FINAL.pt")
torch.save(agent.critic_network.state_dict(), f"models/critic/checkpoint/critic_gen_FINAL.pt")
print(f"Checkpoint saved at final generation.")
        
# --- END OF TRAINING ---
env.close()
print("Training complete.")