from time import time
import gym
import numpy as np
from d2ppo_agent import PPOAgent
from baselines.pure_pursuit import PurePursuit
from supervised_agent import SupervisedAgent
from utils.utils import *

# --- Car Physics Parameters ---
desired_speed = 7.0
param_scalar = desired_speed / 7.0
params_dict = {'mu': 1.0489,
               'C_Sf': 4.718,
               'C_Sr': 5.4562,
               'lf': 0.15875,
               'lr': 0.17145,
               'h': 0.074,
               'm': 3.74,
               'I': 0.04712,
               's_min': -0.4189,
               's_max': 0.4189,
               'sv_min': -3.2,
               'sv_max': 3.2,
               'v_switch': 7.319,
               'a_max': 9.51,
               'v_min':-5.0 * param_scalar,
               'v_max': 20.0 * param_scalar,
               'width': 0.31,
               'length': 0.58
               }

# --- Race Parameters ---
NUM_AGENTS = 5
MODEL_AGENTS = 1
MAP_NAME = "BrandsHatch"
NUM_RACES = 1 # How many races to run
MAX_EPISODE_TIME = 25.0 # Max time in seconds before a race resets

# --- IMPORTANT: These MUST match your trained model ---
LIDAR_BEAMS = 1080
LIDAR_FOV = 4.7

# --- Environment Setup ---
env = gym.make(
    "f110_gym:f110-v0",
    map=get_map_dir(MAP_NAME) + f"/{MAP_NAME}_map",
    num_agents=NUM_AGENTS,
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    params=params_dict
)

# --- Agent Setup ---
agent = SupervisedAgent(
    num_agents=MODEL_AGENTS,
    map_name=MAP_NAME,
    steps=None,
    params=params_dict,
    transfer=["actor_Mamba2Simple_avoid.pt", "actor_LSTM_avoid.pt"]
    )
device = agent.device

# --- Pure Pursuit Controller Setup ---
pp_controller = PurePursuit(
    map_name=MAP_NAME,
    lookahead_distance=2,
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    target_speed=7.92
)
# pp_indices = np.random.choice(NUM_AGENTS, size=NUM_AGENTS-3, replace=False)

# --- Generate Starting Poses ---
INITIAL_POSES = generate_start_poses(MAP_NAME, NUM_AGENTS, race=True)

# --- Main Race Loop ---
for race in range(NUM_RACES):
    print(f"\n--- Starting Race {race+1} / {NUM_RACES} ---")
    
    obs, _, _, _ = env.reset(poses=INITIAL_POSES)
    current_physics_time = 0.0
    done = False
    collision_start = time()
    total_sim_time = 0.0
    
    while not done:
        env.render(mode='human')
        
        # Get Action from Agent
        scan_tensors, state_tensor, _ = agent._obs_to_tensors(obs)
        
        # --- Use deterministic=True for racing ---
        action_tensor, example_tensor = agent.get_action_and_value(
            scan_tensors, state_tensor
        )
        
        # Pure Pursuit Override
                
        # Convert to NumPy for the Gym environment
        model_np = action_tensor.cpu().numpy()
        action_np = pp_controller.get_actions_batch(obs, use_raceline_speed=True)
        action_np[range(0, MODEL_AGENTS)] = model_np  # Override only the selected agents with Pure Pursuit
        
        # Step the Environment
        next_obs, step_reward, done_from_env, info = env.step(action_np)
        total_sim_time += step_reward
        
        print(f"Time: {total_sim_time:.2f}s, Speed: {obs['linear_vels_x'][0]:.2f} m/s", end='\r')
        
        if obs['collisions'].sum() != 0:
            time_in_collision = time() - collision_start
            print(f"\n Agent in collision for {time_in_collision:.2f} seconds")
        else:
            collision_start = time()
            time_in_collision = 0.0        
               
        if time_in_collision > 5.0:
            print(f"\n Agent in collision for {time_in_collision:.2f} seconds")
            break
        
        # if done_from_env and obs['collisions'].sum() == 0:
        #     print(f"Race {race+1} finished, finished lap in: {total_sim_time / 2.0:.2f} seconds")
        #     break
        
        # Update observation for next loop
        obs = next_obs

# --- END OF RACING ---
env.close()
print("All races complete.")
