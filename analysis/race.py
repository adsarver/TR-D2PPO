import gym
import numpy as np
from baselines.pure_pursuit import PurePursuit
from supervised_agent import SupervisedAgent
import torch
import os
from utils.utils import *

# --- Car Physics Parameters ---
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
               'v_switch':7.319,
               'a_max': 9.51,
               'v_min':-5.0,
               'v_max': 20.0,
               'width': 0.31,
               'length': 0.58
               }

# --- Race Parameters ---
NUM_AGENTS = 3
MAP_NAME = "BrandsHatch"
NUM_RACES = 5 # How many races to run
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
    params=params_dict,
    show_agent_ids=True
)

# --- Agent Setup ---
agent = SupervisedAgent(
    num_agents=NUM_AGENTS,
    map_name=MAP_NAME,
    steps=None,
    params=params_dict,
    transfer=["actor_Mamba2Simple_avoid.pt", "actor_BrandsHatch.pt", "actor_BrandsHatch.pt"]
    )
device = agent.device

pp_driver = PurePursuit(
    map_name=MAP_NAME,
    lookahead_points=4,
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    max_speed=7.0,
    min_speed=2.0
)

# --- Generate Starting Poses ---
INITIAL_POSES = generate_start_poses(MAP_NAME, NUM_AGENTS, race=True)

# --- Main Race Loop ---
for race in range(NUM_RACES):
    print(f"\n--- Starting Race {race+1} / {NUM_RACES} ---")
    
    obs, _, _, _ = env.reset(poses=INITIAL_POSES)
    current_physics_time = 0.0
    done = False

    while not done:
        env.render(mode='human')
        
        # Get Action from Agent
        scan_tensors, state_tensor, _ = agent._obs_to_tensors(obs)
        
        # --- Use deterministic=True for racing ---
        action_tensor, example_tensor, _ = agent.get_action_and_value(
            scan_tensors, state_tensor
        )
        
        pp_action = pp_driver.get_actions_batch(obs)
        example_tensor = example_tensor.cpu().numpy()
                
        # Convert to NumPy for the Gym environment
        action_np = action_tensor.cpu().numpy()
        
        action_np[1] = example_tensor[1]
        action_np[2] = pp_action[2]
        
        # Step the Environment
        next_obs, step_reward, done_from_env, info = env.step(action_np)
        agents_to_reset = np.where(obs['collisions'] == 1)[0]
        if agents_to_reset.size > 0:
            # Generate new poses for stuck agents
            poses = np.array([[x, y, theta] for x, y, theta in zip(
                next_obs['poses_x'], next_obs['poses_y'], next_obs['poses_theta']
            )])
            INITIAL_POSES = generate_start_poses(MAP_NAME, NUM_AGENTS, agent_poses=poses)
            
            # Reset the environment for stuck agents
            next_obs, _, _, _ = env.reset(poses=INITIAL_POSES, agent_idxs=agents_to_reset)
            
            # Reset agent buffers and trackers
            agent.reset_buffers(agents_to_reset)
            print(f"Collision detected for agents {agents_to_reset}. Resetting their positions.")
        
        print(f"Mamba2: {obs['linear_vels_x'][0]:.2f} m/s - Pure Pursuit: {obs['linear_vels_x'][2]:.2f} m/s - LSTM: {obs['linear_vels_x'][1]:.2f} m/s", end='\r')
        
        # Update observation for next loop
        obs = next_obs

# --- END OF RACING ---
env.close()
print("All races complete.")
