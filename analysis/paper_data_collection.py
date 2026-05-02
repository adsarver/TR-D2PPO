import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
import numpy as np
from baselines.mpc_agent import MPCAgent
from baselines.pure_pursuit import PurePursuit
from baselines.gap_follow import GapFollow
from baselines.gap_follow_pure_pursuit import GapFollowPurePursuit
from baselines.sim_pure_pursuit import SimPurePursuit
from D2PPO_agent import D2PPOAgent
from analysis.bc_lstm_agent import BCLSTMAgent
import torch
from utils.utils import *
import time
import pickle

# Path to the supervised BC-pretrained LSTM checkpoint (racing_rl repo).
BC_LSTM_WEIGHTS = "/home/WVU-AD/ads00024/racing_rl/actor_val_best.pt"

# Tuple of agent classes that share the D2PPO-style neural API
# (``_obs_to_tensors`` + ``get_action_and_value``).  Used in race() to
# dispatch between MPC/PP-style ``get_actions_batch`` agents and
# neural-policy agents.
NEURAL_AGENT_TYPES = (D2PPOAgent, BCLSTMAgent)

BEST_GFPP_PARAMS = {
    "lookahead_distance": 1.4,
    "threshold_at_v_min": 1.0,
    "threshold_at_v_max": 2.5,
}
BEST_MPC_PARAMS = {
    "horizon": 8,
    "speed_scale": 0.8,
    "emergency_dist": 0.8,
}

# --- Race Parameters ---
NUM_AGENTS_TEST = 1
NUM_OVERTAKE_AGENTS = 0
NUM_AGENTS = NUM_OVERTAKE_AGENTS + NUM_AGENTS_TEST  # Total agents including the main one
MAPS = os.listdir("maps")
if "BrandsHatchObs" in MAPS and "IMS" in MAPS:
    bh_idx = MAPS.index("BrandsHatchObs")
    ims_idx = MAPS.index("IMS")
    MAPS[bh_idx], MAPS[ims_idx] = MAPS[ims_idx], MAPS[bh_idx]
# MAPS = ["BrandsHatch", "Catalunya", "Nuerburgring"]

EASY_MAPS = ["Monza", "BrandsHatch"]
MEDIUM_MAPS = ["Sakhir", "SaoPaulo", "Budapest"]
HARD_MAPS = ["MoscowRaceway", "Spielberg", "Oschersleben", "Catalunya", "Budapest"]

# MAPS = [m for m in MAPS if m not in EASY_MAPS + MEDIUM_MAPS + HARD_MAPS]
# MAPS = ["Catalunya"]
MAP_NAME = "Catalunya"
RACE_DATA = dict()
NUM_RACES = 10

LIDAR_BEAMS = 1080
LIDAR_FOV = 4.7

# --- Car Physics Parameters ---
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

INITIAL_POSES = generate_start_poses(MAP_NAME, NUM_AGENTS, race=True)
obs, _, _, _ = env.reset(poses=INITIAL_POSES)
env.render(mode='human')

failed_maps = set()

slow_pp = PurePursuit(
    map_name=MAP_NAME,
    lookahead_distance=1.5,        # 1.5 m base — tighter tracking near walls
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    max_speed=8.0,                # Limit speed for initial collision avoidance testing
    min_speed=2.0
)

def race(agent, map, lap_count=10, speed_cap=None, old_lstm=False, label=None):
    """Run *agent* on *map* and record the trajectory under RACE_DATA[label].

    ``label`` defaults to ``type(agent).__name__`` for backwards compatibility,
    but should be passed explicitly when comparing two agents of the same
    class (e.g. BC-LSTM vs D2PPO, both wrapped in :class:`D2PPOAgent`).
    """
    if label is None:
        label = type(agent).__name__
    in_collision = True
    env.update_map(get_map_dir(map) + f"/{map}_map", ".png")
    if not isinstance(agent, NEURAL_AGENT_TYPES):
        agent.update_map(map)
    offset = 0.1
    
    while in_collision:
        INITIAL_POSES = generate_start_poses(map, NUM_AGENTS, race=True, race_offset=offset)
        offset += 0.1
        obs, _, _, _ = env.reset(poses=INITIAL_POSES)
        in_collision = obs['collisions'][0] != 0
        env.render(mode='human')
            
    if isinstance(agent, NEURAL_AGENT_TYPES):
        agent.reset_buffers()
        
    print(f"\n--- {label} Starting Lap 0 / {lap_count} on {map} ---")
    
    obs['col_exit'] = False
    RACE_DATA[label][map] = [obs]

    sim_time = 0.0
    lap_start_sim_time = 0.0
    current_lap = 0
    collision_sim_time = 0.0
    
    while current_lap < lap_count and sim_time - lap_start_sim_time < 200.0: 
        # Exit if agent has been in collision for more than 1 second (sim time)
        if obs['collisions'][0] == 1:
            collision_sim_time += env.timestep
            if collision_sim_time > 0.50:
                obs['col_exit'] = True
                RACE_DATA[label][map] += [obs]
                print(f"\n--- {label} stuck in collision for >0.5s on {map}, exiting ---")
                if current_lap == 0: failed_maps.add(map + '-' + label)
                break
        else:
            collision_sim_time = 0.0
            
        env.render(mode='human_fast')
        
        # Get Action from Agent
        if not isinstance(agent, NEURAL_AGENT_TYPES):
            action_np = [agent.get_actions_batch(obs)[0]]
        else:
            scan_tensors, state_tensor = agent._obs_to_tensors(obs)
            
            action_tensor, _, _ = agent.get_action_and_value(
                scan_tensors, state_tensor, deterministic=True
            )
                                
            # Convert to NumPy for the Gym environment
            
            action_np = action_tensor.cpu().numpy()
                        
            # if current_lap > 0:
            #     action_np[..., 1] *= 1.0
            # else:
            #     action_np[..., 1] *= 0.5

        action_np = np.concatenate([action_np, slow_pp.get_actions_batch(obs)[NUM_AGENTS_TEST:]], axis=0)
        if speed_cap is not None:
            action_np[..., 1] = np.clip(action_np[..., 1], None, speed_cap)
            
        # Step the Environment
        next_obs, step_time, _, _ = env.step(action_np)
        sim_time += step_time
        
        lap_time = sim_time - lap_start_sim_time
        next_obs['col_exit'] = False
        next_obs['lap_time'] = lap_time
        
        if next_obs['lap_counts'][0] > current_lap:
            current_lap = next_obs['lap_counts'][0]
            print(f"\n--- {label} Completed Lap {current_lap} on {map} in {lap_time:.2f}s ---")
            lap_start_sim_time = sim_time
        
        # Update observation for next loop
        obs = next_obs.copy()
        
        del next_obs['scans']
        RACE_DATA[label][map].append(next_obs)
    
        print(f"{lap_time:.2f}s: Speed: {next_obs['linear_vels_x'][0]:.2f}", end='\r')

# --- Agent Setup ---
# BC-LSTM baseline: racing_rl's pretrained ExampleNetwork (LSTM)
# checkpoint, loaded via a thin inference-only wrapper.  This is the
# supervised reference for the BC-vs-RL comparison plots.
bc_lstm = BCLSTMAgent(
    num_agents=NUM_AGENTS_TEST,
    weights_path=BC_LSTM_WEIGHTS,
)
device = bc_lstm.device

# D2PPO RL-trained actor: same architecture, RL-fine-tuned weights
# (latest checkpoint).  This is the model under evaluation.
d2ppo = D2PPOAgent(
    num_agents=NUM_AGENTS_TEST,
    map_name=MAP_NAME,
    steps=None,
    params=params_dict,
    transfer=['actor_best3_goodenough_for677.pt', None]
)
# Switch the diffusion actor into deploy mode: 5 DDIM steps, no action
# repeat, fp16 inference.  Use ``ddim_steps=1`` for the paper's
# fastest setting (single denoiser call, jumps straight to x_0).
d2ppo.deploy(action_repeat=0, ddim_steps=5, compile_model=False)

# Backwards-compatible alias for any downstream code still expecting
# ``supervised``.
supervised = bc_lstm

pp_driver = PurePursuit(
    map_name=MAP_NAME,
    lookahead_distance=1.5,        # 1.5 m base — tighter tracking near walls
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    max_speed=10.0,
    min_speed=1.0
)

ppsim_driver = SimPurePursuit(
    map_name=MAP_NAME,
)

gap_follow = GapFollow(
    map_name=MAP_NAME,
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    max_speed=8.0,
    min_speed=1.0,
    hysteresis=0.2,
    max_sight=8.0
)

gfpp = GapFollowPurePursuit(
    map_name=MAP_NAME,
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    **BEST_GFPP_PARAMS
)

mpc = MPCAgent(
    map_name=MAP_NAME,
    wheelbase=params_dict['lf'] + params_dict['lr'],
    max_steering=params_dict['s_max'],
    max_accel=params_dict['a_max'],
    **BEST_MPC_PARAMS
)

speed_limits = [4.0, 8.0, 10.0, 12.0]

# Compare BC-LSTM (supervised baseline) vs D2PPO (RL-trained).  Each
# entry is (label, agent); the label keys RACE_DATA so two agents of
# the same class don't collide.
AGENTS_TO_RACE = [
    ("D2PPO",   d2ppo),
    # ("BC_LSTM", bc_lstm),
]

# for speed_cap in speed_limits:
for label, agent in AGENTS_TO_RACE:
    # if type(agent) != D2PPOAgent: agent.max_speed = speed_cap
    RACE_DATA[label] = dict()
    for map in MAPS:
        RACE_DATA[label][map] = dict()
        race(agent, map, lap_count=3, speed_cap=None, old_lstm=True, label=label)

analysis_dir = os.path.join(os.path.dirname(__file__), "analysis")
os.makedirs(analysis_dir, exist_ok=True)
output_path = os.path.join(analysis_dir, f"race_data_{int(time.time())}.pkl")

with open(output_path, "wb") as f:
    pickle.dump(RACE_DATA, f)

good_maps = [
    m for m in MAPS
    if all(m + '-' + lbl not in failed_maps for lbl, _ in AGENTS_TO_RACE)
]
print(f"Successful races completed on maps: {good_maps}")
# --- END OF RACING ---
print("All races complete.")
