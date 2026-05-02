import os
import sys
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
import numpy as np
from baselines.mpc_agent import MPCAgent
from baselines.pure_pursuit import PurePursuit
from baselines.gap_follow_pure_pursuit import GapFollowPurePursuit
from D2PPO_agent import D2PPOAgent
from analysis.bc_lstm_agent import BCLSTMAgent
from utils.utils import generate_start_poses, get_map_dir

BC_LSTM_WEIGHTS = "/home/WVU-AD/ads00024/racing_rl/actor_val_best.pt"
RL_D2PPO_WEIGHTS = "actor_best3_goodenough_for677.pt"

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

NUM_AGENTS_TEST = 1
NUM_OVERTAKE_AGENTS = 0
NUM_AGENTS = NUM_OVERTAKE_AGENTS + NUM_AGENTS_TEST
MAP_NAME = "Catalunya"

LIDAR_BEAMS = 1080
LIDAR_FOV = 4.7

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

env = None
slow_pp = None
failed_maps = set()
RACE_DATA = dict()


def _ordered_maps():
    maps = os.listdir("maps")
    if "BrandsHatchObs" in maps and "IMS" in maps:
        bh_idx = maps.index("BrandsHatchObs")
        ims_idx = maps.index("IMS")
        maps[bh_idx], maps[ims_idx] = maps[ims_idx], maps[bh_idx]
    return maps


def _make_env():
    race_env = gym.make(
        "f110_gym:f110-v0",
        map=get_map_dir(MAP_NAME) + f"/{MAP_NAME}_map",
        num_agents=NUM_AGENTS,
        num_beams=LIDAR_BEAMS,
        fov=LIDAR_FOV,
        params=params_dict,
        show_agent_ids=True,
    )
    poses = generate_start_poses(MAP_NAME, NUM_AGENTS, race=True)
    race_env.reset(poses=poses)
    race_env.render(mode='human')
    return race_env


def _make_safety_driver():
    return PurePursuit(
        map_name=MAP_NAME,
        lookahead_distance=1.5,
        wheelbase=params_dict['lf'] + params_dict['lr'],
        max_steering=params_dict['s_max'],
        max_speed=8.0,
        min_speed=2.0,
    )

def race(agent, map, lap_count=10, speed_cap=None, label=None):
    """Run *agent* on *map* and record the trajectory under RACE_DATA[label].

    Pass labels explicitly when comparing multiple wrappers with similar APIs.
    """
    if label is None:
        label = type(agent).__name__
    in_collision = True
    env.update_map(get_map_dir(map) + f"/{map}_map", ".png")
    if not isinstance(agent, NEURAL_AGENT_TYPES):
        agent.update_map(map)
    offset = 0.1

    while in_collision:
        initial_poses = generate_start_poses(map, NUM_AGENTS, race=True, race_offset=offset)
        offset += 0.1
        obs, _, _, _ = env.reset(poses=initial_poses)
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
        if obs['collisions'][0] == 1:
            collision_sim_time += env.timestep
            if collision_sim_time > 0.50:
                obs['col_exit'] = True
                RACE_DATA[label][map] += [obs]
                print(f"\n--- {label} stuck in collision for >0.5s on {map}, exiting ---")
                if current_lap == 0:
                    failed_maps.add(map + '-' + label)
                break
        else:
            collision_sim_time = 0.0

        if not isinstance(agent, NEURAL_AGENT_TYPES):
            action_np = [agent.get_actions_batch(obs)[0]]
        else:
            scan_tensors, state_tensor = agent._obs_to_tensors(obs)

            action_tensor, _, _ = agent.get_action_and_value(
                scan_tensors, state_tensor, deterministic=True
            )

            action_np = action_tensor.cpu().numpy()

        action_np = np.concatenate([action_np, slow_pp.get_actions_batch(obs)[NUM_AGENTS_TEST:]], axis=0)
        if speed_cap is not None:
            action_np[..., 1] = np.clip(action_np[..., 1], None, speed_cap)

        next_obs, step_time, _, _ = env.step(action_np)
        sim_time += step_time

        lap_time = sim_time - lap_start_sim_time
        next_obs['col_exit'] = False
        next_obs['lap_time'] = lap_time

        if next_obs['lap_counts'][0] > current_lap:
            current_lap = next_obs['lap_counts'][0]
            print(f"\n--- {label} Completed Lap {current_lap} on {map} in {lap_time:.2f}s ---")
            lap_start_sim_time = sim_time

        obs = next_obs.copy()

        del next_obs['scans']
        RACE_DATA[label][map].append(next_obs)

        print(f"{lap_time:.2f}s: Speed: {next_obs['linear_vels_x'][0]:.2f}", end='\r')

def _build_agents():
    bc_lstm = BCLSTMAgent(
        num_agents=NUM_AGENTS_TEST,
        weights_path=BC_LSTM_WEIGHTS,
    )

    d2ppo = D2PPOAgent(
        num_agents=NUM_AGENTS_TEST,
        map_name=MAP_NAME,
        steps=None,
        params=params_dict,
        transfer=[RL_D2PPO_WEIGHTS, None],
    )
    d2ppo.deploy(action_repeat=0, ddim_steps=5, compile_model=False)

    gfpp = GapFollowPurePursuit(
        map_name=MAP_NAME,
        wheelbase=params_dict['lf'] + params_dict['lr'],
        max_steering=params_dict['s_max'],
        num_beams=LIDAR_BEAMS,
        fov=LIDAR_FOV,
        **BEST_GFPP_PARAMS,
    )

    mpc = MPCAgent(
        map_name=MAP_NAME,
        wheelbase=params_dict['lf'] + params_dict['lr'],
        max_steering=params_dict['s_max'],
        max_accel=params_dict['a_max'],
        **BEST_MPC_PARAMS,
    )

    return [
        ("D2PPO", d2ppo),
        ("BC_LSTM", bc_lstm),
        ("GFPP", gfpp),
        ("MPC", mpc),
    ]


def _save_race_data():
    analysis_dir = os.path.join(os.path.dirname(__file__), "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    output_path = os.path.join(analysis_dir, f"race_data_{int(time.time())}.pkl")

    with open(output_path, "wb") as file:
        pickle.dump(RACE_DATA, file)


def main():
    global env, slow_pp, failed_maps, RACE_DATA

    maps = _ordered_maps()
    env = _make_env()
    slow_pp = _make_safety_driver()
    failed_maps = set()
    RACE_DATA = dict()
    agents_to_race = _build_agents()

    for label, agent in agents_to_race:
        RACE_DATA[label] = dict()
        for map_name in maps:
            RACE_DATA[label][map_name] = dict()
            race(agent, map_name, lap_count=3, speed_cap=None, label=label)

    _save_race_data()
    good_maps = [
        map_name for map_name in maps
        if all(map_name + '-' + label not in failed_maps for label, _ in agents_to_race)
    ]
    print(f"Successful races completed on maps: {good_maps}")
    print("All races complete.")


if __name__ == "__main__":
    main()
