import os
import sys
import time
import pickle
import queue
import traceback
import multiprocessing as mp

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
NUM_OVERTAKE_AGENTS = 5
NUM_AGENTS = NUM_OVERTAKE_AGENTS + NUM_AGENTS_TEST
MAP_NAME = "Catalunya"
NUM_LAPS = 1

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
RENDER_RACES = os.getenv("TR_RENDER_RACES", "0") == "1"
STRICT_RACE_ERRORS = os.getenv("TR_STRICT_RACE_ERRORS", "0") == "1"


def _ordered_maps():
    maps = os.listdir("maps")
    if "BrandsHatchObs" in maps and "IMS" in maps:
        bh_idx = maps.index("BrandsHatchObs")
        ims_idx = maps.index("IMS")
        maps[bh_idx], maps[ims_idx] = maps[ims_idx], maps[bh_idx]
    return maps


def _make_env(render=False):
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
    if render:
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


def _as_action_batch(action, source):
    action_np = np.asarray(action, dtype=np.float32)
    if action_np.ndim == 1:
        if action_np.size != 2:
            raise ValueError(f"{source} returned action with shape {action_np.shape}")
        action_np = action_np.reshape(1, 2)
    elif action_np.ndim > 2:
        action_np = action_np.reshape(-1, action_np.shape[-1])

    if action_np.ndim != 2 or action_np.shape[1] != 2:
        raise ValueError(f"{source} returned action batch with shape {action_np.shape}")
    return action_np


def race(agent, map, env, slow_pp, race_data, failed_maps, lap_count=10,
         speed_cap=None, label=None, render=False):
    """Run *agent* on *map* and record the trajectory under RACE_DATA[label].

    Pass labels explicitly when comparing multiple wrappers with similar APIs.
    """
    if label is None:
        label = type(agent).__name__
    in_collision = True

    env.render(mode='human')
    env.update_map(get_map_dir(map) + f"/{map}_map", ".png")

    if not isinstance(agent, NEURAL_AGENT_TYPES):
        agent.update_map(map)
    if slow_pp is not None:
        slow_pp.update_map(map)
    offset = 0.1

    while in_collision:
        initial_poses = generate_start_poses(map, NUM_AGENTS, race=True, race_offset=offset)
        offset += 0.1
        obs, _, _, _ = env.reset(poses=initial_poses)
        in_collision = obs['collisions'][0] != 0
        if render:
            env.render(mode='human')

    if isinstance(agent, NEURAL_AGENT_TYPES):
        agent.reset_buffers()

    print(f"\n--- {label} Starting Lap 0 / {lap_count} on {map} ---")

    obs['col_exit'] = False
    race_data[label][map] = [obs]

    sim_time = 0.0
    lap_start_sim_time = 0.0
    current_lap = 0
    collision_sim_time = 0.0

    while current_lap < lap_count and sim_time - lap_start_sim_time < 200.0:
        if obs['collisions'][0] == 1:
            collision_sim_time += env.timestep
            if collision_sim_time > 0.50:
                obs['col_exit'] = True
                race_data[label][map] += [obs]
                print(f"\n--- {label} stuck in collision for >0.5s on {map}, exiting ---")
                if current_lap == 0:
                    failed_maps.add(map + '-' + label)
                break
        else:
            collision_sim_time = 0.0

        if not isinstance(agent, NEURAL_AGENT_TYPES):
            action_np = _as_action_batch(
                agent.get_actions_batch(obs),
                f"{label}.get_actions_batch",
            )[:NUM_AGENTS_TEST]
        else:
            scan_tensors, state_tensor = agent._obs_to_tensors(obs)
            action_tensor, _, _ = agent.get_action_and_value(
                scan_tensors, state_tensor, deterministic=True
            )
            action_np = _as_action_batch(action_tensor.cpu().numpy(), label)

        if NUM_OVERTAKE_AGENTS > 0:
            action_np = np.concatenate(
                [
                    action_np,
                    _as_action_batch(
                        slow_pp.get_actions_batch(obs),
                        "safety_driver.get_actions_batch",
                    )[NUM_AGENTS_TEST:],
                ],
                axis=0,
            )
        if speed_cap is not None:
            action_np[..., 1] = np.clip(action_np[..., 1], None, speed_cap)

        action_np = _as_action_batch(action_np, label)
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
        race_data[label][map].append(next_obs)

        print(f"{lap_time:.2f}s: Speed: {next_obs['linear_vels_x'][0]:.2f}", end='\r')

def _agent_labels():
    return ["D2PPO", "BC_LSTM", "GFPP", "MPC"]


def _build_agent(label):
    if label == "BC_LSTM":
        return BCLSTMAgent(
            num_agents=NUM_AGENTS_TEST,
            weights_path=BC_LSTM_WEIGHTS,
        )

    if label == "D2PPO":
        d2ppo = D2PPOAgent(
            num_agents=NUM_AGENTS_TEST,
            map_name=MAP_NAME,
            steps=None,
            params=params_dict,
            transfer=[RL_D2PPO_WEIGHTS, None],
        )
        d2ppo.deploy(action_repeat=0, ddim_steps=5, compile_model=False)
        return d2ppo

    if label == "GFPP":
        return GapFollowPurePursuit(
            map_name=MAP_NAME,
            wheelbase=params_dict['lf'] + params_dict['lr'],
            max_steering=params_dict['s_max'],
            num_beams=LIDAR_BEAMS,
            fov=LIDAR_FOV,
            **BEST_GFPP_PARAMS,
        )

    if label == "MPC":
        return MPCAgent(
            map_name=MAP_NAME,
            wheelbase=params_dict['lf'] + params_dict['lr'],
            max_steering=params_dict['s_max'],
            max_accel=params_dict['a_max'],
            **BEST_MPC_PARAMS,
        )

    raise ValueError(f"Unknown agent label: {label}")


def _race_worker(label, maps, result_queue, render=False):
    race_env = None
    race_data = {label: dict()}
    local_failed_maps = set()
    errors = []
    try:
        race_env = _make_env(render=render)
        safety_driver = _make_safety_driver()
        agent = _build_agent(label)

        for map_name in ["SaoPaulo"]:
            race_data[label][map_name] = dict()
            try:
                race(
                    agent,
                    map_name,
                    race_env,
                    safety_driver,
                    race_data,
                    local_failed_maps,
                    lap_count=NUM_LAPS,
                    speed_cap=None,
                    label=label,
                    render=render,
                )
            except Exception:
                local_failed_maps.add(map_name + '-' + label)
                errors.append({
                    "label": label,
                    "map": map_name,
                    "traceback": traceback.format_exc(),
                })
                print(f"\n--- {label} failed on {map_name}; keeping partial data and continuing ---")

        result_queue.put({
            "label": label,
            "race_data": race_data,
            "failed_maps": sorted(local_failed_maps),
            "errors": errors,
        })
    except Exception:
        result_queue.put({
            "label": label,
            "race_data": race_data,
            "failed_maps": sorted(local_failed_maps),
            "errors": errors + [{
                "label": label,
                "map": None,
                "traceback": traceback.format_exc(),
            }],
        })
    finally:
        if race_env is not None:
            try:
                race_env.close()
            except Exception:
                pass


def _save_race_data(race_data):
    analysis_dir = os.path.join(os.path.dirname(__file__), "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    output_path = os.path.join(analysis_dir, f"race_data_{int(time.time())}.pkl")

    with open(output_path, "wb") as file:
        pickle.dump(race_data, file)

    return output_path


def _save_error_report(errors, output_path):
    if not errors:
        return None

    report_path = os.path.splitext(output_path)[0] + "_errors.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        for error in errors:
            label = error.get("label", "unknown")
            map_name = error.get("map") or "worker startup/shutdown"
            file.write(f"--- {label} failed on {map_name} ---\n")
            file.write(error.get("traceback", ""))
            if not file.tell() or not str(error.get("traceback", "")).endswith("\n"):
                file.write("\n")
            file.write("\n")
    return report_path


def main():
    maps = _ordered_maps()
    labels = _agent_labels()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for label in labels:
        process = ctx.Process(
            target=_race_worker,
            args=(label, maps, result_queue, RENDER_RACES),
            name=f"paper-data-{label}",
        )
        process.start()
        processes.append((label, process))

    race_data = dict()
    failed_maps = set()
    errors = []
    pending_labels = set(labels)

    while pending_labels:
        try:
            result = result_queue.get(timeout=5.0)
        except queue.Empty:
            for label, process in processes:
                if label in pending_labels and not process.is_alive():
                    pending_labels.remove(label)
                    race_data.setdefault(label, dict())
                    errors.append({
                        "label": label,
                        "map": None,
                        "traceback": (
                            f"Worker exited before returning results "
                            f"(exitcode={process.exitcode}).\n"
                        ),
                    })
            continue

        label = result["label"]
        pending_labels.discard(label)
        race_data.update(result["race_data"])
        failed_maps.update(result["failed_maps"])
        errors.extend(result.get("errors", []))

    for _, process in processes:
        process.join()

    output_path = _save_race_data(race_data)
    error_report_path = _save_error_report(errors, output_path)
    good_maps = [
        map_name for map_name in maps
        if all(map_name + '-' + label not in failed_maps for label in labels)
    ]
    print(f"Saved race data to {output_path}")
    if error_report_path:
        print(f"Saved error report to {error_report_path}")
        for error in errors:
            label = error.get("label", "unknown")
            map_name = error.get("map") or "worker startup/shutdown"
            print(f"\n--- {label} failed on {map_name} ---\n{error.get('traceback', '')}")
    print(f"Successful races completed on maps: {good_maps}")
    if errors:
        print(f"Race collection completed with {len(errors)} error(s); partial data was saved.")
        if STRICT_RACE_ERRORS:
            raise RuntimeError(f"{len(errors)} race error(s); partial data saved to {output_path}")
    else:
        print("All races complete.")


if __name__ == "__main__":
    main()
