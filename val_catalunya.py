"""
val_catalunya.py — Validate a D2PPO actor on Catalunya (training map).
======================================================================
Quick sanity check: run the loaded actor deterministically on Catalunya,
report lap count, lap times, and collisions.  Useful for verifying that
BC pretraining produced a functional policy before RL fine-tuning.

Usage:
    python val_catalunya.py --actor models/actor/pretrained/actor_pretrained.pt
    python val_catalunya.py --actor models/actor/best/actor_gen_3.pt --laps 3
"""
import argparse
import numpy as np
import torch
import gym

from D2PPO_agent import D2PPOAgent as PPOAgent
from utils.sim_config import LIDAR_BEAMS, LIDAR_FOV, SIM_PARAMS
from utils.utils import get_map_dir, generate_start_poses

PARAMS_DICT = SIM_PARAMS.copy()
LAP_TIMEOUT = 180.0       # seconds
COLLISION_TIMEOUT = 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", required=True, help="Path to actor .pt")
    p.add_argument("--critic", default="models/critic/pretrained/critic_pretrained.pt")
    p.add_argument("--map", default="Catalunya")
    p.add_argument("--laps", type=int, default=3)
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-deploy", action="store_true")
    p.add_argument("--ddim-steps", type=int, default=5)
    p.add_argument("--action-repeat", type=int, default=0)
    args = p.parse_args()

    map_name = args.map

    env = gym.make(
        "f110_gym:f110-v0",
        map=get_map_dir(map_name) + f"/{map_name}_map",
        num_agents=1,
        num_beams=LIDAR_BEAMS,
        fov=LIDAR_FOV,
        params=PARAMS_DICT,
    )

    poses = generate_start_poses(map_name, 1, race=True)
    obs, _, _, _ = env.reset(poses=poses)

    if args.render:
        env.render(mode="human")

    agent = PPOAgent(
        num_agents=1,
        map_name=map_name,
        steps=256,
        params=PARAMS_DICT,
        transfer=[args.actor, args.critic],
    )
    if not args.no_deploy:
        agent.deploy(action_repeat=args.action_repeat, ddim_steps=args.ddim_steps)

    print(f"\n{'='*60}")
    print(f"  Catalunya Validation")
    print(f"  Actor: {args.actor}")
    print(f"  Laps target: {args.laps}, DDIM steps: {args.ddim_steps}, "
            f"action_repeat: {args.action_repeat}")
    print(f"{'='*60}\n")

    lap_count = 0
    lap_start = 0.0
    lap_times = []
    collision_count = 0
    collision_dur = 0.0
    sim_time = 0.0
    speeds = []

    while sim_time < LAP_TIMEOUT:
        scan_t, state_t = agent._obs_to_tensors(obs)
        with torch.no_grad():
            action_t, _, _ = agent.get_action_and_value(
                scan_t, state_t, deterministic=True
            )
        action = action_t.cpu().numpy()

        next_obs, step_time, _, _ = env.step(action)
        sim_time += step_time
        speeds.append(float(next_obs["linear_vels_x"][0]))

        if args.render:
            env.render(mode="human")

        # Collision handling
        if next_obs["collisions"][0] == 1:
            collision_dur += step_time
            if collision_dur > COLLISION_TIMEOUT:
                collision_count += 1
                collision_dur = 0.0
                cur_poses = np.stack([
                    next_obs["poses_x"], next_obs["poses_y"],
                    next_obs["poses_theta"],
                ], axis=1)
                new_poses = generate_start_poses(
                    map_name, 1, agent_poses=cur_poses)
                next_obs, _, _, _ = env.reset(
                    poses=new_poses, agent_idxs=np.array([0]))
                agent.reset_buffers(np.array([0]))
        else:
            collision_dur = 0.0

        # Lap tracking
        new_laps = int(next_obs["lap_counts"][0])
        if new_laps > lap_count:
            lap_time = sim_time - lap_start
            lap_times.append(lap_time)
            print(f"  Lap {new_laps} done: {lap_time:.2f}s")
            lap_start = sim_time
            lap_count = new_laps
            if lap_count >= args.laps:
                break

        obs = next_obs

        v = obs["linear_vels_x"][0]
        print(f"  t={sim_time:5.1f}s  v={v:4.1f}  lap={lap_count}/{args.laps}  "
                f"collisions={collision_count}", end="\r")

    print()
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Laps completed : {lap_count}/{args.laps}")
    print(f"  Collisions     : {collision_count}")
    print(f"  Sim time       : {sim_time:.1f}s")
    if lap_times:
        print(f"  Avg lap time   : {np.mean(lap_times):.2f}s")
        print(f"  Best lap time  : {np.min(lap_times):.2f}s")
    if speeds:
        print(f"  Avg speed      : {np.mean(speeds):.2f} m/s")
        print(f"  Max speed      : {np.max(speeds):.2f} m/s")
    print(f"{'='*60}\n")

    if lap_count == 0:
        print("  ❌  Policy failed to complete even one lap on training map.")
        print("      Likely cause: pretraining did not learn a usable policy.")
    elif collision_count > args.laps:
        print("  ⚠️  Policy crashes frequently on training map.")
    else:
        print(f"  ✅  Policy is functional on Catalunya "
              f"({lap_count} laps, {collision_count} collisions).")


if __name__ == "__main__":
    main()
