"""
Minimal reproducer for segfault at ~gen 65.
Rapidly switches maps with the env + track generator to trigger the crash.
"""
import gc
import os
import sys
import time
import shutil
import signal
import faulthandler

import numpy as np
import gym
from track_generator import TrackGenerator
from utils.utils import generate_start_poses, get_map_dir

# Enable faulthandler to print a traceback on segfault
faulthandler.enable()

params_dict = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562,
               'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
               'm': 3.74, 'I': 0.04712, 's_min': -0.34, 's_max': 0.34,
               'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319,
               'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
               'width': 0.31, 'length': 0.58}

NUM_AGENTS = 10
LIDAR_BEAMS = 1080
LIDAR_FOV = 4.7
STEPS_PER_SWITCH = 50  # Minimal steps per map, just enough to exercise scan sim

track_gen = TrackGenerator(
    min_track_length=50, max_track_length=600,
    min_turns=6, max_turns=35,
    min_track_width=0.5, max_track_width=2.0,
    min_turn_radius=3.0, seed=None,
)

CURRENT_MAP = "Hockenheim"
env = gym.make(
    "f110_gym:f110-v0",
    map=get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map",
    num_agents=NUM_AGENTS,
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    params=params_dict,
)

poses = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
obs, _, _, _ = env.reset(poses=poses)
# Patch out the renderer since we don't have a display
# The real training calls env.render() which initializes the renderer
from f110_gym.envs.f110_env import F110Env
class _DummyRenderer:
    def update_map(self, *a, **kw): pass
F110Env.renderer = _DummyRenderer()

last_gen_track = None
REAL_MAPS = ["Hockenheim", "Monza", "Melbourne", "BrandsHatch",
             "Oschersleben", "Sakhir", "Sepang", "SaoPaulo",
             "Budapest", "Catalunya", "Silverstone"]

print(f"Starting rapid map-switch stress test ({STEPS_PER_SWITCH} steps/map)...")
print(f"PID: {os.getpid()}")

for switch_idx in range(200):  # Enough switches to reproduce
    # Alternate between generated and real maps
    if switch_idx % 2 == 0:
        # Generated track
        if last_gen_track is not None:
            old_dir = os.path.join("maps", last_gen_track)
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir, ignore_errors=True)
            last_gen_track = None
        
        track_name = f"debug_track_{switch_idx}"
        try:
            track_gen.generate(track_name)
            CURRENT_MAP = track_name
            last_gen_track = track_name
        except RuntimeError as e:
            print(f"  Track gen failed ({e}), using real map")
            CURRENT_MAP = REAL_MAPS[switch_idx % len(REAL_MAPS)]
    else:
        CURRENT_MAP = REAL_MAPS[switch_idx % len(REAL_MAPS)]

    poses = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
    env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
    obs, _, _, _ = env.reset(poses=poses)

    # Run a few steps to exercise the scan simulator
    for step in range(STEPS_PER_SWITCH):
        action = np.zeros((NUM_AGENTS, 2))
        action[:, 1] = 3.0  # Low speed forward
        obs, _, _, _ = env.step(action)
    
    mem_mb = 0
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_mb = int(line.split()[1]) / 1024
                    break
    except:
        pass
    
    print(f"Switch {switch_idx+1}/200: {CURRENT_MAP} - OK (RSS: {mem_mb:.0f} MB)")
    
    # Force GC every 10 switches
    if switch_idx % 10 == 0:
        gc.collect()

# Cleanup
if last_gen_track is not None:
    old_dir = os.path.join("maps", last_gen_track)
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir, ignore_errors=True)

env.close()
print("Stress test completed without segfault!")
