# TR-D2PPO: Diffusion Policy Optimization for Autonomous Racing

A research implementation of **D²PPO** (Diffusion Policy with Dispersive loss + PPO) for autonomous racing on the [F1TENTH](https://f1tenth.org/) platform. The agent learns to race on diverse track maps using a diffusion-based policy combined with proximal policy optimization (PPO) and a curriculum over map difficulty.

---

## Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Behavioral Cloning Pretraining](#1-behavioral-cloning-pretraining)
  - [2. RL Training](#2-rl-training)
  - [3. Evaluation](#3-evaluation)
- [Training Pipeline](#training-pipeline)
- [Reward Shaping](#reward-shaping)
- [Project Structure](#project-structure)
- [Maps and Curriculum](#maps-and-curriculum)
- [Baselines](#baselines)
- [Configuration](#configuration)
- [Requirements](#requirements)

---

## Overview

TR-D2PPO trains reinforcement learning agents to race on F1-style tracks. Instead of the typical Gaussian policy used in PPO, it uses a **DDPM (Denoising Diffusion Probabilistic Model)** as the action distribution, which enables richer, multimodal exploration and smoother driving behaviour.

Training runs up to 8 AI agents simultaneously alongside 4 Pure Pursuit baseline drivers. A curriculum progressively exposes agents to harder tracks as performance improves.

---

## Key Innovations

| Feature | Description |
|---------|-------------|
| **Diffusion Policies** | Replaces Gaussian actor with DDPM for expressive multimodal action distributions |
| **Dispersive Loss** | InfoNCE-style contrastive penalty on denoiser features to prevent mode collapse |
| **Advantage-Weighted Regression (AWR)** | Weights diffusion MSE loss by exponential of the PPO advantage |
| **Temporal Backbone** | LSTM or Mamba2 encoder maintains a rolling memory window for long-horizon planning |
| **Multi-Agent Curriculum** | Trains concurrently on 8 agents with difficulty-tiered map switching |
| **Critic Pretraining** | Separate supervised value-network pretraining on collected rollouts |
| **DDIM Inference** | Uses 5-step DDIM sampling at inference for fast action generation |

---

## Architecture

```
Input per step (50 Hz):
  ├─ LIDAR scan (1 080 beams)
  └─ Vehicle state [vx, vy, yaw-rate, accel]
           │
           ▼
  VisionEncoder (1-D CNN on LIDAR scan)
           │
           ▼
  Feature projection + state concat
           │
           ▼
  LSTM / Mamba2  (temporal memory, window = 64 steps)
           │  obs_features
           ▼
  ConditionalDenoisingMLP  (noise prediction ε_θ)
           │
           ▼
  DDIM reverse process (k = 5 denoising steps)
           │
           ▼
  Action: [steering ∈ [-0.34, 0.34] rad,
           speed   ∈ [0, 20] m/s]
```

The critic shares the same LSTM encoder and predicts a scalar value estimate for each observation.

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/adsarver/TR-D2PPO
cd TR-D2PPO
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Hardware note**: Training requires an NVIDIA GPU with CUDA 12.8 support. The code makes heavy use of `torch.compile()` for performance.

---

## Usage

### 1. Behavioral Cloning Pretraining

Initialises the diffusion policy by imitating the GapFollowPurePursuit expert baseline.

**Option A – Use pre-generated demonstrations:**

```bash
python weight_initializer.py --load demos/expert_demos.pt
```

**Option B – Collect new demonstrations on-the-fly:**

```bash
python weight_initializer.py
```

This saves pretrained actor weights that are loaded automatically by `train.py`.

### 2. RL Training

```bash
python train.py
```

- Runs 8 D2PPO agents + 4 Pure Pursuit baselines concurrently
- Applies a map curriculum: **Easy → Medium → Hard** (60 generations per map)
- Checkpoints are saved to `models/actor/` and `models/critic/`
- Training diagnostics and plots are written to `plots/`

### 3. Evaluation

```bash
python analysis/race.py
```

Loads a trained checkpoint and evaluates it on held-out tracks, collecting lap times, collision counts, and average speed.

---

## Training Pipeline

```
1. Behavioral Cloning Pretraining  (weight_initializer.py)
   ├─ Collect GapFollowPurePursuit demos on diverse maps
   ├─ Supervise DiffusionLSTM with:
   │     L = L_diffusion + λ · L_dispersive
   └─ Save pretrained actor weights

2. Critic Pretraining  (first phase of train.py)
   ├─ Freeze pretrained actor, run rollouts on all maps
   ├─ Compute Monte-Carlo returns
   └─ Supervise critic by MSE on returns

3. RL Fine-tuning  (main loop of train.py)
   ├─ Per generation (256 environment steps):
   │   ├─ Rollout: collect (obs, action, reward, done)
   │   ├─ Compute GAE advantages
   │   └─ Update actor + critic (10 PPO epochs, minibatch 128)
   │       Actor loss:  L_PPO + L_diff_AWR + λ_disp · L_dispersive
   │       Critic loss: MSE value regression
   └─ Every 60 generations: advance curriculum map
```

---

## Reward Shaping

| Term | Weight | Description |
|------|--------|-------------|
| Progress reward | ×48 | Advance along the raceline waypoints |
| Lap completion  | ×80 | Bonus for finishing a full lap |
| Checkpoint bonus | ×8 | 10 evenly-spaced waypoint checkpoints |
| Speed reward    | ×3  | Reward proportional to forward velocity |
| Collision penalty | −4 | Penalty for hitting walls or other agents |

---

## Project Structure

```
TR-D2PPO/
├── train.py                  # Main RL training loop
├── D2PPO_agent.py            # D²PPO agent (diffusion policy + PPO)
├── ppo_agent.py              # Standard PPO agent (Gaussian, for comparison)
├── weight_initializer.py     # Behavioral cloning pretraining
├── val.py                    # Validation stub
├── requirements.txt          # Python dependencies
│
├── models/
│   ├── DiffusionLSTM.py      # Main policy: DDPM denoiser + LSTM backbone
│   ├── DiffusionMamba2.py    # Alternative: DDPM denoiser + Mamba2 backbone
│   ├── CriticNetworks.py     # Value network (LSTM / Mamba2 encoder)
│   ├── HybridLSTM.py         # Classic LSTM + Gaussian policy (baseline)
│   ├── Mamba2Racer.py        # Mamba2 policy variant
│   ├── RLMamba2Racer.py      # Mamba2 RL policy head
│   └── AuxModels.py          # VisionEncoder, ResidualBlocks, SinusoidalPosEmb
│
├── utils/
│   ├── utils.py              # Reward calculation, waypoint loading, state normalisation
│   ├── diffusion_utils.py    # DDPM/DDIM schedules and sampling utilities
│   └── control_handler.py   # Human / gamepad control and demo collection
│
├── baselines/
│   ├── gap_follow_pure_pursuit.py  # Hybrid baseline (obstacle avoidance + raceline)
│   ├── gap_follow.py               # Reactive gap-following controller
│   ├── pure_pursuit.py             # Classic raceline-tracking controller
│   ├── sim_pure_pursuit.py         # Simulation variant of pure pursuit
│   └── mpc_agent.py                # Model predictive controller
│
├── analysis/
│   ├── race.py                     # Race evaluation and metric collection
│   ├── paper_data_collection.py    # Experiment orchestration for paper results
│   ├── post_process.py             # Log / results post-processing
│   └── pp_compare.py              # Pure pursuit baseline comparisons
│
├── maps/                     # 22 racing tracks (YAML configs + occupancy maps)
└── plots/                    # Training diagnostic plots
```

---

## Maps and Curriculum

Training uses 22 F1-style tracks split into three difficulty tiers:

| Tier | Maps |
|------|------|
| **Easy** | Hockenheim, Monza, Melbourne, BrandsHatch |
| **Medium** | Oschersleben, Sakhir, Sepang, SaoPaulo, Budapest, Catalunya, Silverstone |
| **Hard** | Zandvoort, MoscowRaceway, Austin, Nuerburgring, Spa, YasMarina, Sochi, … |

The agent trains on each map for `GEN_PER_MAP = 60` generations before switching to the next track in the pool.

---

## Baselines

| Baseline | Description |
|----------|-------------|
| **GapFollowPurePursuit** | Hybrid: follows raceline waypoints, switches to gap-following on obstacle detection |
| **PurePursuit** | Classic geometric controller; tracks pre-computed waypoints with curvature-based speed |
| **GapFollow** | Reactive: steers towards the largest gap in the LiDAR scan |
| **MPC** | Model predictive controller optimising over a receding horizon |

Baseline drivers also participate in multi-agent training as non-learning competitors.

---

## Configuration

Key hyperparameters are set at the top of `train.py` and `D2PPO_agent.py`:

```python
# train.py
NUM_AGENTS_AI        = 8        # Number of learning agents
NUM_AGENTS_PP        = 4        # Number of pure-pursuit baseline agents
TOTAL_TIMESTEPS      = 12_000_000
STEPS_PER_GENERATION = 256
GEN_PER_MAP          = 60       # Curriculum: generations per track

# D2PPO_agent.py
lr_actor             = 3e-5
lr_critic            = 1e-4
gamma                = 0.999
gae_lambda           = 0.95
clip_epsilon         = 0.2
num_diffusion_steps  = 10       # Denoising steps during training
awr_temperature      = 1.0      # AWR advantage temperature
dispersive_lambda    = 0.1      # Weight of dispersive (contrastive) loss
minibatch_size       = 128
epochs               = 10
```

---

## Requirements

- Python ≥ 3.9
- NVIDIA GPU with CUDA 12.8
- Key packages (see `requirements.txt` for pinned versions):
  - `torch==2.10.0+cu128`
  - `gym==0.19.0`
  - `mamba-ssm==2.3.0`
  - `torchrl==0.10.0`
  - `numpy`, `scipy`, `matplotlib`, `PyYAML`, `einops`
