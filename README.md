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
- [Current Training Profile](#current-training-profile)
- [Training Pipeline](#training-pipeline)
- [Reward Shaping](#reward-shaping)
- [Paper Comparison Tools](#paper-comparison-tools)
- [Project Structure](#project-structure)
- [Maps and Curriculum](#maps-and-curriculum)
- [Baselines](#baselines)
- [Configuration](#configuration)
- [Requirements](#requirements)

---

## Overview

TR-D2PPO trains reinforcement learning agents to race on F1-style tracks. Instead of the typical Gaussian policy used in PPO, it uses a **DDPM (Denoising Diffusion Probabilistic Model)** as the action distribution, which enables richer, multimodal exploration and smoother driving behavior.

The current training profile runs 3 D²PPO agents alongside 8 Pure Pursuit baseline drivers. Real-map generations use heterogeneous PPO minibatches across several maps, while the curriculum still alternates between real circuits and procedurally generated tracks. Dedicated held-out tracks are kept out of the curriculum for deterministic generalist checkpoint selection.

---

## Key Innovations

| Feature | Description |
|---------|-------------|
| **Diffusion Policies** | Replaces Gaussian actor with DDPM for expressive multimodal action distributions |
| **Dispersive Loss** | InfoNCE-style contrastive penalty on denoiser features to prevent mode collapse |
| **Advantage-Weighted Regression (AWR)** | Weights diffusion MSE loss by exponential of the PPO advantage |
| **Mamba2 Temporal Backbone** | Recurrent selective-SSM state carries temporal context without a full sliding feature buffer |
| **Multi-Agent Curriculum** | Alternates real F1 circuits with procedurally generated tracks every `GEN_PER_MAP` generations |
| **Multi-Map PPO Generations** | Rotates across multiple real maps inside one rollout generation for heterogeneous minibatches |
| **Composite Checkpoint Selection** | Saves best models by distance-per-collision adjusted for pace and progress, avoiding slow-collapse policies |
| **Held-Out Generalist Eval** | Periodically evaluates deterministic policy behavior on maps excluded from training |
| **Critic Pretraining** | Separate supervised value-network pretraining on Monte-Carlo returns |
| **DDIM Chain-based PPO** | Stochastic DDIM chain (η=0.5) stored during rollout and replayed for PPO importance ratios |
| **TBTT Learning** | Truncated BPTT through the Mamba2 encoder (chunk = `tbtt_length`) |

---

## Architecture

```
Input per step (50 Hz):
  ├─ LIDAR scan (1 080 beams)
  └─ Vehicle state [vx, vy, yaw-rate]
           │
           ▼
  VisionEncoder (1-D CNN on LIDAR scan)
           │
           ▼
  Feature projection + state concat
           │
           ▼
  Mamba2 recurrent SSM state
           │  obs_features
           ▼
  ConditionalDenoisingMLP  (noise prediction ε_θ)
           │
           ▼
  DDIM reverse process (10 rollout steps, 5 deploy steps)
           │
           ▼
  Action: [steering ∈ [-0.34, 0.34] rad,
           speed   ∈ [0, 20] m/s]
```

The critic uses the same LiDAR/state observation interface and a Mamba2 temporal encoder, then predicts a scalar value estimate for each observation.

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

> **Hardware note**: Training requires an NVIDIA GPU with CUDA 12.8 support. `torch.compile()` is currently disabled for the Mamba2 path because the custom CUDA kernels have been unstable with Inductor during recurrent `step()` inference.

---

## Usage

### 1. Behavioral Cloning Pretraining

Initializes the diffusion policy by imitating an expert racing policy.

The current pretraining path can also use the previous BC-LSTM racing policy from `../racing_rl/actor_val_best.pt` as the expert, collecting demonstrations from batched subprocess environments while the expert forward pass stays on the GPU.

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

- Runs 3 D2PPO agents + 8 Pure Pursuit baselines concurrently
- Starts from BC-pretrained actor weights and a pretrained critic by default
- Trains on real curriculum maps plus validated procedural tracks
- Uses `MAPS_PER_GEN = 3` for real-map generations, rotating through several maps inside one PPO rollout
- Saves per-map best snapshots to `models/actor/actor_best.pt` and `models/critic/critic_best.pt` using a composite score based on distance-per-collision, pace, and progress
- Runs deterministic held-out evaluation on Spa, YasMarina, and Spielberg every `TR_HELDOUT_EVAL_EVERY` generations and saves `actor_generalist.pt` / `critic_generalist.pt` when the mean held-out composite score improves
- Full resumable checkpoints are written to `models/checkpoint.pt`
- Training diagnostics and plots are written to `plots/`

### 3. Evaluation

```bash
python analysis/race.py
```

Loads a trained checkpoint and evaluates it on held-out tracks, collecting lap times, collision counts, and average speed.

For quick actor sanity checks on a single training map:

```bash
python val_catalunya.py --actor models/actor/actor_best.pt --laps 3 --ddim-steps 5
```

For paper-style model comparison, use `analysis/paper_data_collection.py` or `analysis/three_way_compare.py`.

---

## Current Training Profile

The active profile is tuned to preserve the BC prior while allowing D²PPO to improve racing behavior through RL fine-tuning.

| Area | Current setting |
|------|-----------------|
| Learning agents | 3 D²PPO agents |
| Background agents | 8 Pure Pursuit drivers |
| Actor backbone | Mamba2 diffusion actor (`DiffusionMamba2`) |
| Critic backbone | Mamba2 critic (`Mamba2CriticNetwork`) |
| Observation state | `[linear_vel_x, linear_vel_y, yaw_rate]` + 1080-beam LiDAR |
| Training DDIM | 10 stochastic DDIM steps with `eta = 0.5` |
| Deploy/eval DDIM | 5 deterministic DDIM steps by default |
| TBTT | Fixed 512-step chunks |
| Steps per generation | `int(raceline_length) * 5`, set per map |
| Curriculum cadence | `GEN_PER_MAP = 16` |
| Real-map mixing | `MAPS_PER_GEN = 3` |
| Critic warmup | 4 critic-only generations after map switches |
| Optimizer reset | Enabled on map switches by default (`TR_RESET_OPT_ON_MAP_SWITCH=1`) |
| Held-out maps | Spa, YasMarina, Spielberg |

Key environment overrides are read from `TR_*` variables: `TR_TOTAL_TIMESTEPS`, `TR_START_MAP`, `TR_GEN_PER_MAP`, `TR_MAX_GENERATIONS`, `TR_SKIP_RESUME`, `TR_CHECKPOINT_PATH`, `TR_DISABLE_HELDOUT_EVAL`, `TR_HELDOUT_EVAL_EVERY`, `TR_CRITIC_WARMUP_GENS`, `TR_MAPS_PER_GEN`, `TR_MIN_SELECTION_SPEED`, and `TR_MIN_SELECTION_PROGRESS_PER_STEP`.

The current checkpoint-selection score is:

```text
selection_score = distance_per_collision * speed_ratio * soft_gate
```

where `speed_ratio = avg_speed / raceline_mean_speed`, and the soft gate down-weights policies below `TR_MIN_SELECTION_SPEED` or `TR_MIN_SELECTION_PROGRESS_PER_STEP`. This prevents slow, low-collision policies from replacing faster racing checkpoints.

Current early-run diagnostics show the intended behavior: recent generations are holding about 6.2 m/s average speed, progress-per-step around 0.08, controlled KL, and no actor early stops. `actor_best.pt` has been updated by the composite selector; `actor_generalist.pt` updates only when deterministic held-out evaluation improves.

---

## Training Pipeline

```
1. Behavioral Cloning Pretraining  (weight_initializer.py)
   ├─ Collect expert demos from BC-LSTM or controller baselines on diverse maps
   ├─ Supervise DiffusionLSTM/Mamba2 with:
   │     L = L_diffusion + λ · L_dispersive
   └─ Save pretrained actor weights

2. Critic Pretraining  (first phase of train.py)
   ├─ Freeze pretrained actor, run rollouts on all maps
   ├─ Compute Monte-Carlo returns
   └─ Supervise critic by MSE on returns

3. RL Fine-tuning  (main loop of train.py)
   ├─ Per generation (5 × raceline_length steps):
   │   ├─ Rollout: collect (obs, action, reward, done, DDIM chain)
   │   ├─ Compute GAE (γ=0.999, λ=0.95) with EMA reward normalization
   │   └─ Update actor + critic (4 PPO epochs, minibatch 128)
   │       Actor loss:  L_PPO_DDIM + λ_diff · L_diff_MSE + λ_disp · L_dispersive
   │       Critic loss: Huber on normalized value targets with PPO clipping
   ├─ Save per-map best models by composite pace/progress-aware score
   ├─ Evaluate deterministic generalist checkpoints on held-out maps
   └─ Every 16 generations: switch focus map (real ↔ generated)
```

---

## Reward Shaping

| Term | Value | Description |
|------|-------|-------------|
| Progress reward | `500 / raceline_length` per m | Forward arc-length progress, auto-normalized per-map so one lap ≈ 500 |
| Checkpoint bonus | +1.0 | 10 evenly-spaced segments along the raceline |
| Lap completion  | +10.0 | One-shot bonus on each new lap |
| Wall collision | −5.0 | One-shot penalty on new wall-contact event |
| Agent collision | −1.0 | One-shot penalty on new agent-contact event |
| Steering-rate penalty | −0.5 × `abs(steer_t - steer_t-1)` | Discourages high-frequency steering oscillation |
| Steering magnitude penalty | 0.0 | Disabled so the agent can still turn hard enough for tight corners |
| Speed bonus | 0.008 × gated speed above 3 m/s | Rewards straight-line pace up to 10 m/s, gated down during steering |

All rewards are passed through a running EMA (α=0.01) inside `_compute_gae` so the advantage scale stays stable across map transitions.

Per-generation diagnostics include map metadata and reward components such as `progress_per_step`, `checkpoint_per_step`, `wall_col_per_step`, `agent_col_per_step`, `steer_rate_per_step`, and `speed_bonus_per_step`. These are written to `plots/d2ppo_training_diagnostics.csv`.

---

## Paper Comparison Tools

The analysis scripts support comparisons against the previous supervised BC-LSTM baseline from the sibling `racing_rl` repository.

| Script | Purpose |
|--------|---------|
| `analysis/bc_lstm_agent.py` | Inference wrapper around `../racing_rl/actor_val_best.pt` using the original `ExampleNetwork` / `VisionEncoder` definitions |
| `analysis/paper_data_collection.py` | Collects labeled race trajectories for selected agents, currently configured for the current D²PPO checkpoint |
| `analysis/three_way_compare.py` | Compares BC-LSTM, BC-pretrained D²PPO, and RL-fine-tuned D²PPO from frozen snapshot weights |
| `analysis/post_process.py` | Normalizes legacy and current labels for downstream plotting |

Diffusion actors are put into deploy mode during comparison with 5 DDIM steps and no action repeat, keeping evaluation deterministic and close to deployment behavior.

---

## Project Structure

```
TR-D2PPO/
├── train.py                  # Main RL training loop
├── D2PPO_agent.py            # D²PPO agent (diffusion policy + PPO)
├── ppo_agent.py              # Standard PPO agent (Gaussian, for comparison)
├── weight_initializer.py     # Behavioral cloning pretraining
├── val.py                    # D2PPO vs GFPP vs MPC validation on generated tracks
├── val_catalunya.py          # Quick deterministic actor sanity check on Catalunya
├── requirements.txt          # Python dependencies
│
├── models/
│   ├── DiffusionLSTM.py      # Legacy DDPM denoiser + LSTM backbone
│   ├── DiffusionMamba2.py    # Current actor: DDPM denoiser + Mamba2 backbone
│   ├── CriticNetworks.py     # Value networks, including Mamba2 critic
│   ├── HybridLSTM.py         # Hybrid training LSTM + Gaussian policy (baseline)
│   ├── Mamba2Racer.py        # Mamba2 variant, no diffusion
│   ├── RLMamba2Racer.py      # Mamba2 RL policy head, no diffusion
│   └── AuxModels.py          # VisionEncoder, ResidualBlocks, SinusoidalPosEmb
│
├── utils/
│   ├── utils.py              # Reward calculation, waypoint loading, state normalization
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
│   ├── bc_lstm_agent.py            # racing_rl BC-LSTM wrapper for comparison
│   ├── three_way_compare.py        # BC-LSTM vs BC-D2PPO vs RL-D2PPO comparison
│   ├── post_process.py             # Log / results post-processing
│   └── pp_compare.py              # Pure pursuit baseline comparisons
│
├── maps/                     # 22 racing tracks (YAML configs + occupancy maps)
└── plots/                    # Training diagnostic plots
```

---

## Maps and Curriculum

Training uses F1-style tracks split into three curriculum tiers. The held-out maps are intentionally excluded from training and used only for generalist checkpoint evaluation.

| Tier | Maps |
|------|------|
| **Easy** | Hockenheim, Monza, Melbourne, BrandsHatch |
| **Medium** | Sakhir, SaoPaulo, Budapest, Silverstone |
| **Hard** | Zandvoort, MoscowRaceway, Sochi |
| **Held-out** | Spa, YasMarina, Spielberg |

The agent trains on each focus map for `GEN_PER_MAP = 16` generations. Real-map generations can rotate through `MAPS_PER_GEN = 3` maps while preserving a single focus map for best-checkpoint scoring. Procedural maps are validated with a short Pure Pursuit smoke test before training so bad spawns and disconnected tracks are rejected early.

---

## Baselines

| Baseline | Description |
|----------|-------------|
| **GapFollowPurePursuit** | Hybrid: follows raceline waypoints, switches to gap-following on obstacle detection |
| **PurePursuit** | Classic geometric controller; tracks pre-computed waypoints with curvature-based speed |
| **GapFollow** | Reactive: steers towards the largest gap in the LiDAR scan |
| **MPC** | Model predictive controller optimizing over a receding horizon |

Pure Pursuit drivers participate in the current multi-agent training profile as non-learning competitors. MPC, GapFollow, and GapFollowPurePursuit remain available for validation and comparison scripts.

---

## Configuration

Key hyperparameters are set at the top of `train.py` and `D2PPO_agent.py`:

```python
# train.py
NUM_AGENTS_AI        = 3        # Number of learning agents
NUM_AGENTS_PP        = 8        # Number of pure-pursuit baseline agents
TOTAL_TIMESTEPS      = 12_000_000
STEPS_PER_GENERATION = 5 * raceline_length   # set per-map
GEN_PER_MAP          = 16       # Curriculum: generations per focus map
MAPS_PER_GEN         = 3        # Real-map rollout mixing
CRITIC_WARMUP_GENS   = 4        # Critic-only gens after focus-map switches
HELDOUT_EVAL_EVERY   = 12       # Generations between generalist evals

# D2PPO_agent.py
lr_actor             = 1e-5
lr_critic            = 5e-5
gamma                = 0.999
gae_lambda           = 0.95
clip_epsilon         = 0.2
num_diffusion_steps  = 100      # Full diffusion schedule
ddim_rl_steps        = 10       # Stochastic DDIM rollout chain
deploy_ddim_steps    = 5        # Default deterministic eval/deploy setting
awr_temperature      = 0.5      # AWR advantage temperature
dispersive_lambda    = 0.1      # Weight of dispersive (contrastive) loss
minibatch_size       = 128
epochs               = 4
kl_target            = 0.5
kl_early_stop        = 1.0
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

---

## License

This project is licensed under a modified **[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)**.

**In short:**

- ✅ **Free** for personal use, academic research, education, and non-profit work.
- ❌ **Commercial use is not permitted** without a separate commercial license.

If you or your organization wish to use this software for commercial purposes (including integration into commercial products or services), please contact **andsarve15@gmail.com** to discuss licensing terms.

See the full [LICENSE](./LICENSE) file for details.
