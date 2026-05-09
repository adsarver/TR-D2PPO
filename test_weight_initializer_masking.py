"""Minimal masked trajectory-batch trainer for weight initializer data.

Hard-coded equivalent of:
    python weight_initializer.py --load demos/best_case_demos.pt --model_type mamba2 --skip_critic

This trains only the actor. It pads variable-length trajectories, masks padded
slots out of the loss, and saves actor weights for rollout testing.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DiffusionMamba2 import DiffusionMamba2
from utils.diffusion_utils import dispersive_loss_infonce_l2, extract
from utils.sim_config import D2PPO_STATE_DIM, LIDAR_BEAMS


DEMO_PATH = "demos/best_case_demos.pt"
MODEL_TYPE = "mamba2"
SKIP_CRITIC = True
SAVE_DIR = "models/actor/pretrained/masked_test"
BEST_SAVE_PATH = os.path.join(SAVE_DIR, "actor_masked_best.pt")
FINAL_SAVE_PATH = os.path.join(SAVE_DIR, "actor_masked_final.pt")
HISTORY_CSV_PATH = os.path.join(SAVE_DIR, "loss_history.csv")
TOTAL_LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "loss_total.png")
COMBINED_LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "loss_total_with_ddim_validation.png")
DIFF_LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "loss_diffusion.png")
DISP_LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "loss_dispersive.png")
X0_LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "loss_x0.png")
VAL_OBJECTIVE_PLOT_PATH = os.path.join(SAVE_DIR, "validation_objective_losses.png")
DDIM1_MSE_PLOT_PATH = os.path.join(SAVE_DIR, "validation_ddim1_action_mse.png")
DDIM5_MSE_PLOT_PATH = os.path.join(SAVE_DIR, "validation_ddim5_action_mse.png")

SEED = 7
TRAIN_FRACTION = 0.75
EPOCHS = 300
TBTT_LENGTH = 512
LR = 3e-4
DISPERSIVE_LAMBDA = 0.5
DISPERSIVE_TEMPERATURE = 0.5
DEPLOYMENT_X0_LAMBDA = 0.1
DEPLOYMENT_X0_SAMPLER_STEPS = (1, 5)
NUM_DIFFUSION_STEPS = 100
PLOT_EVERY = 1
VALIDATION_MAP = "Catalunya"
VALIDATION_SOURCE_AGENT = "BC_LSTM"
VALIDATION_STEPS = 1024
VALIDATION_DDIM1_STEPS = 1
VALIDATION_DDIM5_STEPS = 5
DEPLOYMENT_DDIM_STEPS = 5
BEST_VALIDATION_METRIC = "ddim5_action_mse"


def write_loss_history(history: list[dict[str, float]]) -> None:
    fieldnames = [
        "epoch",
        "chunks",
        "valid_transitions",
        "padded_slots",
        "diff_loss",
        "disp_loss",
        "x0_loss",
        "total_loss",
        "lr",
        "val_diff_loss",
        "val_disp_loss",
        "val_x0_loss",
        "val_total_loss",
        "ddim1_action_mse",
        "ddim1_steer_mse",
        "ddim1_speed_mse",
        "ddim5_action_mse",
        "ddim5_steer_mse",
        "ddim5_speed_mse",
        "best_val_loss",
    ]
    with open(HISTORY_CSV_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_loss_history(history: list[dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        ("total_loss", "Main Total Loss", TOTAL_LOSS_PLOT_PATH, "#4dd0e1"),
        ("diff_loss", "Diffusion Loss", DIFF_LOSS_PLOT_PATH, "#81c784"),
        ("disp_loss", "Dispersive Loss", DISP_LOSS_PLOT_PATH, "#ffb74d"),
        ("x0_loss", "Deployment X0 Reconstruction Loss", X0_LOSS_PLOT_PATH, "#64b5f6"),
        ("ddim1_action_mse", "DDIM-1 Validation Action MSE", DDIM1_MSE_PLOT_PATH, "#ef5350"),
        ("ddim5_action_mse", "DDIM-5 Validation Action MSE", DDIM5_MSE_PLOT_PATH, "#ba68c8"),
    ]
    epochs = [int(row["epoch"]) for row in history]

    plt.style.use("dark_background")
    for key, title, path, color in plot_specs:
        values = [float(row[key]) for row in history]
        if not any(np.isfinite(values)):
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, values, color=color, linewidth=2.5, marker=".", markersize=5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        tmp_path = f"{path}.tmp.png"
        fig.savefig(tmp_path, dpi=140)
        plt.close(fig)
        os.replace(tmp_path, path)

    fig, ax_train = plt.subplots(figsize=(11, 6))
    train_total = [float(row["total_loss"]) for row in history]
    ddim1_mse = [float(row["ddim1_action_mse"]) for row in history]
    ddim5_mse = [float(row["ddim5_action_mse"]) for row in history]

    ax_train.plot(
        epochs,
        train_total,
        color="#4dd0e1",
        linewidth=2.5,
        marker=".",
        markersize=5,
        label="train total loss",
    )
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("train total loss", color="#4dd0e1")
    ax_train.tick_params(axis="y", labelcolor="#4dd0e1")
    ax_train.grid(True, alpha=0.25)

    ax_val = ax_train.twinx()
    ax_val.plot(
        epochs,
        ddim1_mse,
        color="#ef5350",
        linewidth=2.2,
        marker=".",
        markersize=5,
        label="DDIM-1 action MSE",
    )
    ax_val.plot(
        epochs,
        ddim5_mse,
        color="#ba68c8",
        linewidth=2.2,
        marker=".",
        markersize=5,
        label="DDIM-5 action MSE",
    )
    ax_val.set_ylabel("validation action MSE", color="#eeeeee")
    ax_val.tick_params(axis="y", labelcolor="#eeeeee")

    lines = ax_train.get_lines() + ax_val.get_lines()
    labels = [line.get_label() for line in lines]
    ax_train.legend(lines, labels, loc="best")
    ax_train.set_title("Train Total Loss vs DDIM Deployment Validation")
    fig.tight_layout()

    tmp_path = f"{COMBINED_LOSS_PLOT_PATH}.tmp.png"
    fig.savefig(tmp_path, dpi=140)
    plt.close(fig)
    os.replace(tmp_path, COMBINED_LOSS_PLOT_PATH)

    fig, ax = plt.subplots(figsize=(11, 6))
    val_plot_specs = [
        ("val_total_loss", "validation total", "#f06292"),
        ("val_diff_loss", "validation diffusion", "#81c784"),
        ("val_disp_loss", "validation dispersive", "#ffb74d"),
        ("val_x0_loss", "validation deployment x0", "#64b5f6"),
    ]
    for key, label, color in val_plot_specs:
        values = [float(row[key]) for row in history]
        if not any(np.isfinite(values)):
            continue
        ax.plot(
            epochs,
            values,
            color=color,
            linewidth=2.4,
            marker=".",
            markersize=5,
            label=label,
        )
    ax.set_title("Validation Stage-1 Objective Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    tmp_path = f"{VAL_OBJECTIVE_PLOT_PATH}.tmp.png"
    fig.savefig(tmp_path, dpi=140)
    plt.close(fig)
    os.replace(tmp_path, VAL_OBJECTIVE_PLOT_PATH)


def build_mamba2_actor(device: torch.device) -> DiffusionMamba2:
    model = DiffusionMamba2(
        state_dim=D2PPO_STATE_DIM,
        action_dim=2,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        inference_steps=0,
        obs_feature_dim=256,
        time_emb_dim=32,
        hidden_dims=(256, 256),
        beta_schedule="cosine",
        d_model=256,
        d_state=128,
        d_conv=4,
        d_head=32,
        expand=2,
        odom_expand=64,
    ).to(device)
    model.denoise_net.register_dispersive_hooks("late")
    return model


def load_trajectories(path: str) -> list[tuple[tuple[str, object], list[dict]]]:
    demos = torch.load(path, weights_only=False)
    trajectories = defaultdict(list)
    for demo in demos:
        trajectories[(demo["map"], demo.get("agent", 0))].append(demo)

    loaded = []
    for key, trajectory in trajectories.items():
        trajectory.sort(key=lambda item: item.get("step", 0))
        if len(trajectory) >= 10:
            loaded.append((key, trajectory))
    loaded.sort(key=lambda item: len(item[1]))
    return loaded


def split_train_validation_trajectories(
    trajectories: list[tuple[tuple[str, object], list[dict]]],
    train_fraction: float,
    seed: int,
) -> tuple[
    list[tuple[tuple[str, object], list[dict]]],
    list[tuple[tuple[str, object], list[dict]]],
]:
    if len(trajectories) < 2:
        raise ValueError("at least two trajectories are required for a train-val split")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(trajectories))
    rng.shuffle(indices)
    train_count = int(round(len(trajectories) * train_fraction))
    train_count = min(max(train_count, 1), len(trajectories) - 1)
    train_indices = set(indices[:train_count].tolist())

    train_trajectories = [
        trajectory for idx, trajectory in enumerate(trajectories) if idx in train_indices
    ]
    validation_trajectories = [
        trajectory for idx, trajectory in enumerate(trajectories) if idx not in train_indices
    ]
    return train_trajectories, validation_trajectories


def select_validation_trajectory(
    trajectories: list[tuple[tuple[str, object], list[dict]]],
) -> tuple[tuple[str, object], list[dict]]:
    for key, trajectory in trajectories:
        map_name, agent_name = key
        if map_name == VALIDATION_MAP and VALIDATION_SOURCE_AGENT in str(agent_name):
            return key, trajectory
    for key, trajectory in trajectories:
        if key[0] == VALIDATION_MAP:
            return key, trajectory
    return trajectories[0]


def build_masked_chunk(
    trajectories: list[tuple[tuple[str, object], list[dict]]],
    chunk_start: int,
    chunk_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(trajectories)
    max_remaining = max(max(0, len(traj) - chunk_start) for _, traj in trajectories)
    time_steps = min(chunk_length, max_remaining)
    if time_steps <= 0:
        raise ValueError("chunk_start is past the end of every trajectory")

    scans = torch.zeros(time_steps, batch_size, 1, LIDAR_BEAMS, device=device)
    states = torch.zeros(time_steps, batch_size, D2PPO_STATE_DIM, device=device)
    actions = torch.zeros(time_steps, batch_size, 2, device=device)
    dones = torch.zeros(time_steps, batch_size, device=device)
    valid_mask = torch.zeros(time_steps, batch_size, dtype=torch.bool, device=device)

    for batch_idx, (_, trajectory) in enumerate(trajectories):
        valid_len = max(0, min(time_steps, len(trajectory) - chunk_start))
        if valid_len == 0:
            continue

        items = trajectory[chunk_start:chunk_start + valid_len]
        scans[:valid_len, batch_idx, 0] = torch.as_tensor(
            np.stack([item["scan"] for item in items]),
            dtype=torch.float32,
            device=device,
        )
        states[:valid_len, batch_idx] = torch.as_tensor(
            np.stack([item["state"] for item in items]),
            dtype=torch.float32,
            device=device,
        )
        actions[:valid_len, batch_idx] = torch.as_tensor(
            np.stack([item["action"] for item in items]),
            dtype=torch.float32,
            device=device,
        )
        dones[:valid_len, batch_idx] = torch.as_tensor(
            [item.get("done", 0) for item in items],
            dtype=torch.float32,
            device=device,
        )
        valid_mask[:valid_len, batch_idx] = True

    return scans, states, actions, dones, valid_mask


def deployment_x0_timesteps(model: DiffusionMamba2, device: torch.device) -> torch.Tensor:
    timesteps = sorted({
        timestep
        for sampler_steps in DEPLOYMENT_X0_SAMPLER_STEPS
        for timestep in model._ddim_timestep_schedule(sampler_steps)
    }, reverse=True)
    return torch.tensor(timesteps, dtype=torch.long, device=device)


def deployment_x0_reconstruction_loss(
    model: DiffusionMamba2,
    normalized_actions: torch.Tensor,
    obs_features: torch.Tensor,
) -> torch.Tensor:
    if normalized_actions.shape[0] == 0 or DEPLOYMENT_X0_LAMBDA <= 0.0:
        return torch.tensor(0.0, device=normalized_actions.device)

    candidate_timesteps = deployment_x0_timesteps(model, normalized_actions.device)
    timestep_indices = torch.randint(
        0,
        candidate_timesteps.numel(),
        (normalized_actions.shape[0],),
        device=normalized_actions.device,
    )
    diffusion_steps = candidate_timesteps[timestep_indices]
    noise = torch.randn_like(normalized_actions)
    noisy_actions = model.q_sample(normalized_actions, diffusion_steps, noise)
    predicted_noise = model.predict_noise(noisy_actions, obs_features, diffusion_steps)

    alpha_t = extract(model.alphas_cumprod, diffusion_steps, noisy_actions.shape)
    x0_pred = (
        noisy_actions - torch.sqrt(1.0 - alpha_t) * predicted_noise
    ) / torch.sqrt(alpha_t).clamp(min=1e-8)
    model.denoise_net.get_intermediate_features()
    return F.mse_loss(x0_pred.tanh(), normalized_actions)


def masked_losses(
    model: DiffusionMamba2,
    actions: torch.Tensor,
    obs_features: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_actions = actions[valid]
    active_features = obs_features[valid]
    if active_actions.shape[0] == 0:
        zero = torch.tensor(0.0, device=actions.device)
        return zero, zero, zero

    normalized_actions = model.normalize_action(active_actions)
    noise = torch.randn_like(normalized_actions)
    diffusion_steps = torch.randint(
        0,
        model.num_diffusion_steps,
        (active_actions.shape[0],),
        device=actions.device,
        dtype=torch.long,
    )
    noisy_actions = model.q_sample(normalized_actions, diffusion_steps, noise)
    predicted_noise = model.predict_noise(noisy_actions, active_features, diffusion_steps)
    diff_loss = F.mse_loss(predicted_noise, noise)

    feature_list = model.denoise_net.get_intermediate_features()
    disp_loss = torch.tensor(0.0, device=actions.device)
    if feature_list:
        for features in feature_list:
            if features.ndim > 2:
                features = features.mean(dim=list(range(1, features.ndim - 1)))
            disp_loss = disp_loss + dispersive_loss_infonce_l2(
                features,
                DISPERSIVE_TEMPERATURE,
            )
        disp_loss = disp_loss / len(feature_list)

    x0_loss = deployment_x0_reconstruction_loss(
        model,
        normalized_actions,
        active_features,
    )

    return diff_loss, disp_loss, x0_loss


def run_masked_chunk(
    model: DiffusionMamba2,
    optimizer: torch.optim.Optimizer,
    trajectories: list[tuple[tuple[str, object], list[dict]]],
    chunk_start: int,
    device: torch.device,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    scans, states, actions, dones, valid_mask = build_masked_chunk(
        trajectories,
        chunk_start,
        TBTT_LENGTH,
        device,
    )
    time_steps, batch_size = valid_mask.shape
    conv_state = conv_state.detach()
    ssm_state = ssm_state.detach()

    optimizer.zero_grad(set_to_none=True)
    total_diff = torch.tensor(0.0, device=device)
    total_disp = torch.tensor(0.0, device=device)
    total_x0 = torch.tensor(0.0, device=device)
    valid_transitions = 0

    for t_idx in range(time_steps):
        valid = valid_mask[t_idx]
        if t_idx > 0 and dones[t_idx - 1].any():
            reset_idx = dones[t_idx - 1].nonzero(as_tuple=False).squeeze(-1)
            conv_state[reset_idx] = 0.0
            ssm_state[reset_idx] = 0.0

        inactive = (~valid).nonzero(as_tuple=False).squeeze(-1)
        if inactive.numel() > 0:
            conv_state[inactive] = 0.0
            ssm_state[inactive] = 0.0

        obs_features, conv_state, ssm_state = model.encode_observation(
            scans[t_idx],
            states[t_idx],
            conv_state,
            ssm_state,
        )
        if not valid.any():
            continue

        active_count = int(valid.sum().item())
        diff_loss, disp_loss, x0_loss = masked_losses(
            model,
            actions[t_idx],
            obs_features,
            valid,
        )
        total_diff = total_diff + diff_loss * active_count
        total_disp = total_disp + disp_loss * active_count
        total_x0 = total_x0 + x0_loss * active_count
        valid_transitions += active_count

    if valid_transitions == 0:
        raise RuntimeError("masked chunk had no valid transitions")

    loss = (
        total_diff
        + DISPERSIVE_LAMBDA * total_disp
        + DEPLOYMENT_X0_LAMBDA * total_x0
    ) / valid_transitions
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    padded_slots = valid_mask.numel() - int(valid_mask.sum().item())
    return {
        "chunk_start": float(chunk_start),
        "batch_size": float(batch_size),
        "time_steps": float(time_steps),
        "valid_transitions": float(valid_transitions),
        "padded_slots": float(padded_slots),
        "diff_loss": float((total_diff / valid_transitions).detach().cpu()),
        "disp_loss": float((total_disp / valid_transitions).detach().cpu()),
        "x0_loss": float((total_x0 / valid_transitions).detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
    }, conv_state.detach(), ssm_state.detach()


def train_epoch(
    model: DiffusionMamba2,
    optimizer: torch.optim.Optimizer,
    trajectories: list[tuple[tuple[str, object], list[dict]]],
    max_len: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    conv_state, ssm_state = model.allocate_state(len(trajectories), device)
    total_valid = 0
    total_padded = 0
    weighted_diff = 0.0
    weighted_disp = 0.0
    weighted_x0 = 0.0
    weighted_total = 0.0
    chunk_count = 0

    for chunk_start in range(0, max_len, TBTT_LENGTH):
        stats, conv_state, ssm_state = run_masked_chunk(
            model,
            optimizer,
            trajectories,
            chunk_start,
            device,
            conv_state,
            ssm_state,
        )
        valid = int(stats["valid_transitions"])
        total_valid += valid
        total_padded += int(stats["padded_slots"])
        weighted_diff += stats["diff_loss"] * valid
        weighted_disp += stats["disp_loss"] * valid
        weighted_x0 += stats["x0_loss"] * valid
        weighted_total += stats["total_loss"] * valid
        chunk_count += 1

    return {
        "chunks": float(chunk_count),
        "valid_transitions": float(total_valid),
        "padded_slots": float(total_padded),
        "diff_loss": weighted_diff / max(total_valid, 1),
        "disp_loss": weighted_disp / max(total_valid, 1),
        "x0_loss": weighted_x0 / max(total_valid, 1),
        "total_loss": weighted_total / max(total_valid, 1),
    }


def evaluate_ddim_action_mse(
    model: DiffusionMamba2,
    trajectories: list[tuple[tuple[str, object], list[dict]]],
    device: torch.device,
    ddim_steps: int,
) -> dict[str, float]:
    model.eval()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = None
    if device.type == "cuda":
        cuda_rng_state = torch.cuda.get_rng_state(device)

    total_steps = 0
    weighted_steer_mse = 0.0
    weighted_speed_mse = 0.0
    try:
        torch.manual_seed(SEED + 1000 + int(ddim_steps))
        if device.type == "cuda":
            torch.cuda.manual_seed(SEED + 1000 + int(ddim_steps))

        with torch.no_grad():
            for _, trajectory in trajectories:
                eval_len = min(VALIDATION_STEPS, len(trajectory))
                if eval_len <= 0:
                    continue

                conv_state, ssm_state = model.allocate_state(1, device)
                predictions = []
                experts = []
                for item in trajectory[:eval_len]:
                    scan = torch.as_tensor(
                        item["scan"], dtype=torch.float32, device=device
                    ).view(1, 1, LIDAR_BEAMS)
                    state = torch.as_tensor(
                        item["state"], dtype=torch.float32, device=device
                    ).view(1, D2PPO_STATE_DIM)
                    obs_features, conv_state, ssm_state = model.encode_observation(
                        scan,
                        state,
                        conv_state,
                        ssm_state,
                    )
                    action = model.ddim_sample_action(
                        obs_features,
                        num_steps=ddim_steps,
                        eta=0.0,
                    )
                    model.denoise_net.get_intermediate_features()
                    predictions.append(action)
                    experts.append(torch.as_tensor(
                        item["action"], dtype=torch.float32, device=device
                    ).view(1, 2))

                predicted = torch.cat(predictions, dim=0)
                expert = torch.cat(experts, dim=0)
                mse_by_dim = F.mse_loss(predicted, expert, reduction="none").mean(dim=0)
                weighted_steer_mse += float(mse_by_dim[0].detach().cpu()) * eval_len
                weighted_speed_mse += float(mse_by_dim[1].detach().cpu()) * eval_len
                total_steps += eval_len
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)

    if total_steps == 0:
        return {
            "action_mse": float("nan"),
            "steer_mse": float("nan"),
            "speed_mse": float("nan"),
        }

    steer_mse = weighted_steer_mse / total_steps
    speed_mse = weighted_speed_mse / total_steps
    return {
        "action_mse": (steer_mse + speed_mse) / 2.0,
        "steer_mse": steer_mse,
        "speed_mse": speed_mse,
    }


def evaluate_stage1_objective(
    model: DiffusionMamba2,
    trajectories: list[tuple[tuple[str, object], list[dict]]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    max_eval_len = min(
        VALIDATION_STEPS,
        max(len(trajectory) for _, trajectory in trajectories),
    )
    if max_eval_len <= 0:
        return {
            "val_diff_loss": float("nan"),
            "val_disp_loss": float("nan"),
            "val_x0_loss": float("nan"),
            "val_total_loss": float("nan"),
        }

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = None
    if device.type == "cuda":
        cuda_rng_state = torch.cuda.get_rng_state(device)

    total_valid = 0
    weighted_diff = 0.0
    weighted_disp = 0.0
    weighted_x0 = 0.0
    try:
        torch.manual_seed(SEED + 2000)
        if device.type == "cuda":
            torch.cuda.manual_seed(SEED + 2000)

        conv_state, ssm_state = model.allocate_state(len(trajectories), device)
        with torch.no_grad():
            for chunk_start in range(0, max_eval_len, TBTT_LENGTH):
                chunk_len = min(TBTT_LENGTH, max_eval_len - chunk_start)
                scans, states, actions, dones, valid_mask = build_masked_chunk(
                    trajectories,
                    chunk_start,
                    chunk_len,
                    device,
                )
                conv_state = conv_state.detach()
                ssm_state = ssm_state.detach()

                for t_idx in range(valid_mask.shape[0]):
                    valid = valid_mask[t_idx]
                    if t_idx > 0 and dones[t_idx - 1].any():
                        reset_idx = dones[t_idx - 1].nonzero(as_tuple=False).squeeze(-1)
                        conv_state[reset_idx] = 0.0
                        ssm_state[reset_idx] = 0.0

                    inactive = (~valid).nonzero(as_tuple=False).squeeze(-1)
                    if inactive.numel() > 0:
                        conv_state[inactive] = 0.0
                        ssm_state[inactive] = 0.0

                    obs_features, conv_state, ssm_state = model.encode_observation(
                        scans[t_idx],
                        states[t_idx],
                        conv_state,
                        ssm_state,
                    )
                    if not valid.any():
                        continue

                    active_count = int(valid.sum().item())
                    diff_loss, disp_loss, x0_loss = masked_losses(
                        model,
                        actions[t_idx],
                        obs_features,
                        valid,
                    )
                    weighted_diff += float(diff_loss.detach().cpu()) * active_count
                    weighted_disp += float(disp_loss.detach().cpu()) * active_count
                    weighted_x0 += float(x0_loss.detach().cpu()) * active_count
                    total_valid += active_count
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)

    val_diff = weighted_diff / max(total_valid, 1)
    val_disp = weighted_disp / max(total_valid, 1)
    val_x0 = weighted_x0 / max(total_valid, 1)
    val_total = val_diff + DISPERSIVE_LAMBDA * val_disp + DEPLOYMENT_X0_LAMBDA * val_x0
    return {
        "val_diff_loss": val_diff,
        "val_disp_loss": val_disp,
        "val_x0_loss": val_x0,
        "val_total_loss": val_total,
    }


def main() -> None:
    if MODEL_TYPE != "mamba2" or not SKIP_CRITIC:
        raise RuntimeError("This smoke test is hard-coded for mamba2 actor-only masking.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajectories = load_trajectories(DEMO_PATH)
    train_trajectories, validation_trajectories = split_train_validation_trajectories(
        trajectories,
        TRAIN_FRACTION,
        SEED,
    )
    validation_key, validation_trajectory = select_validation_trajectory(validation_trajectories)
    train_lengths = [len(trajectory) for _, trajectory in train_trajectories]
    validation_lengths = [len(trajectory) for _, trajectory in validation_trajectories]
    train_max_len = max(train_lengths)

    print(f"[Mask Test] demos={DEMO_PATH}")
    print(f"[Mask Test] model_type={MODEL_TYPE}, skip_critic={SKIP_CRITIC}, device={device}")
    print(
        f"[Mask Test] trajectories={len(trajectories)}, "
        f"train={len(train_trajectories)}, val={len(validation_trajectories)}, "
        f"split={TRAIN_FRACTION:.0%}/{1.0 - TRAIN_FRACTION:.0%}, batch=train_all"
    )
    print(
        f"[Mask Test] train_min_len={min(train_lengths)}, "
        f"train_max_len={max(train_lengths)}, "
        f"val_min_len={min(validation_lengths)}, "
        f"val_max_len={max(validation_lengths)}"
    )
    print(
        f"[Mask Test] epochs={EPOCHS}, tbtt={TBTT_LENGTH}, lr={LR}, "
        f"save={BEST_SAVE_PATH}"
    )
    print(f"[Mask Test] loss_csv={HISTORY_CSV_PATH}")
    print(f"[Mask Test] total_loss_plot={TOTAL_LOSS_PLOT_PATH}")
    print(f"[Mask Test] combined_loss_plot={COMBINED_LOSS_PLOT_PATH}")
    print(f"[Mask Test] diff_loss_plot={DIFF_LOSS_PLOT_PATH}")
    print(f"[Mask Test] disp_loss_plot={DISP_LOSS_PLOT_PATH}")
    print(f"[Mask Test] x0_loss_plot={X0_LOSS_PLOT_PATH}")
    print(f"[Mask Test] val_objective_plot={VAL_OBJECTIVE_PLOT_PATH}")
    print(f"[Mask Test] ddim1_mse_plot={DDIM1_MSE_PLOT_PATH}")
    print(f"[Mask Test] ddim5_mse_plot={DDIM5_MSE_PLOT_PATH}")
    print(
        f"[Mask Test] deployment_x0_lambda={DEPLOYMENT_X0_LAMBDA}, "
        f"deployment_x0_sampler_steps={DEPLOYMENT_X0_SAMPLER_STEPS}"
    )
    print(
        f"[Mask Test] validation={validation_key}, "
        f"steps={min(VALIDATION_STEPS, len(validation_trajectory))}, "
        f"val_trajectories={len(validation_trajectories)}, "
        f"deployment_ddim={DEPLOYMENT_DDIM_STEPS}"
    )
    print(f"[Mask Test] best_checkpoint_metric={BEST_VALIDATION_METRIC}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    model = build_mamba2_actor(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.01,
    )

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        stats = train_epoch(model, optimizer, train_trajectories, train_max_len, device)
        val_objective = evaluate_stage1_objective(model, validation_trajectories, device)
        ddim1_stats = evaluate_ddim_action_mse(
            model,
            validation_trajectories,
            device,
            VALIDATION_DDIM1_STEPS,
        )
        ddim5_stats = evaluate_ddim_action_mse(
            model,
            validation_trajectories,
            device,
            VALIDATION_DDIM5_STEPS,
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        marker = ""
        current_val_loss = {
            "ddim1_action_mse": ddim1_stats["action_mse"],
            "ddim5_action_mse": ddim5_stats["action_mse"],
            "val_total_loss": val_objective["val_total_loss"],
        }[BEST_VALIDATION_METRIC]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), BEST_SAVE_PATH)
            marker = " *"

        epoch_row = {
            "epoch": float(epoch),
            "chunks": stats["chunks"],
            "valid_transitions": stats["valid_transitions"],
            "padded_slots": stats["padded_slots"],
            "diff_loss": stats["diff_loss"],
            "disp_loss": stats["disp_loss"],
            "x0_loss": stats["x0_loss"],
            "total_loss": stats["total_loss"],
            "lr": lr,
            "val_diff_loss": val_objective["val_diff_loss"],
            "val_disp_loss": val_objective["val_disp_loss"],
            "val_x0_loss": val_objective["val_x0_loss"],
            "val_total_loss": val_objective["val_total_loss"],
            "ddim1_action_mse": ddim1_stats["action_mse"],
            "ddim1_steer_mse": ddim1_stats["steer_mse"],
            "ddim1_speed_mse": ddim1_stats["speed_mse"],
            "ddim5_action_mse": ddim5_stats["action_mse"],
            "ddim5_steer_mse": ddim5_stats["steer_mse"],
            "ddim5_speed_mse": ddim5_stats["speed_mse"],
            "best_val_loss": best_val_loss,
        }
        history.append(epoch_row)
        write_loss_history(history)
        if epoch % PLOT_EVERY == 0 or epoch == EPOCHS:
            plot_loss_history(history)

        if epoch <= 5 or epoch % 10 == 0 or marker:
            print(
                f"[Mask Test] epoch={epoch:03d}/{EPOCHS} "
                f"chunks={int(stats['chunks'])} "
                f"valid={int(stats['valid_transitions'])} "
                f"padded={int(stats['padded_slots'])} "
                f"diff={stats['diff_loss']:.5f} "
                f"disp={stats['disp_loss']:.5f} "
                f"x0={stats['x0_loss']:.5f} "
                f"total={stats['total_loss']:.5f} "
                f"val_x0={val_objective['val_x0_loss']:.5f} "
                f"val_total={val_objective['val_total_loss']:.5f} "
                f"ddim1={ddim1_stats['action_mse']:.5f} "
                f"ddim5={ddim5_stats['action_mse']:.5f} "
                f"lr={lr:.2e}{marker}"
            )

    torch.save(model.state_dict(), FINAL_SAVE_PATH)

    sample_key, sample_trajectory = validation_key, validation_trajectory
    model.eval()
    with torch.no_grad():
        conv_state, ssm_state = model.allocate_state(1, device)
        warmup = min(len(sample_trajectory) - 8, 200)
        for item in sample_trajectory[:warmup]:
            scan = torch.as_tensor(item["scan"], dtype=torch.float32, device=device).view(1, 1, LIDAR_BEAMS)
            state = torch.as_tensor(item["state"], dtype=torch.float32, device=device).view(1, D2PPO_STATE_DIM)
            obs_features, conv_state, ssm_state = model.encode_observation(
                scan,
                state,
                conv_state,
                ssm_state,
            )

        sampled_actions = []
        expert_actions = []
        for item in sample_trajectory[warmup:warmup + 8]:
            scan = torch.as_tensor(item["scan"], dtype=torch.float32, device=device).view(1, 1, LIDAR_BEAMS)
            state = torch.as_tensor(item["state"], dtype=torch.float32, device=device).view(1, D2PPO_STATE_DIM)
            obs_features, conv_state, ssm_state = model.encode_observation(
                scan,
                state,
                conv_state,
                ssm_state,
            )
            sampled_actions.append(model.sample_action(obs_features, deterministic=True))
            expert_actions.append(torch.as_tensor(item["action"], dtype=torch.float32, device=device).view(1, 2))

        sampled = torch.cat(sampled_actions, dim=0)
        expert = torch.cat(expert_actions, dim=0)
        action_mse = F.mse_loss(sampled, expert).item()
        print(
            f"[Mask Test] validation_key={sample_key} action_mse={action_mse:.5f}"
        )

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[Mask Test] peak_cuda_memory={peak_mb:.1f} MiB")
    print(f"[Mask Test] best_val_loss={best_val_loss:.5f} ({BEST_VALIDATION_METRIC})")
    print(f"[Mask Test] saved_best={BEST_SAVE_PATH}")
    print(f"[Mask Test] saved_final={FINAL_SAVE_PATH}")
    print(f"[Mask Test] loss_csv={HISTORY_CSV_PATH}")
    print(f"[Mask Test] total_loss_plot={TOTAL_LOSS_PLOT_PATH}")
    print(f"[Mask Test] combined_loss_plot={COMBINED_LOSS_PLOT_PATH}")
    print(f"[Mask Test] diff_loss_plot={DIFF_LOSS_PLOT_PATH}")
    print(f"[Mask Test] disp_loss_plot={DISP_LOSS_PLOT_PATH}")
    print(f"[Mask Test] x0_loss_plot={X0_LOSS_PLOT_PATH}")
    print(f"[Mask Test] val_objective_plot={VAL_OBJECTIVE_PLOT_PATH}")
    print(f"[Mask Test] ddim1_mse_plot={DDIM1_MSE_PLOT_PATH}")
    print(f"[Mask Test] ddim5_mse_plot={DDIM5_MSE_PLOT_PATH}")
    print(f"[Mask Test] OK: masked {EPOCHS}-epoch Mamba2 actor training completed.")


if __name__ == "__main__":
    main()