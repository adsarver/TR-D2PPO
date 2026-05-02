# -*- coding: utf-8 -*-
"""
BC-LSTM baseline wrapper for paper data collection.

Loads the supervised-pretrained ExampleNetwork (LSTM) checkpoint
``actor_val_best.pt`` from the racing_rl repo and exposes the same
runtime API as :class:`D2PPOAgent` so it can be dropped into the
shared ``race()`` loop.

Architecture / hyper-parameters mirror what was used to train the
checkpoint inside racing_rl's ``SupervisedAgent``:
    state_dim=4, lstm_hidden_size=512, lstm_num_layers=2,
    memory_length=350, memory_stride=20
"""

import os
import sys

import numpy as np
import torch

# The BC-LSTM checkpoint (``actor_val_best.pt``) was trained against the
# racing_rl repo's VisionEncoder + ExampleNetwork.  The local copies in
# ``models/`` have diverged (different CNN width / pooling), so we
# import the original definitions to guarantee weight compatibility.
RACING_RL_PATH = os.environ.get(
    "RACING_RL_PATH", "/home/WVU-AD/ads00024/racing_rl")
if os.path.isdir(RACING_RL_PATH) and RACING_RL_PATH not in sys.path:
    sys.path.insert(0, RACING_RL_PATH)

try:
    from model import ExampleNetwork, VisionEncoder  # type: ignore
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        f"Could not import racing_rl model from {RACING_RL_PATH}. "
        "Set the RACING_RL_PATH env var to the racing_rl repo root."
    ) from exc


# Default state ranges expected by ExampleNetwork (must match training)
NUM_BEAMS = 1080


class BCLSTMAgent:
    """Inference-only wrapper around racing_rl's BC-pretrained LSTM."""

    def __init__(
        self,
        num_agents,
        weights_path,
        state_dim=4,
        lstm_hidden_size=512,
        lstm_num_layers=2,
        memory_length=350,
        memory_stride=20,
        num_beams=NUM_BEAMS,
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        encoder = VisionEncoder(num_scan_beams=num_beams)
        self.actor_network = ExampleNetwork(
            state_dim=state_dim,
            action_dim=2,
            encoder=encoder,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            memory_length=memory_length,
            memory_stride=memory_stride,
        ).to(self.device)

        self._load_checkpoint(weights_path)
        self.actor_network.eval()

        # Rolling temporal buffers
        self.obs_buffer = self.actor_network.create_observation_buffer(
            num_agents, self.device)
        self.hidden_h, self.hidden_c = self.actor_network.get_init_hidden(
            num_agents, self.device, transpose=True)

    # ------------------------------------------------------------------
    # Checkpoint loading (handles racing_rl's "0.module." DataParallel
    # prefix and torch.compile's "_orig_mod." prefix gracefully)
    # ------------------------------------------------------------------
    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"Expected a state_dict at {path}, got {type(ckpt)}")

        cleaned = {}
        for k, v in ckpt.items():
            if not isinstance(v, torch.Tensor):
                continue
            ck = k
            for prefix in ("_orig_mod.", "0.module."):
                if ck.startswith(prefix):
                    ck = ck[len(prefix):]
            cleaned[ck] = v

        # Filter shape mismatches (e.g. state_lo/state_hi if state_dim differs)
        model_sd = self.actor_network.state_dict()
        filtered, mismatched = {}, []
        for k, v in cleaned.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            elif k in model_sd:
                mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

        result = self.actor_network.load_state_dict(filtered, strict=False)
        loaded = len(filtered)
        total = sum(1 for v in cleaned.values() if isinstance(v, torch.Tensor))
        print(f"[BC-LSTM] Loaded {loaded}/{total} tensors from {path}")
        if mismatched:
            print(f"  - {len(mismatched)} shape mismatches (skipped):")
            for k, c, m in mismatched[:5]:
                print(f"    - {k}: ckpt={c} model={m}")
        if result.missing_keys:
            print(f"  - {len(result.missing_keys)} missing in checkpoint (random init)")
        if result.unexpected_keys:
            print(f"  - {len(result.unexpected_keys)} unexpected keys (ignored)")

    # ------------------------------------------------------------------
    # Observation / action API (mirrors D2PPOAgent so race() can call
    # the same methods regardless of which model is plugged in)
    # ------------------------------------------------------------------
    def _obs_to_tensors(self, obs):
        scans = np.array(obs["scans"])[: self.num_agents]
        scan_t = torch.from_numpy(scans.astype(np.float32)).unsqueeze(1)

        components = [
            np.asarray(obs["linear_vels_x"])[: self.num_agents],
            np.asarray(obs["linear_vels_y"])[: self.num_agents],
            np.asarray(obs["ang_vels_z"])[: self.num_agents],
        ]
        if self.state_dim >= 4:
            accel = obs.get("linear_accel_x")
            if accel is None:
                # Some f1tenth gym builds don't expose linear_accel_x;
                # fall back to zeros to keep state_dim=4.
                accel = np.zeros(self.num_agents, dtype=np.float32)
            components.append(np.asarray(accel)[: self.num_agents])

        state_data = np.stack(components, axis=1).astype(np.float32)
        state_t = torch.from_numpy(state_data)
        return scan_t.to(self.device), state_t.to(self.device)

    @torch.no_grad()
    def get_action_and_value(self, scan_tensor, state_tensor,
                             deterministic=True, store=True):
        """Return ``(action, None, None)`` to match D2PPOAgent's signature."""
        loc, scale, obs_buf, hh, hc = self.actor_network(
            scan_tensor, state_tensor, self.obs_buffer,
            self.hidden_h, self.hidden_c,
        )
        if store:
            self.obs_buffer = obs_buf
            self.hidden_h = hh
            self.hidden_c = hc

        action = loc if deterministic else loc + scale * torch.randn_like(loc)
        return action, None, None

    def reset_buffers(self, agent_idxs=None):
        if agent_idxs is None:
            self.obs_buffer = self.actor_network.create_observation_buffer(
                self.num_agents, self.device)
            self.hidden_h, self.hidden_c = self.actor_network.get_init_hidden(
                self.num_agents, self.device, transpose=True)
        else:
            self.obs_buffer[agent_idxs] = 0.0
            self.hidden_h[agent_idxs] = 0.0
            self.hidden_c[agent_idxs] = 0.0

    # No per-map state — but expose for race() compatibility.
    def update_map(self, map_name):  # noqa: D401
        return
