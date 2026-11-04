"""Gymnasium environment: the drone simulated by the identified LSTM."""
import json
import numpy as np
import torch
from torch import nn
import gymnasium as gym
from gymnasium import spaces

TS = 0.05          # sampling period [s]
N_STEPS = 200      # 10 s mission
U_MIN, U_MAX = 0.0, 0.8   # command range, centred near hover (u ≈ 0.385)
I_MAX = 1.0        # bound on the error integral (anti-windup)


def mission(t):
    """Reference altitude at time t (must be identical in rl_controller.m)."""
    if t < 1.0:
        return 0.0
    elif t < 5.0:
        return 0.5
    else:
        return -0.25


class LSTMModel(nn.Module):
    """Same layers as model_nn in the notebook, without Lightning."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)


class DroneEnv(gym.Env):

    def __init__(self, model_path="models/lstm_az.ckpt",
                 scaler_path="models/lstm_az_scaler.json", randomize=True):
        super().__init__()
        # 1. Load the identified model
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = LSTMModel()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        scaler = json.load(open(scaler_path))
        self.a_min, self.a_max = scaler["a_min"], scaler["a_max"]
        self.randomize = randomize          # True for training, False for the export

        # 2. What the agent can do, and what it sees
        self.action_space = spaces.Box(U_MIN, U_MAX, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

    # ---- helpers ----
    def _lstm_step(self, u):
        """Advance the LSTM by one step with command u, return a_z in m/s²."""
        x = torch.tensor([[[u]]], dtype=torch.float32)
        with torch.no_grad():
            out, self.hc = self.model.rnn(x, self.hc)
            y_mm = self.model.head(out).item()
        return (y_mm + 1) / 2 * (self.a_max - self.a_min) + self.a_min   # back to m/s²

    def _measurements(self):
        return {"t": self.t, "z": self.z, "vz": self.vz, "az_prev": self.az_prev,
                "z_ref": mission(self.t), "u_prev": self.u_prev}

    def _observation(self, m):
        """Called exactly once per reset()/step(), like observation() in rl_controller.m."""
        self.I = float(np.clip(self.I + TS * (m["z_ref"] - m["z"]), -I_MAX, I_MAX))
        return np.array([m["z_ref"] - m["z"], m["vz"], m["az_prev"] / 10,
                         m["u_prev"], m["z_ref"], self.I], dtype=np.float32)

    # ---- Gymnasium API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.k, self.t = 0, 0.0
        self.z, self.vz = 0.0, 0.0
        self.u_prev, self.az_prev = 0.0, 0.0     # Delay_u and Delay_az start at 0 in Simulink
        self.I = 0.0                             # error integral
        # Model error for this flight: constant bias on a_z (domain randomization)
        self.bias = self.np_random.uniform(-1.0, 1.0) if self.randomize else 0.0
        # Warm-up: 40 steps at u = 0, like the start of every experiment in the data
        self.hc = None
        for _ in range(40):
            self._lstm_step(0.0)
        m = self._measurements()
        return self._observation(m), {"m": m}

    def step(self, action):
        u = float(np.clip(action[0], U_MIN, U_MAX))

        # 1. The model predicts the acceleration (+ model error and noise during training)
        az = self._lstm_step(u) + self.bias
        if self.randomize:
            az += 0.1 * self.np_random.normal()

        # 2. Integrate: acceleration -> velocity -> altitude (no ground: UAV_model.slx has none)
        self.vz += TS * az
        self.z += TS * self.vz

        # 3. Reward: 1 on the reference, decreasing quickly away from it
        z_ref = mission(self.t + TS)
        e = z_ref - self.z
        reward = np.exp(-(e / 0.15)**2) - 0.05 * min(self.vz**2, 4) - 0.2 * (u - self.u_prev)**2
        terminated = abs(e) > 2          # drone lost: end the episode

        # 4. Advance time
        self.u_prev, self.az_prev = u, az
        self.k += 1
        self.t = self.k * TS
        truncated = self.k >= N_STEPS
        m = self._measurements()
        return self._observation(m), float(reward), terminated, truncated, {"m": m}
