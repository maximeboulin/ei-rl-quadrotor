"""Horizontal flight of the quadrotor along x, as a Gymnasium environment (Track B, advanced task in 2D: x + z).

Same organisation as drone_env.py (TD 4): parameters at the top, the physics in `dynamics`, the reward in
`compute_reward`, then the environment, then `run_episode` and `indicators`.

The height stays controlled by the Track B policy (drone_env.py, rl_controller_platform.m). This second policy only
decides the DIFFERENCE between motors 3 and 1:  u1 = u_z - delta,  u3 = u_z + delta  (u2 = u4 = u_z).
It replaces both loops of the cascade (position -> angle -> delta): it receives the gap to the target and the angle,
and sends delta.

Model of the pitch axis, identified in notebooks/x_axis.ipynb from explore_x.csv (UAV_model.slx):
    motors:  x_m' = (GAIN u - x_m) / TAU            (first order, the same as the Wiener model of the height)
    thrust:  F = tanh(C x_m)                          (static curve of the motors)
    pitch:   d(wy)/dt = K (F3 - F1),  d(theta)/dt = wy
    x:       d(vx)/dt = (az + g) tan(theta)           (the thrust is tilted by theta)

Target = the platform of the vehicle. In training, random scenarios: the vehicle starts at a random time at a random speed
(possibly 0), or the target jumps (land elsewhere), or the vehicle drives a curve (seen along one axis: x = R sin(s/R) or
y = R (1 - cos(s/R))), plus noise and model errors.

The same policy also flies the y axis (3D): same dynamics with motors 2-4 and the roll phi, see platform3d_rl_controller.m.

    from drone_env_x import DroneEnvX
    env = DroneEnvX()               # evaluation: vehicle at 0.3 m/s from 1.5 s (car_position.m), no noise
    env = DroneEnvX(train=True)     # training: random scenarios, noise, model changed
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

Ts = 0.05               # sampling period (s)
N_STEPS = 200           # one episode = 10 s, like the mission of the height
G = 9.81
SUB = 10                # integration sub-steps per Ts (the pitch is fast)

# Identified model (notebooks/x_axis.ipynb)
TAU = -Ts / np.log(0.7787)          # time constant of the motors, ~0.2 s
GAIN = 0.2231 / (1 - 0.7787)        # static gain of the motors, ~1.0
C = 0.992                           # F = tanh(C x_m)
K = 100.0                           # d(wy)/dt = K (F3 - F1)
U_HOVER = 0.389                     # common command u_z in hover

# Action and observation scales
D_MAX = 0.1             # delta = D_MAX * action, action in [-1, 1] (same bound as the cascade)
THETA_SCALE = 0.2       # theta is divided by 0.2 rad in the observation (~1)

# Weights of the smoothing terms of the reward (the height policy uses 0.2 and 2.0)
W_JUMP, W_ZIGZAG = 0.2, 2.0

# Evaluation scenario: the same as car_position.m
V_CAR, T_START = 0.3, 1.5

# Training conditions
NOISE_STD = 0.0045      # measurement noise at g_noise = 0.1 (as in drone_env.py)
THETA_EST_STD = 0.005   # error of the complementary filter on theta (measured < 0.01 rad in the emulator)
MODEL_SPREAD = 0.3      # K and the thrust changed by up to +/- 30 % (robustness, as the grid search of the cascade)


def car(t, t0, v, jump, R=0.0, axis=0):
    """Position, speed and acceleration of the target at time t, along the axis flown by the policy.
    At rest at 0, then from t0: straight line at speed v (R = 0) and/or shift by jump, or a curve of radius R driven at
    speed v, seen along the axis of the vehicle (axis 0: R sin(s/R)) or across it (axis 1: R (1 - cos(s/R)))."""
    if t < t0 - 1e-6:
        return 0.0, 0.0, 0.0
    if R == 0.0:
        return jump + v * (t - t0), v, 0.0
    a = v * (t - t0) / R
    if axis == 0:
        return R * np.sin(a), v * np.cos(a), -v * v / R * np.sin(a)
    return R * (1 - np.cos(a)), v * np.sin(a), v * v / R * np.cos(a)


def dynamics(state, delta, uz, k_gain, thrust):
    """Next state after Ts. state = (x, vx, theta, wy, xm1, xm3). The 2 motors of the pitch axis get uz -/+ delta."""
    x, vx, th, wy, x1, x3 = state
    u1, u3 = np.clip(uz - delta, -1, 1), np.clip(uz + delta, -1, 1)
    dt = Ts / SUB
    for _ in range(SUB):
        x1 += dt * (GAIN * u1 - x1) / TAU
        x3 += dt * (GAIN * u3 - x3) / TAU
        wy += dt * k_gain * (np.tanh(C * x3) - np.tanh(C * x1))
        th += dt * wy
        vx += dt * thrust * G * np.tan(th)      # az + g ~ g in hover (thrust: model error)
        x += dt * vx
    return (x, vx, th, wy, x1, x3)


def compute_reward(ex, ev, theta, a, a_prev, a_prev2, lost):
    """Reward of one step. ex, ev: gap of position and speed to the target; a: action (delta / D_MAX)."""
    if lost:
        return -100.0
    return (- abs(ex)                                  # be above the target (guides the agent from far away)
            + 0.5 * np.exp(-(ex / 0.05) ** 2)          # precision bonus: up to +0.5 per step when |ex| < ~2 cm
            - 0.1 * ev ** 2                            # same speed as the target (no overshoot)
            - W_JUMP * (a - a_prev) ** 2               # large jumps of the motor difference  (same terms as the height)
            - W_ZIGZAG * (a - 2 * a_prev + a_prev2) ** 2   # zigzags of the motor difference
            - 2.0 * max(abs(theta) - 0.2, 0))          # do not tilt more than ~11 degrees


class DroneEnvX(gym.Env):
    def __init__(self, train=False, seed=None):
        super().__init__()
        self.train = train
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)    # delta / D_MAX
        self.rng = np.random.default_rng(seed)

    def target(self, t):
        return car(t, self.t0, self.v, self.jump, self.R, self.axis)

    def measurements(self):
        """What the controller has in MATLAB (xz_rl_controller.m): x, vx (noisy), theta estimated, wy (noisy)."""
        n, r = self.noise, self.rng.standard_normal
        x, vx, th, wy = self.state[:4]
        return {'t': self.k * Ts, 'x': x + n * r(), 'vx': vx + n * r(),
                'theta': th + self.theta_bias + self.theta_err * r(), 'wy': wy + n * r()}

    def observation(self, m):
        """Same computation as in xz_rl_controller.m, in the same order."""
        x_c, v_c, a_c = self.target(m['t'])
        return np.array([x_c - m['x'],                  # gap of position to the target (m)
                         v_c - m['vx'],                 # gap of speed (m/s)
                         m['theta'] / THETA_SCALE,      # pitch angle, scaled to ~1
                         m['wy'],                       # pitch rate (rad/s)
                         self.a_prev,                   # previous action (delta / D_MAX)
                         a_c / G / THETA_SCALE],        # acceleration of the target, as the angle it needs (a/g), scaled
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rnd = self.rng
        if self.train:
            kind = rnd.integers(4)                          # 0: vehicle that starts, 1: target that jumps, 2: both, 3: curve
            self.t0 = rnd.uniform(0.0, 4.0)
            self.v = rnd.uniform(-0.6, 0.6) if kind != 1 else 0.0
            self.jump = rnd.uniform(-1.2, 1.2) if kind in (1, 2) else 0.0
            self.R = rnd.uniform(0.8, 3.0) if kind == 3 else 0.0
            self.axis = int(rnd.integers(2))
            self.noise = NOISE_STD * rnd.uniform(0, 2)
            self.theta_err, self.theta_bias = THETA_EST_STD * rnd.uniform(0, 2), rnd.uniform(-0.01, 0.01)
            self.k_gain = K * (1 + MODEL_SPREAD * rnd.uniform(-1, 1))
            self.thrust = 1 + 0.1 * rnd.uniform(-1, 1)
            self.uz0 = rnd.uniform(0.2, 0.6)                # common command: changes the slope of tanh (the "gain" of delta)
            x0 = rnd.uniform(-0.1, 0.1)
        else:
            o = options or {}                               # other scenario, e.g. {'v': 0.0, 'jump': 1.0}: land 1 m further,
            self.t0, self.v, self.jump = o.get('t0', T_START), o.get('v', V_CAR), o.get('jump', 0.0)   # {'R': 1.5, 'axis': 1}: curve
            self.R, self.axis = o.get('R', 0.0), o.get('axis', 0)
            self.noise, self.theta_err, self.theta_bias = 0.0, 0.0, 0.0
            self.k_gain, self.thrust, self.uz0, x0 = K, 1.0, U_HOVER, 0.0
        xm = GAIN * self.uz0                                # motors already at the common command
        self.state = (x0, 0.0, 0.0, 0.0, xm, xm)
        self.k, self.a_prev, self.a_prev2 = 0, 0.0, 0.0
        self.m = self.measurements()
        return self.observation(self.m), {'m': self.m}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        uz = self.uz0 + (0.02 * self.rng.standard_normal() if self.train else 0.0)   # u_z moves a little (height policy)
        self.state = dynamics(self.state, D_MAX * a, uz, self.k_gain, self.thrust)
        self.k += 1
        x, vx, th = self.state[:3]
        x_c, v_c, _ = self.target(self.k * Ts)
        lost = bool(abs(x_c - x) > 3.0 or abs(th) > 0.8)  # too far, or the drone flips: end of the episode
        reward = compute_reward(x_c - x, v_c - vx, th, a, self.a_prev, self.a_prev2, lost)
        self.a_prev2, self.a_prev = self.a_prev, a
        self.m = self.measurements()
        return self.observation(self.m), float(reward), lost, bool(self.k >= N_STEPS and not lost), {'m': self.m}


# -- Evaluation ----------------------------------------------------------------

def run_episode(env, policy, seed=0, options=None):
    """Flies one scenario. policy(obs) -> action. Returns the trajectory (t, x, x_target, theta, delta) and the return."""
    obs, info = env.reset(seed=seed, options=options)
    log, total = [], 0.0
    while True:
        a = float(np.asarray(policy(obs)).reshape(-1)[0])
        t = info['m']['t']
        log.append((t, env.state[0], env.target(t)[0], env.state[2], D_MAX * a))
        obs, r, terminated, truncated, info = env.step(np.array([a]))
        total += r
        if terminated or truncated:
            return np.array(log), total


def cascade_policy(obs):
    """The cascade of platform3d_controller.m written on the observation (reference to compare the RL policy with)."""
    ex, ev, th, wy, a_ff = obs[0], obs[1], obs[2] * THETA_SCALE, obs[3], obs[5] * THETA_SCALE
    th_ref = np.clip(a_ff + 0.1 * ex + 0.15 * ev, -0.2, 0.2)       # with the feedforward of the acceleration, a / g
    return [np.clip(0.1 * (th_ref - th) - 0.04 * wy, -D_MAX, D_MAX) / D_MAX]


def indicators(L):
    t, x, xc, th, d = L.T
    gap = x - xc
    return {'gap at 5.5 s (m)': gap[np.searchsorted(t, 5.5)],
            'max gap after 4 s (m)': np.abs(gap[t >= 4]).max(),
            'mean |gap| last 2 s (m)': np.abs(gap[t >= 8]).mean(),
            'max |theta| (rad)': np.abs(th).max(),
            'mean |d delta| (zigzag)': np.abs(np.diff(d)).mean()}
