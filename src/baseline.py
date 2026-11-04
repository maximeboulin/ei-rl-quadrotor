"""Linear baseline: a PID altitude controller using the same measurements as the RL policy."""
import numpy as np
from src.env import TS, U_MIN, U_MAX, I_MAX

U_HOVER = 0.385    # hover command, read on the static characteristic (a_z = 0)


class PID:
    """u = u_hover + Kp*e + Ki*∫e - Kd*vz   (derivative on the measured velocity)."""

    def __init__(self, Kp=0.4, Ki=0.05, Kd=0.3):
        # gains tuned by grid search on the LSTM environment, with a_z bias of -1, 0, +1 m/s²
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.reset()

    def reset(self):
        self.I = 0.0

    def __call__(self, m):
        e = m["z_ref"] - m["z"]
        self.I = float(np.clip(self.I + TS * e, -I_MAX, I_MAX))
        u = U_HOVER + self.Kp * e + self.Ki * self.I - self.Kd * m["vz"]
        return float(np.clip(u, U_MIN, U_MAX))


def fly(env, controller):
    """One 10 s flight; controller(obs, m) -> u. Returns t, z, z_ref, u as arrays."""
    obs, info = env.reset()
    log = {"t": [], "z": [], "z_ref": [], "u": []}
    for _ in range(200):
        u = controller(obs, info["m"])
        obs, r, term, trunc, info = env.step(np.array([u], dtype=np.float32))
        m = info["m"]
        log["t"].append(m["t"]); log["z"].append(m["z"]); log["z_ref"].append(m["z_ref"]); log["u"].append(u)
        if term or trunc:
            break
    return {k: np.array(v) for k, v in log.items()}
