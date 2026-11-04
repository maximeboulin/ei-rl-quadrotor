"""FNN model of the quadrotor vertical dynamics, as plain functions (no classes).

Retained model (1-step delay: the command u[k] has no effect on az[k], it acts on az[k+1]):

    az[k+1] = f(az[k], u[k])        <=>        az[k] = f(az[k-1], u[k-1]) = f(az_prev, u_prev)

f is a feedforward network (nn.Sequential): 2 inputs -> 64 -> 64 -> 1, tanh.
The "model" is a dict: {"net": nn.Sequential, "norm": {...}, "n_az": int, "Ts": float, "meta": {...}}

Use in env.py (physical units everywhere, normalization is applied inside):

    from identification_fnn.fnn import load_fnn, predict, rollout
    model = load_fnn()                              # identification_fnn/model/fnn.pt
    state = [az_k, u_km1]                           # [az[k], u[k-1]] = [az, u_prev]
    next_state = predict(model, state, u_k)         # [az[k+1], u[k]]
    az_seq = rollout(model, state0, u_seq)          # free run on its own predictions

u[k-1] does not enter f, but it is kept in the state because the controller observes it (u_prev)
and the reward penalizes the command variation u[k] - u[k-1].

train_fnn / f_phys also accept other input sets (first n_az columns = az values, the others = u
values): the notebook uses this only to compare candidate inputs. predict / rollout are for the
retained 2-input model.
"""

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEFAULT_PATH = Path(__file__).resolve().parent / "model" / "fnn.pt"
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


def build_fnn(cfg, n_inputs=2) -> nn.Sequential:
    """n_inputs -> hidden layers (cfg["hidden"], cfg["activation"]) -> 1 output (az at the next step)."""
    layers, n_in = [], n_inputs
    for h in cfg["hidden"]:
        layers += [nn.Linear(n_in, h), ACTIVATIONS[cfg["activation"]]()]
        n_in = h
    return nn.Sequential(*layers, nn.Linear(n_in, 1))


def make_regressors(u, az, valid):
    """Training pairs of the retained model from trajectories (N, T): X = [az[k], u[k]], y = az[k+1].
    Only pairs whose samples are all valid; pairs never cross two trajectories."""
    X = np.stack([az[:, :-1], u[:, :-1]], axis=-1)
    ok = valid[:, :-1] & valid[:, 1:]
    return X[ok], az[:, 1:][ok]


def state_at(az, u, k):
    """State at sample k from the measured histories of ONE trajectory: [az[k], u[k-1]]."""
    return np.array([az[k], u[k - 1]])


def _normalize(norm, X, n_az):
    mean = np.r_[np.full(n_az, norm["az_mean"]), np.full(X.shape[1] - n_az, norm["u_mean"])]
    std = np.r_[np.full(n_az, norm["az_std"]), np.full(X.shape[1] - n_az, norm["u_std"])]
    return (X - mean) / std


def f_phys(model, X):
    """Network in physical units: X (B, n_inputs) -> az at the next step (B,)."""
    n = model["norm"]
    x = _normalize(n, np.asarray(X, np.float64), model["n_az"])
    with torch.no_grad():
        y = model["net"](torch.from_numpy(np.ascontiguousarray(x))).numpy()[:, 0]
    return y * n["az_std"] + n["az_mean"]


def predict(model, state, u):
    """One step of the retained model. state (..., 2) = [az[k], u[k-1]], u (...,) = u[k]
    -> next state (..., 2) = [az[k+1], u[k]]."""
    state = np.asarray(state, np.float64)
    batch = state.shape[:-1]
    s = state.reshape(-1, 2)
    a = np.asarray(u, np.float64).reshape(-1)
    az_next = f_phys(model, np.column_stack([s[:, 0], a]))
    return np.column_stack([az_next, a]).reshape(*batch, 2)


def rollout(model, state0, u_seq):
    """Free run: state0 (B, 2) at sample k, u_seq (B, H) = u[k..k+H-1] -> az[k+1..k+H] (B, H)."""
    s, u_seq = np.asarray(state0, np.float64), np.asarray(u_seq, np.float64)
    out = np.empty(u_seq.shape)
    for h in range(u_seq.shape[1]):
        s = predict(model, s, u_seq[:, h])
        out[:, h] = s[:, 0]
    return out


def _rmse_phys(net, x, y, az_std, chunk=500_000):
    """RMSE in m/s^2 of a network on normalized tensors, evaluated in chunks (memory)."""
    se = 0.0
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            se += float(((net(x[i:i + chunk]) - y[i:i + chunk]) ** 2).sum())
    return float(np.sqrt(se / len(x))) * az_std


def train_fnn(cfg, X_tr, y_tr, X_va, y_va, Ts, n_epochs, seed=0, n_az=1, verbose=True):
    """Mini-batch Adam on the MSE of the normalized az, for exactly n_epochs epochs.
    After each epoch, the RMSE (m/s^2) on the whole training and validation sets is recorded;
    the weights of the epoch with the lowest validation RMSE are kept.
    Normalization statistics come from the training set only. Returns (model dict, history)."""
    torch.manual_seed(seed)
    norm = {"az_mean": float(X_tr[:, 0].mean()), "az_std": float(X_tr[:, 0].std()),
            "u_mean": float(X_tr[:, n_az].mean()), "u_std": float(X_tr[:, n_az].std())}

    def tensors(X, y):
        return (torch.tensor(_normalize(norm, X, n_az), dtype=torch.float32),
                torch.tensor((y - norm["az_mean"]) / norm["az_std"], dtype=torch.float32)[:, None])

    (xt, yt), (xv, yv) = tensors(X_tr, y_tr), tensors(X_va, y_va)
    net = build_fnn(cfg, X_tr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    best, best_val, best_epoch, hist, t0 = None, np.inf, 0, [], time.time()
    for epoch in range(1, n_epochs + 1):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss_fn(net(xt[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        tr, va = _rmse_phys(net, xt, yt, norm["az_std"]), _rmse_phys(net, xv, yv, norm["az_std"])
        hist.append({"epoch": epoch, "train_rmse": tr, "val_rmse": va})
        if va < best_val:
            best, best_val, best_epoch = copy.deepcopy(net.state_dict()), va, epoch
        if verbose:
            print(f"epoch {epoch:2d}  train RMSE {tr:.4f}  val RMSE {va:.4f} m/s^2"
                  f"{'  *' if best_epoch == epoch else ''}  ({time.time() - t0:.0f} s)", flush=True)
    net.load_state_dict(best)
    model = {"net": net.double().eval(), "norm": norm, "n_az": n_az, "Ts": float(Ts),
             "meta": {"history": hist, "best_epoch": best_epoch, "config": dict(cfg), "seed": seed}}
    return model, hist


def save_fnn(model, path=DEFAULT_PATH):
    """One file: weights + normalization + Ts + metadata (config, history, test split)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model["net"].state_dict().items()},
                "norm": model["norm"], "n_az": model["n_az"], "Ts": model["Ts"], "meta": model["meta"]}, path)


def load_fnn(path=DEFAULT_PATH):
    """Load the trained model dict, ready for predict() / rollout()."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    n_inputs = d["state_dict"]["0.weight"].shape[1]
    net = build_fnn(d["meta"]["config"], n_inputs).double()
    net.load_state_dict({k: v.double() for k, v in d["state_dict"].items()})
    return {"net": net.eval(), "norm": d["norm"], "n_az": d.get("n_az", 1), "Ts": d["Ts"], "meta": d["meta"]}
