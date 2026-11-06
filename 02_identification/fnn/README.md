# FNN identification of the quadrotor vertical dynamics (step 3)

Retained model (1-step delay: the command has no instantaneous effect):

    az[k+1] = f(az[k], u[k])      <=>      az[k] = f(az[k-1], u[k-1]) = f(az_prev, u_prev)

f is a feedforward network with 2 inputs (2 -> 64 -> 64 -> 1, tanh). The notebook justifies this
input choice physically and compares it with 3 other input sets on the validation set.

## Files

| File | Role |
|---|---|
| `identification_fnn.ipynb` | the whole study, as a report: data, choice of inputs, training, free-run validation, conclusion |
| `fnn.py` | model functions only (no classes): `build_fnn`, `make_regressors`, `train_fnn`, `predict`, `rollout`, `save_fnn`, `load_fnn` |
| `fnn_config.yaml` | all settings (seed, Ts, network, training, evaluation) |
| `model/fnn.pt` | trained model: weights + normalization statistics + Ts + config + history + test split |
| `model/input_comparison.csv` | result of the input comparison (section 3.1) |
| `figures/` | `learning_curve.png`, `overlay_rollout.png` (for the slides) |

## Run

Python 3.10–3.12 (the repo `.venv` works). Put `bdd_in_mat_05.csv` and `bdd_out_mat_05.csv` in
`data/raw/` at the repo root (not versioned, ~105 MB each), open the notebook in VS Code with the
`.venv` kernel and run all cells (~10 min on CPU). Top cell: `RETRAIN` / `RUN_COMPARISON` set to
`False` reload the saved results instead of training again.

## Use in env.py

```python
import sys; sys.path.append("02_identification/fnn")   # from the repo root
from fnn import load_fnn, predict, rollout
model = load_fnn()                         # dict: net, norm, n_az, Ts, meta
state = [az_k, u_km1]                      # [az[k], u[k-1]], physical units
next_state = predict(model, state, u_k)    # next_state = [az[k+1], u[k]]
```

Link with `rl_controller.m`: at sample k it receives `az_prev = az[k-1]` and `u_prev = u[k-1]`
(UnitDelay blocks in `EINT_RL_closed_loop.slx`), i.e. the inputs of f: `f(az_prev, u_prev) = az[k]`.

## Working hypotheses (not verified)

- **H1**: u = thrust command in [-1, 1]; az = vertical acceleration in m/s², z up.
- **H2**: the CSVs are sampled at Ts = 0.05 s (one row = one 100 s trajectory).
- **H3**: no ground contact in the data: ground and platform must be handled in env.py.
- **H4**: CSV column k = sample at t = k·Ts, u[k] held on [t_k, t_k+1).
