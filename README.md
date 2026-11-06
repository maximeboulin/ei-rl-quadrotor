# EI Data-Driven Control – Track B: reinforcement learning on a quadrotor

CentraleSupélec, EI Data-Driven Control, November 2026 – **Group 1**.

A quadrotor must take off from a platform, climb to 0.5 m and land on the ground (−0.25 m). We only have
flight data (command `u` → vertical acceleration `a_z`). The method:

```
flight data ──► identified model ──► Gymnasium environment ──► SAC agent ──► MATLAB simulator (UAV_model.slx)
  01_data       02_identification          03 / 05                03 / 05         the "real" drone, for testing
```

The identified model replaces the drone during training. The trained policy is exported to MATLAB
(`policy.mat`) and flown on the provided Simulink drone, which plays the role of the real drone.

We then extended the work to the advanced tasks: rotation on the spot (yaw) and landing on a moving car
(2D with an LSTM model, 2D/3D with an RNN model).

## Repository layout

```
├── data/raw/                 provided CSV files + generated datasets (not versioned, see "Data")
├── matlab/                   provided files: UAV_model.slx, EINT_system_simulation.slx, EINT_RL_closed_loop.slx,
│                             quadcopter_model.p, random_init.m (added to the MATLAB path by every script)
├── src/                      shared Python code
│   ├── env.py                  Gymnasium environment, altitude, LSTM model (DroneEnv)
│   ├── env_2d.py               Gymnasium environment, landing on the car in 2D, LSTM model (LandingEnv)
│   ├── baseline.py             PID baseline, same law as in rl_controller.m
│   └── export_policy.py        SB3 policy -> policy.mat (provided, used by every experiment)
├── models/                   trained models: lstm_az, lstm_2d, rnn_az, final SAC agents (sac_*.zip)
│
├── 01_data/                  data analysis and cleaning
├── 02_identification/        one folder per model: arx/, fnn/, rnn/, lstm/
├── 03_altitude_rl/           take-off and landing on the ground
│   ├── lstm_sac/               SAC on the LSTM model
│   ├── rnn_sac/                SAC on the RNN model
│   └── rnn_robustness/         SAC on an RNN model, robustness to delays, losses, gusts, bias
├── 04_rotation/              advanced task: one full turn on the spot (yaw)
├── 05_landing_car/           advanced task: landing on a moving car
│   ├── lstm_2d/                2D (x, z): LSTM model + SAC (theta_cmd) + classical descent law
│   └── rnn_xyz/                x axis, 2D and 3D (car in a curve): cascade and SAC, RNN altitude policy
│
├── results/                  training outputs (TensorBoard, checkpoints), not versioned
└── slides/                   final presentation + script exporting the MATLAB figures
```

Each experiment folder is **self-contained**: notebooks, MATLAB scripts, exported policy (`policy*.mat`)
and figures sit side by side. The notebooks write `policy.mat` in their own folder, where the MATLAB
scripts of the same folder read it.

## Setup

**Python** 3.10–3.12:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebooks **from their own folder** (default in VS Code and Jupyter): all paths are relative to it.

**MATLAB** with Simulink. Every script moves to its own folder and adds `matlab/` to the path, so it can be
run from anywhere (`run('03_altitude_rl/lstm_sac/EINT_RL_eval.m')` or open it and press Run).

## Data

The provided files are too big for GitHub (2 × 105 MB). Copy them into `data/raw/`:

```
data/raw/bdd_in_mat_05.csv     3000 flights × 2000 samples, command u in [-1, 1]
data/raw/bdd_out_mat_05.csv    measured vertical acceleration a_z [m/s²]
```

Generated files also go to `data/` (not versioned): `data/processed/dataset_v1.npz` (01_data),
`data/raw/landing_data.mat` and `landing_split.npz` (05_landing_car/lstm_2d).

## Steps and how to run them

### 01 – Data (`01_data/`)

| File | Role |
|---|---|
| `data_analysis.ipynb` | inspection, cleaning (4 faulty flights removed: #122, #455, #1233, #1677), static map, noise analysis, split 70/15/15 by flight → `data/processed/dataset_v1.npz` |
| `replay_data.m` | replays CSV commands in Simulink: checks that the output is `a_z`, measures the noise |
| `check_Ts.m`, `plot_tr5.m` | step response of `a_z`: time constant τ ≈ 0.2 s, justification of Ts = 0.05 s |

Findings: first-order dynamics (τ ≈ 0.2 s) followed by a saturation (`tanh`), hover at u ≈ 0.385,
white Gaussian sensor noise σ ≈ 0.045 m/s².

### 02 – Identification (`02_identification/`)

| Folder | Model | Entry point |
|---|---|---|
| `arx/` | linear ARX, least squares | `python linear_model_arx.py`, `python arx_processing.py <na> <nb>` (e.g. `1 1`), `python parameters_arx_exploration.py` (choice of na, nb, nk, slow) |
| `fnn/` | feedforward network, a_z[k+1] = f(a_z[k], u[k]) | `identification_fnn.ipynb` (see its README) |
| `rnn/` | RNN (state-space network), 32 units | `rnn_identification.ipynb` → `models/rnn_az.pt`; `gen_long_data.m` for 10 min tests |
| `lstm/` | LSTM, 64 units | `lstm_identification.ipynb` → `models/lstm_az.ckpt` |

Free-run validation over 100 s on the test flights (RMSE on a_z): LSTM 0.051, FNN 0.064, RNN 0.137,
ARX 0.817 m/s². The linear model cannot reproduce the saturation; the networks reach the noise level.

### 03 – Take-off and landing on the ground (`03_altitude_rl/`)

Mission: z_ref = 0 → 0.5 m at t = 1 s → −0.25 m (ground) at t ≈ 5 s. Observation:
height error, v_z, previous a_z and u, reference (or preview 0.5 s ahead), integral of the error.

| Folder | Train (Python) | Test on the drone (MATLAB) |
|---|---|---|
| `lstm_sac/` | `env_test.ipynb`, `train_sac.ipynb` (uses `src/env.py`) | `EINT_RL_eval.m`, `EINT_PID_eval.m`, `EINT_compare.m` (SAC vs PID), `EINT_robustness.m` (noise levels) |
| `rnn_sac/` | `track_b.ipynb` (uses `drone_env.py`) | `EINT_RL_eval.m` |
| `rnn_robustness/` | `RNN.ipynb` (model), `drone_env.ipynb`, `main_and_figures.ipynb` | `EINT_RL_eval.m` |

`EINT_RL_eval.m` first replays the Python trajectory (`traj_model.mat`) through `rl_controller.m`: it must
print `Check OK` (MATLAB computes the same observation and command as Python), then flies the mission on
`UAV_model.slx`.

### 04 – Rotation (`04_rotation/`)

`motor_test.m` (effect of each motor on the angles), `rotation_demo.m` (take-off to 0.5 m, then one full
turn: altitude PID, yaw PD with +du on motors 1-3 and −du on motors 2-4, so the total thrust is unchanged).

### 05 – Landing on a moving car (`05_landing_car/`)

| Folder | Content | Run |
|---|---|---|
| `lstm_2d/` | data from a cascade with exogenous inputs, LSTM (U_common, θ_cmd) → (a_x, a_z, θ), SAC gives θ_cmd, classical descent law for z | MATLAB `generate_data.m` → notebooks `04`, `05`, `06` → MATLAB `eval_landing.m` → `python animate_landing.py`. Details in its README |
| `rnn_xyz/` | x axis identified from `explore_x.m`, cascade and SAC on x (and y), altitude by the RNN policy; car in a straight line (2D) or in a curve (3D) | `x_axis.ipynb`, `track_b_x.ipynb` → MATLAB `EINT_XZ_eval`, `EINT_PLATFORM_eval`, `EINT_PLATFORM_RL_eval`, `EINT_PLATFORM3D_eval`, `EINT_PLATFORM3D_RL_eval` → `compare_rl_cascade.ipynb`, `platform_animation*.ipynb` |

## Main results (on the MATLAB drone unless stated)

| Experiment | Result |
|---|---|
| Altitude, LSTM + SAC | tracking RMSE 0.17 m vs 0.22 m for a tuned PID (LSTM environment) |
| Altitude, RNN + SAC, robustness | SAC better than the PID in 9 / 9 defect scenarios (delays, command/measurement loss, gusts, ground effect, battery) |
| Rotation | full turn on the spot with the altitude kept at 0.5 m |
| Moving car, 2D, LSTM + SAC | 4 / 9 landings (PD baseline 9 / 9); failures 19–26 cm ahead at high car speed (model / drone mismatch) |
| Moving car, 3D, RNN + SAC | peak gap 12.5 cm in the curve vs 19.5 cm for the cascade |

Main limit: the agents exploit model errors outside the training data (bang-bang commands, unseen inputs).
Next step: retrain the models with flights of the agents themselves.

## Team (group 1)

| Name | Main contributions |
|---|---|
| Alban Derue | LSTM identification, SAC on the LSTM (`03/lstm_sac`), landing on the car in 2D (`05/lstm_2d`) |
| Edouard Zablit | data analysis (`01`), RNN identification, SAC on the RNN (`03/rnn_sac`), x axis and 2D/3D landing (`05/rnn_xyz`) |
| Johary Meraud | RNN + SAC robustness to delays and defects (`03/rnn_robustness`) |
| Maxime Boulin | FNN identification (`02/fnn`), rotation (`04`) |
| Milo Brugnon | linear ARX model (`02/arx`) |
