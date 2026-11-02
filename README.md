# EINT – Data-Driven Control – Track B (Reinforcement Learning)

Quadrotor take-off / climb / landing from a ground vehicle platform, controlled by an RL policy
trained on an identified (data-driven) model, validated on `UAV_model.slx`.

Client demo: Friday 6 November 2026.

## Repository layout

```
configs/            Hyperparameters and settings (YAML) – single source of truth
data/raw/           Provided CSV files (bdd_in_mat_05.csv, bdd_out_mat_05.csv) – never modified
data/processed/     Cleaned / normalized data produced by src/data.py (git-ignored)
matlab/             Provided Simulink models, random_init.m, quadcopter_model.p
rl_interface/       Provided export tools (export_policy.py, rl_controller.m, ...)
src/
  data.py           Loading, validation (outliers, gaps, drift), normalization
  model.py          System identification (NN model + linear baseline)
  env.py            Custom Gymnasium environment around the identified model
  train.py          RL training with stable-baselines3
  evaluate.py       Evaluation, RL vs baseline, figures
  baseline.py       Linear controller (PID / LQ)
notebooks/          Exploration only – reusable code goes in src/
results/            Figures, logs, trained models (git-ignored except figures/)
slides/             Demo slides
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

- `main` is protected: no direct push, changes go through Merge Requests.
- One branch per task: `data/validation`, `model/nn-identification`, `rl/env`, `rl/reward`, ...
- Commit small, commit often. Do not commit large model checkpoints or TensorBoard logs.

## Team

| Name | Main role |
|------|-----------|
|      |           |
|      |           |
|      |           |
|      |           |
|      |           |
