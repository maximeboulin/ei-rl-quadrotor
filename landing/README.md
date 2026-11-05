# Axis 2 – Landing on a moving car (2D: x, z)

Client request (Friday demo): the drone takes off from the platform of the car, flies, and lands back
on the car **while the car is moving**. 2D problem: altitude `z` and longitudinal position `x`.

## What we know about the drone (from UAV_model.slx)

- 4 inputs `u1..u4` = the **4 motors**. Each one goes through a motor lag `1/(0.2 s + 1)`, a `tanh`
  saturation and a gain `10`, then into the protected physical model `quadcopter_model(x, u)`.
- 12 states: `[x y z vx vy vz phi theta psi wx wy wz]`. Sensor noise (`g_noise`) is added on the outputs only.
- In `EINT_RL_closed_loop.slx` the same `u` is sent to the 4 motors: the drone can only go up/down.
  To move in `x`, the drone must **tilt** (pitch), i.e. front and rear motors must receive different commands.
- The 0.2 s motor lag is the time constant (tau ≈ 0.21 s) we identified from the data.

## Scenario

- t = 0: car at rest, drone on the platform (x = 0, z = 0).
- The drone takes off and climbs to 0.5 m.
- t ≈ 2 s: the car starts and accelerates smoothly up to 1 m/s (≈ 4 km/h, manoeuvring speed).
- The drone follows the car, places itself above the platform, then descends and lands on it.

Success at touchdown (z back to the platform level 0):
|x − x_car| < 0.10 m, |vz| < 0.3 m/s, |vx − v_car| < 0.2 m/s.

## Method (same path as the vertical axis: data -> model -> environment -> RL agent)

| Step | Vertical axis (done) | 2D axis |
|---|---|---|
| 1. Data | CSV files provided | `generate_data.m`: closed-loop flights of `uav_plant.m`, random steps, sensor noise |
| 2. Validation | 4 faulty experiments removed | check noise, lost experiments |
| 3. Model | LSTM u -> a_z | model (common thrust, front/rear difference) -> a_x, a_z, pitch |
| 4. Environment | `src/env.py` | 2D environment with the moving car |
| 5. Agent | SAC on u | SAC 2D (common thrust + front/rear difference) |
| 6. Test | `EINT_RL_eval` on UAV_model.slx | flight on `uav_plant.m` (the real model, 4 motors) |

## Files (MATLAB, run from this folder)

| File | Role |
|---|---|
| `uav_plant.m` | the drone without Simulink: same blocks as UAV_model.slx around `quadcopter_model.p`, 4 independent motors |
| `motor_test.m` | which motor tilts the drone in which direction; check of uav_plant against Simulink |
| `car_motion.m` | car position, speed and acceleration at time t |
| `generate_data.m` | step 1: 500 closed-loop experiments of 20 s -> `../data/raw/landing_data.mat` |

## Motor layout (motor_test.m, one motor at u_hover + 0.05 during 1 s)

| motor | roll phi | pitch theta | yaw psi | x | y |
|---|---|---|---|---|---|
| u1 | +0.03 | **−1.45** | +0.72 | **−0.83** | −0.24 |
| u2 | **+1.45** | −0.03 | −0.72 | −0.23 | **−0.83** |
| u3 | −0.03 | **+1.45** | +0.72 | **+0.83** | +0.24 |
| u4 | **−1.45** | +0.03 | −0.72 | +0.23 | **+0.83** |

"+" configuration: u1/u3 on the x axis (pitch), u2/u4 on the y axis (roll); u1, u3 and u2, u4 spin in
opposite directions (yaw). To move +x: u3 up and u1 down by the same amount (yaw effects cancel).
Very sensitive: +0.05 on one motor gives 83° of pitch in 1 s, hence small gains and an angle limit.
