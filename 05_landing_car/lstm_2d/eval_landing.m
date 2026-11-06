%% Step 6 of the 2D axis: land on the moving car with the "real" drone (uav_plant.m, 4 motors)
%  Hybrid control (same as src/env_2d.py):
%   - horizontal: the trained policy (policy_2d.mat, written by 06_train_sac_landing.ipynb) gives theta_cmd;
%   - vertical: a classical altitude law (descent_law) gives U_common: follow at 0.5 m, then descend at
%     0.3 m/s once aligned with the car;
%   - the attitude PD of generate_data.m turns theta_cmd into the 4 motor commands.
%  Baseline: same altitude law and attitude PD, with a PD on x instead of the policy for theta_cmd.
%  Both controllers only see the noisy measurements (g_noise = 0.1, as in UAV_model.slx).
clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', '..', 'matlab'));
if isfile('eval_landing_log.txt'), delete('eval_landing_log.txt'); end
diary('eval_landing_log.txt');

%% Parameters (identical to src/env_2d.py)
Ts = 0.05;  N_STEPS = 300;  Z0 = 0.5;  u_hover = 0.385;
UC = [0.285 0.485];  TH_MAX = 0.25;  DTH_MAX = 0.02;  I_MAX = 1;
car_speeds = [0.5 1.0 1.5];  P.car_x0 = 0.75;               % car scenario (car_motion.m): 3 speeds
TOL_X = 0.10;  TOL_VZ = 0.3;  TOL_VX = 0.2;               % landing success criteria
g_noise = 0.1;  noise_std = sqrt(1e-4 / Ts) * g_noise;
Kth = 0.08;  Kq = 0.04;  d_max = 0.1;                     % attitude PD (generate_data.m)
seeds = 1:3;                                              % 3 flights per speed (sensor noise)

pol = load(fullfile(here, 'policy_2d.mat'));
pol.acts = strsplit(pol.activations, ',');

names = {'SAC', 'PID'};
R = struct();
for c = 1:2
    i = 0;
    for v = car_speeds
      P.car_v = v;
      for s = seeds
        rng(s);  i = i + 1;
        R(c, i).log = fly(names{c}, pol, P, Ts, N_STEPS, Z0, u_hover, UC, TH_MAX, DTH_MAX, I_MAX, ...
                          noise_std, Kth, Kq, d_max);
        L = R(c, i).log;
        ex = L.x_car(end) - L.x(end);  evx = L.v_car(end) - L.vx(end);  vz = L.vz(end);
        R(c, i).success = L.landed && abs(ex) < TOL_X && abs(vz) < TOL_VZ && abs(evx) < TOL_VX;
        fprintf('%s car %.1f m/s seed %d: landed %d at t = %5.2f s | x - x_car = %+.3f m | vz = %+.2f m/s | vx - v_car = %+.2f m/s -> %s\n', ...
            names{c}, v, s, L.landed, L.t(end), -ex, vz, -evx, string(R(c, i).success));
      end
    end
    fprintf('%s: %d / %d successful landings\n\n', names{c}, sum([R(c, :).success]), i);
end

%% Plot one flight of each controller
cols = lines(2);
fig = figure('Name', 'Landing on the moving car', 'Position', [100 100 1000 800]);
for c = 1:2
    L = R(c, 4).log;                                     % car at 1.0 m/s, seed 1
    subplot(4, 1, 1); hold on; plot(L.t, L.x - L.x_car, 'Color', cols(c, :), 'DisplayName', names{c});
    subplot(4, 1, 2); hold on; plot(L.t, L.z, 'Color', cols(c, :), 'DisplayName', names{c});
    subplot(4, 1, 3); hold on; plot(L.t, L.theta, 'Color', cols(c, :), 'DisplayName', names{c});
    subplot(4, 1, 4); hold on; stairs(L.t, L.uc, 'Color', cols(c, :), 'DisplayName', names{c});
end
labels = {'x - x_{car} [m]', 'z above platform [m]', '\theta [rad]', 'U_{common}'};
for i = 1:4
    subplot(4, 1, i); grid on; ylabel(labels{i}); legend('Location', 'best');
end
subplot(4, 1, 2); yline(0, 'k--', 'platform', 'HandleVisibility', 'off');
subplot(4, 1, 4); xlabel('t [s]');
if ~isfolder('figures'), mkdir('figures'); end
saveas(fig, fullfile('figures', 'landing_sac_vs_pid.png'));
diary off;
save(fullfile(here, 'eval_landing_results.mat'), 'R', 'names', 'car_speeds', 'seeds', 'P', '-v7');   % for the animation


%% ------------------------------------------------------------------------------------------------
function L = fly(ctrl, pol, P, Ts, N_STEPS, Z0, u_hover, UC, TH_MAX, DTH_MAX, I_MAX, noise_std, Kth, Kq, d_max)
s = struct('x', zeros(12, 1), 'm', u_hover * ones(4, 1));     % hovering ...
s.x(3) = Z0;                                                 % ... 0.5 m above the platform
meas = s.x + noise_std * randn(12, 1);
prev_att = meas(7:9);  th_prev_obs = meas(8);
uc_prev = u_hover;  tc_prev = 0;  I = 0;
mem = struct('phase', 0, 'ok', 0, 'z_ref', Z0, 'Iz', 0);    % altitude law memory (phase 0 follow, 1 descend)
L = struct('t', [], 'x', [], 'z', [], 'vx', [], 'vz', [], 'theta', [], 'x_car', [], 'v_car', [], ...
           'uc', [], 'tc', [], 'landed', false);
for k = 0:N_STEPS - 1
    t = k * Ts;
    [x_car, v_car, a_car] = car_motion(t, P);
    ex = x_car - meas(1);  evx = v_car - meas(4);
    % ---- horizontal: theta_cmd ----
    if strcmp(ctrl, 'SAC')
        I = min(max(I + Ts * ex, -I_MAX), I_MAX);
        q = (meas(8) - th_prev_obs) / Ts;
        obs = [ex; meas(3); evx; meas(6); meas(8) * 5; q; uc_prev; tc_prev * 4; I; min(t, 5) / 5];
        tc = policy(pol, obs);
    else
        tc = 0.15 * ex + 0.3 * evx + a_car / 9.81;          % PD on x + car acceleration
    end
    th_prev_obs = meas(8);
    tc = min(max(tc, -TH_MAX), TH_MAX);
    % ---- vertical: U_common from the altitude law ----
    [uc, mem] = descent_law(meas(3), meas(6), ex, evx, mem, Ts, Z0, u_hover, UC);
    tc = tc_prev + min(max(tc - tc_prev, -DTH_MAX), DTH_MAX);   % rate limit (src/env_2d.py)
    % ---- inner loop: attitude PD (part of the drone, as in generate_data.m) ----
    rates = (meas(7:9) - prev_att) / Ts;  prev_att = meas(7:9);
    d_pitch = min(max(Kth * (tc - meas(8)) - Kq * rates(2), -d_max), d_max);
    d_roll  = min(max(Kth * (0 - meas(7))  - Kq * rates(1), -d_max), d_max);
    d_yaw   = min(max(Kth * (0 - meas(9))  - Kq * rates(3), -d_max), d_max);
    u = [uc - d_pitch + d_yaw; uc + d_roll - d_yaw; uc + d_pitch + d_yaw; uc - d_roll - d_yaw];
    u = min(max(u, 0), 1);
    % ---- the drone ----
    s = uav_plant(s, u, Ts);
    meas = s.x + noise_std * randn(12, 1);
    uc_prev = uc;  tc_prev = tc;
    % ---- log (true state) ----
    [x_car, v_car] = car_motion(t + Ts, P);
    L.t(end + 1) = t + Ts;  L.x(end + 1) = s.x(1);  L.z(end + 1) = s.x(3);
    L.vx(end + 1) = s.x(4);  L.vz(end + 1) = s.x(6);  L.theta(end + 1) = s.x(8);
    L.x_car(end + 1) = x_car;  L.v_car(end + 1) = v_car;  L.uc(end + 1) = uc;  L.tc(end + 1) = tc;
    if s.x(3) <= 0                                           % touchdown on the platform
        L.landed = true;
        return
    end
    if abs(s.x(8)) > 0.8 || abs(x_car - s.x(1)) > 3          % drone lost
        return
    end
end
end

function a = policy(P, obs)
% The SAC actor exported by export_policy.py (same code as 03_altitude_rl/lstm_sac/rl_controller.m)
x = obs(:);
for i = 1:double(P.n_layers)
    x = P.(sprintf('W%d', i)) * x + P.(sprintf('b%d', i));
    switch P.acts{i}
        case 'relu', x = max(x, 0);
        case 'tanh', x = tanh(x);
    end
end
lo = P.action_low(:);  hi = P.action_high(:);
a = lo + 0.5 * (tanh(x) + 1) .* (hi - lo);
end

function [uc, mem] = descent_law(z, vz, ex, evx, mem, Ts, Z0, u_hover, UC)
% Same as descent_law in src/env_2d.py: follow at Z0, descend at 0.3 m/s once aligned for 0.5 s,
% climb back to Z0 if the drone gets more than 0.25 m away from the car.
if mem.phase == 0
    if abs(ex) < 0.10 && abs(evx) < 0.15, mem.ok = mem.ok + Ts; else, mem.ok = 0; end
    if mem.ok >= 0.5, mem.phase = 1; end
else
    mem.z_ref = max(-0.05, mem.z_ref - 0.3 * Ts);
    if abs(ex) > 0.25
        mem.phase = 0;  mem.ok = 0;  mem.z_ref = max(mem.z_ref, Z0);
    end
end
ez = mem.z_ref - z;
mem.Iz = min(max(mem.Iz + Ts * ez, -1), 1);
uc = u_hover + 0.4 * ez + 0.05 * mem.Iz - 0.3 * vz;        % altitude PID (src/baseline.py)
uc = min(max(uc, UC(1)), UC(2));
end
