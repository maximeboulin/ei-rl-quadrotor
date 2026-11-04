%% RL policy vs PID on the "real" drone (UAV_model.slx), on the same figure
%  Runs EINT_RL_closed_loop.slx twice with the SAME noise seeds: once with the RL policy
%  (policy.mat), once with the PID (pid_law in rl_controller.m), then overlays both flights.
%  Needs policy.mat: run the training and the export in 03_train_sac.ipynb first.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', 'matlab'));

random_init;                   % noise seeds, drawn once: both controllers see the same noise
Ts = 0.05;
T_end = 10;
g_noise = 0.1;                 % 0 without sensor noise, 0.1 for the nominal test

global USE_PID
names = {'RL (SAC)', 'PID'};
res = cell(1, 2);
for k = 1:2
    USE_PID = (k == 2);        % false: RL policy, true: PID
    clear rl_controller        % resets the persistent memories (integral, policy)
    out = sim('EINT_RL_closed_loop');
    res{k} = out.rl_log;       % columns: t, z, vz, az, u, z_ref
end
USE_PID = false;
clear rl_controller

figure('Name', 'RL vs PID on the simulator');
subplot(2, 1, 1); hold on; grid on;
t = res{1}(:, 1); z_ref = res{1}(:, 6);
plot(t, z_ref, 'k--', 'LineWidth', 1.2);
txt = '';
for k = 1:2
    L = res{k};
    plot(L(:, 1), L(:, 2), 'LineWidth', 1.2);
    rmse = sqrt(mean((L(:, 6) - L(:, 2)).^2));
    txt = [txt, sprintf('%s: RMSE %.3f m   ', names{k}, rmse)];
    fprintf('%s: RMSE %.3f m, min z %.2f m\n', names{k}, rmse, min(L(:, 2)));
end
ylabel('z [m]'); legend(['reference', names], 'Location', 'best');
title(sprintf('g\\_noise = %g   |   %s', g_noise, txt));

subplot(2, 1, 2); hold on; grid on;
for k = 1:2
    stairs(res{k}(:, 1), res{k}(:, 5));
end
ylabel('u'); xlabel('t [s]'); legend(names, 'Location', 'best');
