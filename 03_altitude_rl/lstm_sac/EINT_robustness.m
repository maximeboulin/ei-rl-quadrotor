%% Robustness to sensor noise: RL policy vs PID on the "real" drone (UAV_model.slx)
%  For each noise level g_noise, flies the mission with the RL policy and with the PID
%  (same noise seeds for both), several times, and reports the mean RMSE.
%  Needs policy.mat: run the training and the export in train_sac.ipynb first.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', '..', 'matlab'));

levels  = [0 0.1 0.2 0.3 0.5];   % g_noise values to test
n_runs  = 3;                     % flights per level (different noise seeds)
Ts = 0.05;
T_end = 10;

global USE_PID
rmse = zeros(numel(levels), 2);  % columns: RL, PID
for i = 1:numel(levels)
    g_noise = levels(i);
    r = zeros(n_runs, 2);
    for run = 1:n_runs
        random_init;             % new noise seeds, shared by RL and PID for this run
        Ts = 0.05;               % random_init.m overwrites Ts and g_noise: set them AFTER it
        g_noise = levels(i);
        for k = 1:2
            USE_PID = (k == 2);
            clear rl_controller
            out = sim('EINT_RL_closed_loop');
            L = out.rl_log;      % t, z, vz, az, u, z_ref
            r(run, k) = sqrt(mean((L(:, 6) - L(:, 2)).^2));
        end
    end
    rmse(i, :) = mean(r, 1);
    fprintf('g_noise = %.2f   RL: RMSE %.3f m   PID: RMSE %.3f m\n', g_noise, rmse(i, 1), rmse(i, 2));
end
USE_PID = false;
clear rl_controller

figure('Name', 'Robustness to sensor noise');
plot(levels, rmse(:, 1), 'o-', levels, rmse(:, 2), 's-', 'LineWidth', 1.2); grid on;
xlabel('g\_noise'); ylabel('mean RMSE [m]'); legend('RL (SAC)', 'PID', 'Location', 'best');
title(sprintf('RMSE vs sensor noise (%d flights per level)', n_runs));
