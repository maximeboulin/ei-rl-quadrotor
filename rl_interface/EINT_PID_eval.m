%% PID baseline flies the "real" drone (UAV_model.slx), same loop and plot as EINT_RL_eval
%  Same simulation and plot as EINT_RL_eval.m, but rl_controller.m flies pid_law (the PID of
%  src/baseline.py) instead of the RL policy. Run EINT_RL_eval.m and this script to compare.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', 'matlab'));

global USE_PID
USE_PID = true;                % rl_controller.m now flies the PID (pid_law)

random_init;                   % noise seeds
Ts = 0.05;
T_end = 10;
g_noise = 0.1;                 % 0 to start, 0.1 for tuning, higher to test robustness

clear rl_controller
out = sim('EINT_RL_closed_loop');
L = out.rl_log;                % t, z, vz, az, u, z_ref
t = L(:, 1); z = L(:, 2); u = L(:, 5); z_ref = L(:, 6);

figure('Name', 'PID baseline on the simulator');
subplot(2, 1, 1); hold on; grid on;
plot(t, z_ref, '--', t, z, 'LineWidth', 1.2); yline(-0.25, ':');
ylabel('z [m]'); legend('reference', 'z', 'ground');
title(sprintf('g\\_noise = %g: RMSE %.2f m, min z %.2f m', g_noise, sqrt(mean((z_ref - z).^2)), min(z)));
subplot(2, 1, 2); stairs(t, u); grid on; ylabel('u'); xlabel('t [s]');


USE_PID = false;
clear rl_controller
