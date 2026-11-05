%% Step A + B of the 2D landing: understand the 4 motors, and validate uav_plant.m
%  Run from the landing/ folder. Needs ../rl_interface/policy.mat (already exported).
clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', 'matlab'), fullfile(here, '..', 'rl_interface'));
if isfile('motor_test_log.txt'), delete('motor_test_log.txt'); end
diary('motor_test_log.txt');          % everything printed below is also saved in this file
Ts = 0.05;
u_hover = 0.385;

%% Step B: same commands in Simulink and in uav_plant.m -> same altitude?
% Fly the Track B mission in Simulink without noise, then replay the same commands in uav_plant.m
random_init;  Ts = 0.05;  T_end = 10;  g_noise = 0;     % random_init overwrites Ts and g_noise
global USE_PID
USE_PID = false;
clear rl_controller
out = sim('EINT_RL_closed_loop');
L = out.rl_log;                                % t, z, vz, az, u, z_ref
s = struct('x', zeros(12, 1), 'm', zeros(4, 1));
z_plant = zeros(size(L, 1), 1);
for k = 1:size(L, 1)
    z_plant(k) = s.x(3);                       % altitude at the start of step k, like the ZOH in Simulink
    s = uav_plant(s, L(k, 5) * ones(4, 1), Ts);
end
fprintf('Step B: max |z_simulink - z_plant| = %.4f m over %d steps\n', max(abs(L(:, 2) - z_plant)), size(L, 1));
figure('Name', 'Step B: uav_plant vs Simulink');
plot(L(:, 1), L(:, 2), L(:, 1), z_plant, '--', 'LineWidth', 1.2); grid on;
xlabel('t [s]'); ylabel('z [m]'); legend('UAV\_model.slx (Simulink)', 'uav\_plant.m');
title('Same commands, same drone?');

%% Step A: push one motor a little more than the others, for 1 s, and look at what happens
names = {'phi (roll)', 'theta (pitch)', 'psi (yaw)', 'x', 'y', 'z'};
idx   = [7 8 9 1 2 3];
fprintf('\nStep A: motor i at u_hover + 0.05, the 3 others at u_hover, during 1 s\n');
fprintf('%-8s', 'motor'); fprintf('%14s', names{:}); fprintf('\n');
res = zeros(4, numel(idx));
for i = 1:4
    s = struct('x', zeros(12, 1), 'm', u_hover * ones(4, 1));   % motors already spun up at hover
    u = u_hover * ones(4, 1);  u(i) = u(i) + 0.05;
    for k = 1:round(1 / Ts)
        s = uav_plant(s, u, Ts);
    end
    res(i, :) = s.x(idx)';
    fprintf('%-8s', sprintf('u%d', i)); fprintf('%14.4f', res(i, :)); fprintf('\n');
end
fprintf('\nReference: all 4 motors at u_hover during 1 s\n');
s = struct('x', zeros(12, 1), 'm', u_hover * ones(4, 1));
for k = 1:round(1 / Ts)
    s = uav_plant(s, u_hover * ones(4, 1), Ts);
end
fprintf('%-8s', 'all'); fprintf('%14.4f', s.x(idx)'); fprintf('\n');
diary off
