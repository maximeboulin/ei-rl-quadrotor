%% EINT_XZ_eval.m - Advanced task: take off, climb, move 1 m in x, land there, on the "real" drone (UAV_model.slx)
%  Builds a closed loop with 4 different motor commands (EINT_RL_closed_loop.slx sends the same command to the 4 motors)
%  and flies it: height = Track B policy (rl_controller.m + policy.mat), x = cascade controller (xz_controller.m).
%  Run from this folder (rl_interface). Duration: about 1 min.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', 'Matlab files'));

random_init;                   % noise seeds
Ts = 0.05;
T_end = 10;
g_noise = 0.1;                 % same as EINT_RL_eval.m

%% 1) Closed loop: measurements (sampled every Ts) -> xz_controller -> 4 motors
mdl = 'xz_closed_loop';
if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile([mdl '.slx']), delete([mdl '.slx']); end
new_system(mdl);
load_system('EINT_system_simulation');
add_block('EINT_system_simulation/Subsystem Reference', [mdl '/UAV']);
add_block('simulink/Sources/Clock', [mdl '/clock']);
% outputs of UAV_model used (port numbers): x 1, z 3, vx 4, vz 6, theta 8, wy 11, ax 13, az 15 (az twice: delayed and current)
meas = {'clock', 1; 'UAV', 3; 'UAV', 6; 'UAV', 15; 'UAV', 1; 'UAV', 4; 'UAV', 8; 'UAV', 11; 'UAV', 13; 'UAV', 15};
N_IN = size(meas, 1);
add_block('simulink/Signal Routing/Mux', [mdl '/mux_in'], 'Inputs', num2str(N_IN));
for i = 1:N_IN
    zoh = sprintf('zoh%d', i);
    add_block('simulink/Discrete/Zero-Order Hold', [mdl '/' zoh], 'SampleTime', 'Ts');
    add_line(mdl, sprintf('%s/%d', meas{i, 1}, meas{i, 2}), [zoh '/1']);
    if i == 4                                                       % az: previous step, as in EINT_RL_closed_loop.slx
        add_block('simulink/Discrete/Unit Delay', [mdl '/delay_az'], 'SampleTime', 'Ts');
        add_line(mdl, [zoh '/1'], 'delay_az/1');
        add_line(mdl, 'delay_az/1', sprintf('mux_in/%d', i));
    else
        add_line(mdl, [zoh '/1'], sprintf('mux_in/%d', i));
    end
end
add_block('simulink/User-Defined Functions/Interpreted MATLAB Function', [mdl '/controller'], ...
    'MATLABFcn', 'xz_controller(u)', 'OutputDimensions', '4', 'SampleTime', 'Ts');
add_line(mdl, 'mux_in/1', 'controller/1');
add_block('simulink/Signal Routing/Demux', [mdl '/demux'], 'Outputs', '4');
add_line(mdl, 'controller/1', 'demux/1');
for k = 1:4
    add_line(mdl, sprintf('demux/%d', k), sprintf('UAV/%d', k));
end
% log: measurements (10) + commands (4)
add_block('simulink/Signal Routing/Mux', [mdl '/mux_log'], 'Inputs', '2');
add_line(mdl, 'mux_in/1', 'mux_log/1');
add_line(mdl, 'controller/1', 'mux_log/2');
add_block('simulink/Sinks/To Workspace', [mdl '/log'], 'VariableName', 'xz_log', 'SaveFormat', 'Array', 'SampleTime', 'Ts');
add_line(mdl, 'mux_log/1', 'log/1');
cs = attachConfigSetCopy(mdl, getActiveConfigSet('EINT_system_simulation'), true);
setActiveConfigSet(mdl, cs.Name);
set_param(mdl, 'StopTime', num2str(T_end));
save_system(mdl);

%% 2) Flight
clear rl_controller xz_controller          % reset the memories (integral of the height policy, previous command)
out = sim(mdl);
L = out.xz_log;                            % t z vz az_prev x vx theta wy ax az u1 u2 u3 u4
t = L(:, 1); z = L(:, 2); x = L(:, 5); U = L(:, 11:14);
z_ref = arrayfun(@(s) (s >= 1 - 1e-6) * 0.5 + (s >= 5.5 - 1e-6) * (-0.75), t);   % mission of rl_controller.m
x_ref = (t >= 1.5 - 1e-6) * 1.0;                                                 % x_mission of xz_controller.m
writematrix(L, 'xz_flight.csv');

figure('Name', 'Advanced task: move in x and land');
subplot(3, 1, 1); plot(t, z_ref, '--', t, z, 'LineWidth', 1.2); grid on; ylabel('z [m]'); legend('reference', 'z');
title(sprintf('g\\_noise = %g: z RMSE %.2f m | x at 5.5 s %.2f m, final x %.2f m', g_noise, ...
      sqrt(mean((z - z_ref).^2)), x(find(t >= 5.5, 1)), x(end)));
subplot(3, 1, 2); plot(t, x_ref, '--', t, x, 'LineWidth', 1.2); grid on; ylabel('x [m]'); legend('reference', 'x');
subplot(3, 1, 3); stairs(t, U); grid on; ylabel('u'); xlabel('t [s]'); legend('u1', 'u2', 'u3', 'u4');
fprintf('Flight done. xz_flight.csv written in rl_interface.\n');
