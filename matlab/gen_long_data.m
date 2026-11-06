%% gen_long_data.m - Long simulations (10 min) of the drone, never seen by the model
%  Used by notebook 02, section 6.2: tests the RNN on horizons 20 x longer than its training sequences.
%  Run from the "Matlab files" folder. Duration: about 1 to 3 min.
%  Writes long_u.csv (commands) and long_az.csv (acceleration), one row per simulation, same format as the provided CSV files.

clear; close all; clc;

N_SIM  = 5;        % number of long simulations
T_END  = 600;      % duration of each simulation [s] (10 min)
Ts     = 0.05;     % sampling period
N      = round(T_END / Ts);   % number of samples per simulation (12000)

%% 1) Commands u: same kind of signal as the provided data (APRBS)
%  - u = 0 during the first 2 s (drone at rest), like in the provided data
%  - then steps of random level, uniform in [-1, 1], and random duration between 5 and 39 samples (0.25 to 1.95 s)
rng(2026);                     % fixed seed: the same commands every time the script is run
U = zeros(N_SIM, N);
for i = 1:N_SIM
    k = 41;                    % first sample after the 2 s at rest
    while k <= N
        len   = randi([5 39]);                         % duration of the step
        level = 2 * rand - 1;                          % level of the step in [-1, 1]
        U(i, k:min(k + len - 1, N)) = level;
        k = k + len;
    end
end

%% 2) Small model: From Workspace -> drone (same command on the 4 motors) -> To Workspace
random_init;                   % creates the noise seeds (and Ts, overwritten just after)
Ts = 0.05;
g_noise = 0;                   % no noise: the error measured in the notebook will be the model error only
mdl = 'long_model';
if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile([mdl '.slx']), delete([mdl '.slx']); end
new_system(mdl);
load_system('EINT_system_simulation');
add_block('simulink/Sources/From Workspace', [mdl '/u_in'], ...
    'VariableName', 'u_ts', 'SampleTime', 'Ts', 'Interpolate', 'off', ...
    'OutputAfterFinalValue', 'Holding final value');
add_block('EINT_system_simulation/Subsystem Reference', [mdl '/UAV']);
add_block('simulink/Sinks/To Workspace', [mdl '/out'], ...
    'VariableName', 'simout', 'SaveFormat', 'Structure With Time', 'SampleTime', 'Ts', 'MaxDataPoints', 'inf');
for k = 1:4
    add_line(mdl, 'u_in/1', sprintf('UAV/%d', k));
end
add_line(mdl, 'UAV/19', 'out/1');                   % all_outputs (az = column 9)
cs = attachConfigSetCopy(mdl, getActiveConfigSet('EINT_system_simulation'), true);
setActiveConfigSet(mdl, cs.Name);
set_param(mdl, 'StopTime', num2str((N - 1) * Ts));
save_system(mdl);

%% 3) Simulations
t = (0:N-1)' * Ts;
AZ = zeros(N_SIM, N);
for i = 1:N_SIM
    u_ts = [t, U(i, :)'];
    out = sim(mdl);
    az = out.simout.signals.values(:, 9);
    AZ(i, :) = az(1:N)';
    fprintf('simulation %d / %d done\n', i, N_SIM);
end
writematrix(U, 'long_u.csv');
writematrix(AZ, 'long_az.csv');
fprintf('\nlong_u.csv and long_az.csv written in "Matlab files": copy them to results/long_sims/ in the repository.\n');
