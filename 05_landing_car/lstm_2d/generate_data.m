%% Step 1 of the 2D identification: generate flight data with the "real" drone (uav_plant.m)
%  Cascade control, as on a real drone (without attitude loop, it flips over in less than 1 s):
%   - an attitude PD (part of the drone) makes the pitch follow a reference theta_cmd and keeps
%     roll and yaw at 0;
%   - the two inputs of the system, U_common and theta_cmd, are random steps (like the inputs of
%     the provided CSV files). They are EXOGENOUS: no feedback from the measurements, otherwise
%     the inputs would contain information on the outputs and the identified model would be wrong;
%   - every measurement carries the sensor noise of UAV_model.slx (g_noise = 0.1);
%   - the stabilizer itself only sees the noisy measurements.
%  We record the commands actually applied to the motors and the measured outputs.
%  Output: ../data/raw/landing_data.mat (not committed, like the provided CSV files)
clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', '..', 'matlab'));
if isfile('generate_data_log.txt'), delete('generate_data_log.txt'); end
diary('generate_data_log.txt');
rng(0);                                    % reproducible data set

%% Parameters
N_exp   = 500;                             % number of experiments
T_exp   = 20;                              % duration of one experiment [s]
Ts      = 0.05;                            % sampling period [s]
g_noise = 0.1;                             % sensor noise, as in UAV_model.slx
u_hover = 0.385;
noise_std = sqrt(1e-4 / Ts) * g_noise;     % Band-Limited White Noise blocks: Cov = 1e-4
n = round(T_exp / Ts);

% stabilizer (only used to collect data)
Kth = 0.08; Kq  = 0.04;  d_max = 0.1;      % attitude PD: angle -> motor difference
% random excitation: piecewise-constant levels held between t_min and t_max
t_min = 0.5;  t_max = 3;
uc_range = u_hover + [-0.1 0.1];           % common thrust around hover
th_range = [-0.25 0.25];                   % pitch reference [rad] (about +-14 deg)

%% Storage (one row per experiment, one column per time step, like the provided CSV files)
U_common = zeros(N_exp, n, 'single');      % common thrust: (u1 + u2 + u3 + u4) / 4
U_diff   = zeros(N_exp, n, 'single');      % front/rear difference: (u3 - u1) / 2
TH_CMD   = zeros(N_exp, n, 'single');      % pitch reference sent to the attitude PD (cascade input)
U_motors = zeros(N_exp, n, 4, 'single');   % the 4 commands actually sent
Y        = zeros(N_exp, n, 14, 'single');  % measured: 12 states + ax + az
names_Y  = {'x','y','z','vx','vy','vz','phi','theta','psi','wx','wy','wz','ax','az'};
lost     = false(N_exp, 1);                % experiments where the drone lost control

steps = @(range) piecewise_levels(n, Ts, t_min, t_max, range);
tic;
for e = 1:N_exp
    uc_ref = steps(uc_range);  uc_ref(1:round(1 / Ts)) = u_hover;   % 1 s of hover at the start
    th_ref = steps(th_range);  th_ref(1:round(1 / Ts)) = 0;
    s = struct('x', zeros(12, 1), 'm', u_hover * ones(4, 1));  % hovering at z = 0
    meas = s.x + noise_std * randn(12, 1);
    prev = meas(7:9);
    for k = 1:n
        % ---- stabilizer, on the noisy measurements ----
        uz = uc_ref(k);                                         % exogenous common thrust
        rates = (meas(7:9) - prev) / Ts;  prev = meas(7:9);
        th_cmd  = th_ref(k);                                    % exogenous pitch reference
        d_pitch = Kth * (th_cmd - meas(8)) - Kq * rates(2);
        d_roll  = Kth * (0 - meas(7)) - Kq * rates(1);
        d_yaw   = Kth * (0 - meas(9)) - Kq * rates(3);
        d_pitch = min(max(d_pitch, -d_max), d_max);
        d_roll  = min(max(d_roll,  -d_max), d_max);
        d_yaw   = min(max(d_yaw,   -d_max), d_max);
        u = [uz - d_pitch + d_yaw; uz + d_roll - d_yaw; uz + d_pitch + d_yaw; uz - d_roll - d_yaw];
        u = min(max(u, 0), 1);
        % ---- the drone ----
        [s, ~, dxdt] = uav_plant(s, u, Ts);
        meas = s.x + noise_std * randn(12, 1);
        % ---- record ----
        U_motors(e, k, :) = u;
        U_common(e, k) = mean(u);
        U_diff(e, k)   = (u(3) - u(1)) / 2;
        TH_CMD(e, k)   = th_cmd;
        Y(e, k, :) = [meas; dxdt(4) + noise_std * randn; dxdt(6) + noise_std * randn];
        if any(abs(s.x(7:9)) > 1) || any(~isfinite(s.x))
            lost(e) = true;
        end
    end
    if mod(e, 25) == 0
        el = toc;
        fprintf('%d / %d experiments  (%.0f s elapsed, about %.0f s left)\n', e, N_exp, el, el / e * (N_exp - e));
    end
end
fprintf('Experiments where the drone lost control (|angle| > 1 rad): %d / %d\n', sum(lost), N_exp);

%% Save
out = fullfile(here, '..', '..', 'data', 'raw', 'landing_data.mat');
save(out, 'U_common', 'U_diff', 'TH_CMD', 'U_motors', 'Y', 'names_Y', 'lost', 'Ts', 'g_noise', 'u_hover', '-v7');
fprintf('Saved %s\n', out);
fprintf('U_common range [%.3f, %.3f], U_diff range [%.3f, %.3f]\n', min(U_common(:)), max(U_common(:)), min(U_diff(:)), max(U_diff(:)));
fprintf('theta range [%.3f, %.3f] rad, ax range [%.2f, %.2f], az range [%.2f, %.2f] m/s^2\n', ...
    min(Y(:, :, 8), [], 'all'), max(Y(:, :, 8), [], 'all'), min(Y(:, :, 13), [], 'all'), max(Y(:, :, 13), [], 'all'), ...
    min(Y(:, :, 14), [], 'all'), max(Y(:, :, 14), [], 'all'));

%% Look at one experiment
t = (1:n) * Ts;  e = 1;
fig = figure('Name', 'One experiment', 'Position', [100 100 900 700]);
subplot(4, 1, 1); stairs(t, [U_common(e, :); U_diff(e, :)]'); grid on; ylabel('commands'); legend('common', 'diff');
subplot(4, 1, 2); plot(t, squeeze(Y(e, :, 8))); grid on; ylabel('\theta [rad]');
subplot(4, 1, 3); plot(t, squeeze(Y(e, :, [13 14]))); grid on; ylabel('a [m/s^2]'); legend('a_x', 'a_z');
subplot(4, 1, 4); plot(t, squeeze(Y(e, :, [1 3]))); grid on; ylabel('[m]'); legend('x', 'z'); xlabel('t [s]');
if ~isfolder('figures'), mkdir('figures'); end
saveas(fig, fullfile('figures', 'data_example.png'));
diary off

function v = piecewise_levels(n, Ts, t_min, t_max, range)
% Random piecewise-constant signal: a level drawn uniformly in range, held between t_min and t_max
v = zeros(1, n);  k = 1;
while k <= n
    len = round((t_min + (t_max - t_min) * rand) / Ts);
    v(k:min(n, k + len - 1)) = range(1) + (range(2) - range(1)) * rand;
    k = k + len;
end
end
