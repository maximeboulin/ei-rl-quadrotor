function [u, z_ref, obs] = rl_controller_platform(t, z, vz, az_prev, u_prev)
%RL_CONTROLLER_PLATFORM  Copy of rl_controller.m (same policy.mat, same observation) whose mission lands on the
%   platform of the vehicle (z = 0) instead of the ground (z = -0.25). Used by platform_controller.m.
%   Write mission() and observation() at the end of this file. policy.mat is written by
%   export_for_matlab (export_policy.py).
persistent P
if isempty(P)
    P = load(fullfile(fileparts(mfilename('fullpath')), 'policy.mat'));
    P.acts = strsplit(P.activations, ',');
end
z_ref = mission(t);
m = struct('t', t, 'z', z, 'vz', vz, 'az_prev', az_prev, 'z_ref', z_ref, 'u_prev', u_prev);
obs = double(observation(m));
x = obs(:);
for i = 1:double(P.n_layers)
    x = P.(sprintf('W%d', i)) * x + P.(sprintf('b%d', i));
    switch P.acts{i}
        case 'relu', x = max(x, 0);
        case 'tanh', x = tanh(x);
    end
end
lo = P.action_low(:);
hi = P.action_high(:);
switch P.output
    case 'tanh_scale', x = lo + 0.5 * (tanh(x) + 1) .* (hi - lo);   % SAC
    case 'scale',      x = lo + 0.5 * (x + 1) .* (hi - lo);         % TD3 (tanh in the network)
    otherwise,         x = min(max(x, lo), hi);                     % PPO
end
u = x(1);
end

function obs = observation(m)
% The observation: the same computation as DroneEnv.observation in drone_env.py, in the same order.
%   m: struct of scalars t, z, vz, az_prev, z_ref, u_prev (the measurements of the drone).
%   1) height error  2) vertical speed  3) previous acceleration / 10  4) previous command
%   5) height error with the reference 0.5 s ahead  6) integral of the height error (only when |error| < 0.3 m, bounded to +/- 0.2, divided by 0.2)
% The integral is a memory: persistent variable, reset by "clear rl_controller" at the start of each flight.
persistent integral
if isempty(integral)
    integral = 0;
end
e = m.z_ref - m.z;
if abs(e) < 0.3                       % integrate only near the reference (anti wind-up), as I_ZONE in drone_env.py
    integral = min(max(integral + 0.05 * e, -0.2), 0.2);   % bounded to +/- I_MAX = 0.2
end
obs = [m.z_ref - m.z; m.vz; m.az_prev / 10; m.u_prev; mission(m.t + 0.5) - m.z; integral / 0.2];
end

function z_ref = mission(t)
% The reference altitude at time t [s], the same as mission() in drone_env.py (sharp steps, Figure 1):
%   0 m (platform) until 1 s, 0.5 m until 5.5 s, then 0 m (platform of the moving vehicle).
% EPS avoids ambiguities when t falls exactly on a switching time (same value in Python).
EPS = 1e-6;
if t < 1.0 - EPS
    z_ref = 0;
elseif t < 5.5 - EPS
    z_ref = 0.5;
else
    z_ref = 0;          % landing on the platform of the vehicle (rl_controller.m: -0.25, the ground)
end
end
