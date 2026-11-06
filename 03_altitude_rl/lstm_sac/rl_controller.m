function [u, z_ref, obs] = rl_controller(t, z, vz, az_prev, u_prev)
%RL_CONTROLLER  The trained policy, called at each step by EINT_RL_closed_loop.slx.
%   Write mission() and observation() at the end of this file. policy.mat is written by
%   export_for_matlab (export_policy.py).
persistent P
global USE_PID                       % set to true by EINT_PID_eval.m to fly the PID baseline instead
if isempty(P)
    P = load(fullfile(fileparts(mfilename('fullpath')), 'policy.mat'));
    P.acts = strsplit(P.activations, ',');
end
z_ref = mission(t);
if ~isempty(USE_PID) && USE_PID
    u = pid_law(t, z, vz, z_ref); obs = [];
    return
end
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
% Same as DroneEnv._observation in src/env.py, in the same order.
% I: integral of the error, updated once per time step (reset by "clear rl_controller").
persistent I t_last
if isempty(I)
    I = 0; t_last = -1;
end
if m.t > t_last
    I = min(max(I + 0.05 * (m.z_ref - m.z), -1), 1);
    t_last = m.t;
end
obs = [m.z_ref - m.z; m.vz; m.az_prev / 10; m.u_prev; m.z_ref; I];
end

function z_ref = mission(t)
if t < 1
    z_ref = 0;
elseif t < 5
    z_ref = 0.5;
else
    z_ref = -0.25;
end
end

function u = pid_law(t, z, vz, z_ref)
% Same as PID in src/baseline.py: u = u_hover + Kp*e + Ki*int(e) - Kd*vz
persistent I t_last
if isempty(I)
    I = 0; t_last = -1;
end
e = z_ref - z;
if t > t_last
    I = min(max(I + 0.05 * e, -1), 1);
    t_last = t;
end
u = 0.385 + 0.4 * e + 0.05 * I - 0.3 * vz;
u = min(max(u, 0), 0.8);
end
