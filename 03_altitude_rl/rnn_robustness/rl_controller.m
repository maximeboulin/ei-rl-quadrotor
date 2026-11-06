function [u, z_ref, obs] = rl_controller(t, z, vz, az_prev, u_prev)
%RL_CONTROLLER  The trained policy, called at each step by EINT_RL_closed_loop.slx.
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
persistent ie zref_prev
if isempty(ie), ie = 0; zref_prev = m.z_ref; end
Ts = 0.05; IE_MAX = 0.2; E_INT = 0.3;
if abs(m.z_ref - zref_prev) > 0.1      % nouvelle marche : remise à zéro de lintegrale
    ie = 0;
end
zref_prev = m.z_ref;
e = m.z_ref - m.z;
if abs(e) < E_INT
    ie = min(max(ie + e*Ts, -IE_MAX), IE_MAX);
end
obs = [e/0.5; m.vz/1.0; m.az_prev/10.0; m.u_prev; m.z_ref/0.5; ie/IE_MAX];
end

function z_ref = mission(t)
EPS_T = 1e-6;
if t < 1 - EPS_T
    z_ref = 0;
elseif t < 6 - EPS_T
    z_ref = 0.5;
else
    z_ref = -0.25;
end
end
