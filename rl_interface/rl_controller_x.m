function a = rl_controller_x(obs)
%RL_CONTROLLER_X  The trained policy of the x axis (notebooks/track_b_x.ipynb), written in policy_x.mat.
%   obs = [x_target - x; v_target - vx; theta / 0.2; wy; a_prev]  (same order as DroneEnvX.observation in drone_env_x.py)
%   a   = action in [-1, 1]; the difference between motors 3 and 1 is delta = 0.1 * a.
%   Same network computation as rl_controller.m. At the first call, checks the export: the network is run on observations
%   saved by the notebook and must give the same actions as Python ("Check x policy OK").
persistent P
if isempty(P)
    P = load(fullfile(fileparts(mfilename('fullpath')), 'policy_x.mat'));
    P.acts = strsplit(P.activations, ',');
    err = 0;
    for k = 1:size(P.test_obs, 1)
        err = max(err, abs(network(P, P.test_obs(k, :)') - P.test_act(k)));
    end
    if err > 1e-4
        error('policy_x.mat: MATLAB and Python give different actions (max difference %.2g)', err);
    end
    fprintf('Check x policy OK (max difference with Python %.1e)\n', err);
end
a = network(P, obs(:));
end

function a = network(P, x)
for i = 1:double(P.n_layers)
    x = P.(sprintf('W%d', i)) * x + P.(sprintf('b%d', i));
    switch P.acts{i}
        case 'relu', x = max(x, 0);
        case 'tanh', x = tanh(x);
    end
end
lo = P.action_low(:);  hi = P.action_high(:);
switch P.output
    case 'tanh_scale', x = lo + 0.5 * (tanh(x) + 1) .* (hi - lo);   % SAC
    case 'scale',      x = lo + 0.5 * (x + 1) .* (hi - lo);         % TD3
    otherwise,         x = min(max(x, lo), hi);                     % PPO
end
a = x(1);
end
