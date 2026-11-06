function u = xz_rl_controller(in)
%XZ_RL_CONTROLLER  Landing on the moving vehicle with TWO RL policies (advanced task in 2D: x + z).
%   Called every Ts by the model built in EINT_PLATFORM_RL_eval.m.
%   in = [t; z; vz; az_prev; x; vx; theta; wy; ax; az]  (measurements of the drone)
%   u  = [u1; u2; u3; u4]                                (commands of the 4 motors)
%   - height: Track B policy landing on the platform (rl_controller_platform.m) -> common command u_z;
%   - x: policy of notebooks/track_b_x.ipynb (rl_controller_x.m) -> difference delta between motors 3 and 1:
%        u1 = u_z - delta, u3 = u_z + delta, u2 = u4 = u_z.
%     It replaces the cascade of platform_controller.m (position loop + angle loop).
%   The measured theta is unusable in UAV_model.slx (huge noise): theta is estimated with the same complementary filter
%   as platform_controller.m (gyroscope integrated + accelerometer).
persistent u_prev theta_hat wy_prev a_prev
if isempty(u_prev)
    u_prev = 0;  theta_hat = 0;  wy_prev = 0;  a_prev = 0;    % memories, reset by "clear xz_rl_controller"
end
t = in(1); z = in(2); vz = in(3); az_prev = in(4);
x = in(5); vx = in(6); wy = in(8); ax = in(9); az = in(10);     % in(7) = measured theta, not used

% Height: Track B policy
u_z = rl_controller_platform(t, z, vz, az_prev, u_prev);
u_prev = u_z;

% Angle estimate (complementary filter)
ALPHA = 0.98;  Ts = 0.05;  G = 9.81;
theta_hat = theta_hat + Ts * (wy + wy_prev) / 2;
if az + G > 2
    theta_hat = ALPHA * theta_hat + (1 - ALPHA) * atan2(ax, az + G);
end
wy_prev = wy;

% x: RL policy. Observation = same computation as DroneEnvX.observation (drone_env_x.py)
[x_car, v_car] = car_position(t);                     % position and speed of the vehicle (sent by the vehicle)
obs = [x_car - x; v_car - vx; theta_hat / 0.2; wy; a_prev; 0];   % last: acceleration of the vehicle / g / 0.2 (0 in a straight line)
a = rl_controller_x(obs);                             % action in [-1, 1]
a_prev = a;
delta = 0.1 * a;                                      % D_MAX = 0.1, as in drone_env_x.py

u = min(max([u_z - delta; u_z; u_z + delta; u_z], -1), 1);
end
