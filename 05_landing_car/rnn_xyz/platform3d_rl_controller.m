function u = platform3d_rl_controller(in)
%PLATFORM3D_RL_CONTROLLER  Same as platform3d_controller.m, but x and y are flown by the RL policy of track_b_x.ipynb.
%   Called every Ts by EINT_PLATFORM3D_RL_eval.m.
%   in = [t; z; vz; az_prev; x; vx; theta; wy; ax; az; y; vy; phi; psi; wx; wz]   (measurements of the drone)
%   u  = [u1; u2; u3; u4]                                                        (commands of the 4 motors)
%   - height: Track B policy landing on the platform (rl_controller_platform.m) gives the common command u_z;
%   - x (pitch theta, motors 1-3): RL policy of track_b_x.ipynb (rl_controller_x.m), which replaces the cascade;
%   - y (roll phi, motors 2-4): THE SAME policy. The roll axis has the same dynamics as the pitch axis (d(wx)/dt = 100 (F2 - F4),
%     identified with explore_x.m), but phi > 0 moves the drone towards -y: the policy flies y' = -y, so the gaps of
%     position, speed and acceleration of y are given with a minus sign; its action goes to motors 2 (+) and 4 (-);
%   - yaw psi (pairs 1+3 / 2+4 turn the drone on itself in opposite directions): held at 0 with the same angle loop.
%   Mixing (signs measured with explore_x.m):  u1 = u_z - dx + dpsi,  u3 = u_z + dx + dpsi,
%                                              u2 = u_z + dy - dpsi,  u4 = u_z - dy - dpsi.
%   theta is estimated (complementary filter): its measurement is unusable in UAV_model.slx. phi and psi are fine.
persistent u_prev theta_hat wy_prev a_x_prev a_y_prev
if isempty(u_prev)
    u_prev = 0;  theta_hat = 0;  wy_prev = 0;  a_x_prev = 0;  a_y_prev = 0;   % memories, reset by "clear platform3d_rl_controller"
end
t = in(1); z = in(2); vz = in(3); az_prev = in(4); x = in(5); vx = in(6); wy = in(8); ax = in(9); az = in(10);
y = in(11); vy = in(12); phi = in(13); psi = in(14); wx = in(15); wz = in(16);      % in(7) = measured theta, not used

% Height: Track B policy (lands at z = 0, the platform)
u_z = rl_controller_platform(t, z, vz, az_prev, u_prev);
u_prev = u_z;

% Pitch estimate (complementary filter: gyroscope integrated + accelerometer)
ALPHA = 0.98;  Ts = 0.05;  G = 9.81;
theta_hat = theta_hat + Ts * (wy + wy_prev) / 2;
if az + G > 2
    theta_hat = ALPHA * theta_hat + (1 - ALPHA) * atan2(ax, az + G);
end
wy_prev = wy;

% x and y: RL policy (same observation as DroneEnvX.observation in drone_env_x.py)
[x_car, y_car, vx_car, vy_car, ax_car, ay_car] = car_path(t);
obs_x = [x_car - x; vx_car - vx; theta_hat / 0.2; wy; a_x_prev; ax_car / G / 0.2];
obs_y = [-(y_car - y); -(vy_car - vy); phi / 0.2; wx; a_y_prev; -ay_car / G / 0.2];     % y' = -y (phi > 0 -> towards -y)
a_x_prev = rl_controller_x(obs_x);              % actions in [-1, 1]
a_y_prev = rl_controller_x(obs_y);
dx = 0.1 * a_x_prev;  dy = 0.1 * a_y_prev;       % D_MAX = 0.1, as in drone_env_x.py

% Yaw: held at 0 by the same angle loop as platform3d_controller.m (not a displacement, so no RL)
sat = @(v, m) min(max(v, -m), m);
dpsi = sat(0.1 * (0 - psi) - 0.04 * wz, 0.1);

u = min(max([u_z - dx + dpsi; u_z + dy - dpsi; u_z + dx + dpsi; u_z - dy - dpsi], -1), 1);
end
