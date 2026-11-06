function u = platform_controller(in)
%PLATFORM_CONTROLLER  Landing on the moving vehicle: same as xz_controller.m, but
%   - height: rl_controller_platform.m (lands at z = 0, the platform);
%   - x: follows the vehicle (car_position.m), with its speed as feedforward.
%   Called every Ts by the model built in EINT_PLATFORM_eval.m.
%   in = [t; z; vz; az_prev; x; vx; theta; wy; ax; az]  (measurements of the drone)
%   u  = [u1; u2; u3; u4]                                (commands of the 4 motors)
%   - height: the RL policy of Track B (rl_controller.m, unchanged) gives the common command u_z;
%   - x: cascade controller tuned in notebooks/x_axis.ipynb gives the difference delta between motors 3 and 1:
%        u1 = u_z - delta, u3 = u_z + delta, u2 = u4 = u_z   (pitch only: total thrust and yaw unchanged).
%   The measured theta is NOT used: its noise in UAV_model.slx is huge (the noise block of theta has Cov = seed_theta,
%   a random integer up to 10000, instead of 1e-4). The angle is estimated with a complementary filter:
%   gyroscope wy integrated (short term) + accelerometer atan(ax / (az + g)) (long term, removes the drift).
persistent u_prev theta_hat wy_prev
if isempty(u_prev)
    u_prev = 0;  theta_hat = 0;  wy_prev = 0;     % memories, reset by "clear xz_controller"
end
t = in(1); z = in(2); vz = in(3); az_prev = in(4);
x = in(5); vx = in(6); wy = in(8); ax = in(9); az = in(10);     % in(7) = measured theta, not used

% Height: Track B policy
u_z = rl_controller_platform(t, z, vz, az_prev, u_prev);
u_prev = u_z;

% Angle estimate (complementary filter)
ALPHA = 0.98;  Ts = 0.05;  G = 9.81;
theta_hat = theta_hat + Ts * (wy + wy_prev) / 2;               % gyroscope, trapezoidal integration
if az + G > 2                                                  % accelerometer only when the thrust is significant
    theta_hat = ALPHA * theta_hat + (1 - ALPHA) * atan2(ax, az + G);
end
wy_prev = wy;

% x: cascade controller (gains from notebooks/x_axis.ipynb)
K_THETA = 0.1;  K_OMEGA = 0.04;            % angle loop (fast)
K_X = 0.1;      K_V = 0.15;                % position loop (slow)
THETA_MAX = 0.2; DELTA_MAX = 0.1;          % bounds: angle (rad) and difference of command
[x_car, v_car] = car_position(t);          % sent by the vehicle
theta_ref = min(max(K_X * (x_car - x) + K_V * (v_car - vx), -THETA_MAX), THETA_MAX);   % follow position AND speed
delta = min(max(K_THETA * (theta_ref - theta_hat) - K_OMEGA * wy, -DELTA_MAX), DELTA_MAX);

u = min(max([u_z - delta; u_z; u_z + delta; u_z], -1), 1);
end
