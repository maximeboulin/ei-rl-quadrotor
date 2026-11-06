function u = platform3d_controller(in)
%PLATFORM3D_CONTROLLER  Landing on the vehicle that follows a curve (x and y), called every Ts by EINT_PLATFORM3D_eval.m.
%   in = [t; z; vz; az_prev; x; vx; theta; wy; ax; az; y; vy; phi; psi; wx; wz]   (measurements of the drone)
%   u  = [u1; u2; u3; u4]                                                        (commands of the 4 motors)
%   - height: Track B policy landing on the platform (rl_controller_platform.m) gives the common command u_z;
%   - x (pitch theta, motors 1-3) and y (roll phi, motors 2-4): same cascade as in x_axis.ipynb, following the
%     position, the speed and the acceleration of the vehicle (car_path.m);
%   - yaw psi (pairs 1+3 / 2+4 turn the drone on itself in opposite directions): held at 0 with the same angle loop.
%   Mixing (signs measured with explore_x.m):  u1 = u_z - dx + dpsi,  u3 = u_z + dx + dpsi,
%                                              u2 = u_z + dy - dpsi,  u4 = u_z - dy - dpsi.
%   theta is estimated (complementary filter): its measurement is unusable in UAV_model.slx. phi and psi are fine.
persistent u_prev theta_hat wy_prev
if isempty(u_prev)
    u_prev = 0;  theta_hat = 0;  wy_prev = 0;     % memories, reset by "clear platform3d_controller"
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

% Gains (x_axis.ipynb), the same for the 3 axes: the identified dynamics have the same gain (see explore_x)
K_ANGLE = 0.1;  K_RATE = 0.04;             % angle loops (fast)
K_POS = 0.1;    K_SPEED = 0.15;            % position loops (slow)
ANGLE_MAX = 0.2; D_MAX = 0.1;              % bounds: angle (rad) and difference of command
sat = @(v, m) min(max(v, -m), m);

% Position loops: angle to take = feedforward of the acceleration of the vehicle + correction of position and speed
[x_car, y_car, vx_car, vy_car, ax_car, ay_car] = car_path(t);
theta_ref = sat(ax_car / G + K_POS * (x_car - x) + K_SPEED * (vx_car - vx), ANGLE_MAX);     % theta > 0 -> towards +x
phi_ref  = -sat(ay_car / G + K_POS * (y_car - y) + K_SPEED * (vy_car - vy), ANGLE_MAX);     % phi < 0   -> towards +y

% Angle loops
dx   = sat(K_ANGLE * (theta_ref - theta_hat) - K_RATE * wy, D_MAX);
dy   = sat(K_ANGLE * (phi_ref - phi) - K_RATE * wx, D_MAX);
dpsi = sat(K_ANGLE * (0 - psi) - K_RATE * wz, D_MAX);

u = min(max([u_z - dx + dpsi; u_z + dy - dpsi; u_z + dx + dpsi; u_z - dy - dpsi], -1), 1);
end
