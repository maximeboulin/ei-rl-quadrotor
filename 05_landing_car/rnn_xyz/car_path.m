function [x_car, y_car, vx_car, vy_car, ax_car, ay_car] = car_path(t)
%CAR_PATH  Position (m), speed (m/s) and acceleration (m/s^2) of the ground vehicle (its platform) at time t.
%   At rest under the drone during the take-off, then constant speed on a circle turning left (towards +y).
%   Used by platform3d_controller.m (the vehicle sends its position, speed and acceleration) and EINT_PLATFORM3D_eval.m.
V_CAR = 0.3;        % speed of the vehicle (m/s)              -> change here
R_TURN = 1.5;       % radius of the curve (m)                 -> change here (large value = almost straight line)
T_START = 1.5;      % the vehicle starts moving at this time (s), when the drone is already flying
if t < T_START - 1e-6
    x_car = 0; y_car = 0; vx_car = 0; vy_car = 0; ax_car = 0; ay_car = 0;
    return
end
a = V_CAR * (t - T_START) / R_TURN;            % angle travelled on the circle (rad)
x_car = R_TURN * sin(a);        y_car = R_TURN * (1 - cos(a));
vx_car = V_CAR * cos(a);        vy_car = V_CAR * sin(a);
ax_car = -V_CAR^2 / R_TURN * sin(a);           % centripetal acceleration, towards the centre of the circle
ay_car =  V_CAR^2 / R_TURN * cos(a);
end
