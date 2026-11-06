function [x_car, v_car] = car_position(t)
%CAR_POSITION  Position (m) and speed (m/s) of the ground vehicle (its platform) at time t.
%   At rest under the drone during the take-off, then constant speed along x.
%   Used by platform_controller.m (the vehicle sends its position and speed to the drone) and by EINT_PLATFORM_eval.m.
V_CAR = 0.3;        % speed of the vehicle (m/s) -> change here (tested up to 1 m/s)
T_START = 1.5;      % the vehicle starts moving at this time (s), when the drone is already flying
if t < T_START - 1e-6
    x_car = 0;  v_car = 0;
else
    x_car = V_CAR * (t - T_START);  v_car = V_CAR;
end
end
