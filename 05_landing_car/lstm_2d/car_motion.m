function [x_car, v_car, a_car] = car_motion(t, P)
%CAR_MOTION  Position, speed and acceleration of the car (and of its platform) at time t.
%   Scenario (same as car_motion in src/env_2d.py): the car already drives at constant speed P.car_v,
%   P.car_x0 metres ahead of the drone at t = 0.
x_car = P.car_x0 + P.car_v * t;
v_car = P.car_v;
a_car = 0;
end
