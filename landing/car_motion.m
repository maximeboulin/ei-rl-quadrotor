function [x_car, v_car, a_car] = car_motion(t, P)
%CAR_MOTION  Position, speed and acceleration of the car (and of its platform) at time t.
%   Scenario: at rest until P.car_t0, then constant acceleration P.car_a up to P.car_v, then constant speed.
if t < P.car_t0
    x_car = 0;  v_car = 0;  a_car = 0;
    return
end
t1 = P.car_t0 + P.car_v / P.car_a;          % end of the acceleration phase
if t < t1
    tau = t - P.car_t0;
    x_car = 0.5 * P.car_a * tau^2;  v_car = P.car_a * tau;  a_car = P.car_a;
else
    x_car = 0.5 * P.car_a * (t1 - P.car_t0)^2 + P.car_v * (t - t1);  v_car = P.car_v;  a_car = 0;
end
end
