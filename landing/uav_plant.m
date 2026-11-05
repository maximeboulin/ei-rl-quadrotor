function [s, az, dxdt] = uav_plant(s, u, Ts)
%UAV_PLANT  One sampling period Ts of the drone, without Simulink.
%   Same blocks as UAV_model.slx (sensor noise excluded):
%   each motor command u(i) -> motor lag 1/(0.2 s + 1) -> tanh -> gain 10 -> quadcopter_model(x, u)
%
%   s.x : 12 states [x y z vx vy vz phi theta psi wx wy wz]
%   s.m : 4 motor lag states
%   u   : 4 motor commands (u1..u4, same order as the inports of UAV_model.slx)
%   az  : vertical acceleration at the end of the period (like the 'az' output of UAV_model.slx)
%   dxdt: full state derivative at the end of the period (dxdt(4) = ax, dxdt(6) = az)
%
%   Create the initial state with: s = struct('x', zeros(12, 1), 'm', zeros(4, 1));
n  = 50;                     % Euler sub-steps inside one sampling period
dt = Ts / n;
u  = u(:);
for i = 1:n
    s.m  = s.m + dt * (u - s.m) / 0.2;              % motor lag 1/(0.2 s + 1)
    dxdt = quadcopter_model(s.x, 10 * tanh(s.m));   % protected physical model (matlab/quadcopter_model.p)
    s.x  = s.x + dt * dxdt;
end
az = dxdt(6);
end
