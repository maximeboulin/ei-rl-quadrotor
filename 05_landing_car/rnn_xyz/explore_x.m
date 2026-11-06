%% explore_x.m - Which motor makes the drone tilt / move in which direction?
%  Step 1 of the x-axis work (advanced task "land at a different position").
%  The 4 motors always received the SAME command until now (pure vertical flight). Here, the drone hovers, then ONE
%  motor gets a little more thrust for 0.5 s. We record all the outputs to see which angle and which position move.
%  Run from anywhere (writes explore_x.csv in this folder). Duration: about 1 min. No noise.
%  Writes explore_x.csv: one block of rows per test (test number, time, then the 18 outputs of UAV_model).

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(fullfile(here, '..', '..', 'matlab'));   % provided Simulink models

Ts      = 0.05;      % sampling period
T_END   = 4;         % duration of each test (s)
U_HOVER = 0.389;     % hover command (az = 0), found in notebook 01
DELTA   = 0.02;      % extra command on one motor (small, to stay close to hover)
T_ON    = [1.0 1.5]; % the extra command is applied between 1.0 s and 1.5 s

%% 1) Small model: From Workspace (4 commands) -> drone -> To Workspace
random_init;                   % creates the noise seeds (and Ts, overwritten just after)
Ts = 0.05;
g_noise = 0;                   % no noise: we want to see clearly what moves
mdl = 'explore_model';
if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile([mdl '.slx']), delete([mdl '.slx']); end
new_system(mdl);
load_system('EINT_system_simulation');
add_block('simulink/Sources/From Workspace', [mdl '/u_in'], ...
    'VariableName', 'u_ts', 'SampleTime', 'Ts', 'Interpolate', 'off', ...
    'OutputAfterFinalValue', 'Holding final value');               % outputs the vector [u1 u2 u3 u4]
add_block('simulink/Signal Routing/Demux', [mdl '/demux'], 'Outputs', '4');   % splits it into 4 signals
add_block('EINT_system_simulation/Subsystem Reference', [mdl '/UAV']);        % the same drone as in the example
add_block('simulink/Sinks/To Workspace', [mdl '/out'], ...
    'VariableName', 'simout', 'SaveFormat', 'Structure With Time', 'SampleTime', 'Ts', 'MaxDataPoints', 'inf');
add_line(mdl, 'u_in/1', 'demux/1');
for k = 1:4
    add_line(mdl, sprintf('demux/%d', k), sprintf('UAV/%d', k));   % motor k receives command k
end
add_line(mdl, 'UAV/19', 'out/1');                                  % all_outputs (18 signals)
cs = attachConfigSetCopy(mdl, getActiveConfigSet('EINT_system_simulation'), true);
setActiveConfigSet(mdl, cs.Name);
set_param(mdl, 'StopTime', num2str(T_END));
save_system(mdl);

%% 2) Tests: test 0 = all motors at hover (reference), tests 1 to 4 = motor k gets + DELTA between 1.0 and 1.5 s
t = (0:Ts:T_END)';
names = {'x','y','z','vx','vy','vz','ax','ay','az','phi','theta','psi','wx','wy','wz','w_dot_x','w_dot_y','w_dot_z'};
R = [];
for test = 0:4
    U = U_HOVER * ones(numel(t), 4);
    if test > 0
        on = t >= T_ON(1) - 1e-9 & t < T_ON(2) - 1e-9;
        U(on, test) = U_HOVER + DELTA;
    end
    u_ts = [t, U];
    out = sim(mdl);
    Y = out.simout.signals.values;                                 % columns in the order of "names"
    R = [R; test * ones(size(Y, 1), 1), out.simout.time, Y];       %#ok<AGROW>
    fprintf('test %d done\n', test);
end
writematrix(R, 'explore_x.csv');

%% 3) Quick look: angles and horizontal positions for each test
figure('Name', 'explore_x');
plots = {'phi', 'theta', 'psi', 'x', 'y', 'z'};
for p = 1:numel(plots)
    subplot(2, 3, p); hold on; grid on;
    c = find(strcmp(names, plots{p}));
    for test = 0:4
        rows = R(:, 1) == test;
        plot(R(rows, 2), R(rows, 2 + c), 'LineWidth', 1.2);
    end
    title(plots{p}); xlabel('t [s]');
end
legend({'hover', 'motor 1 +', 'motor 2 +', 'motor 3 +', 'motor 4 +'}, 'Location', 'best');
fprintf('\nexplore_x.csv written in this folder (read by x_axis.ipynb).\n');
