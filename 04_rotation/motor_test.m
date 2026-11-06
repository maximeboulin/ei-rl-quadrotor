%% motor_test.m - Which input of the drone is which motor?
%  The drone model (UAV_model, used through a Subsystem Reference, never opened) has 4 inputs.
%  Test: all 4 inputs at the hover command, then at t = 2 s a small step on ONE input only.
%  - the drone ROTATES on itself (psi, wz change)  -> this input changes the yaw
%  - the drone TILTS (phi or theta change)         -> tells where the motor is
%  Run without the GUI (lighter):  matlab -batch "motor_test"
%  Writes motor_test.png and motor_test_results.txt in this folder.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(fullfile(here, '..', 'matlab'));  % UAV_model, quadcopter_model.p

random_init;                % random seeds used inside the drone model
Ts = 0.05;                  % sampling period (as in EINT_RL_eval.m)
g_noise = 0;                % no noise for these tests
u_hover = 0.386;            % hover command found with the FNN (az = 0 when the 4 inputs are equal)
du = 0.05;                  % small step on one input
t_step = 2;
T_end = 6;

fprintf('MATLAB started, building the test model...\n');
mdl = 'motor_test_mdl';
build_model(mdl);
fprintf('model built\n');

t = (0:Ts:T_end)';
names = {'z', 'phi', 'theta', 'psi', 'wz'};
res = zeros(4, numel(names));
figure('Name', 'Motor test', 'Position', [100 100 1100 700]);
for i = 1:4
    U = u_hover * ones(numel(t), 4);
    U(t >= t_step, i) = u_hover + du;
    for j = 1:4
        assignin('base', sprintf('u%d', j), timeseries(U(:, j), t));
    end
    fprintf('simulation %d/4 (step on input %d)... ', i, i); tic;
    out = sim(mdl);
    fprintf('done in %.0f s\n', toc);
    for k = 1:numel(names)
        s = out.get(names{k});
        y = squeeze(s.Data);
        y0 = interp1(s.Time, y, t_step - Ts);          % value just before the step
        res(i, k) = y(end) - y0;                       % change caused by the step
        subplot(numel(names), 1, k); hold on; grid on;
        plot(s.Time, y, 'LineWidth', 1.2); ylabel(names{k});
    end
end
subplot(numel(names), 1, 1); legend('input 1', 'input 2', 'input 3', 'input 4', 'Location', 'best');
title(sprintf('Step of +%.2f on one input at t = %g s (others at %.3f)', du, t_step, u_hover));
xlabel('t [s]');
saveas(gcf, 'motor_test.png');

T = array2table(res, 'VariableNames', strcat('d_', names), ...
                'RowNames', {'input 1', 'input 2', 'input 3', 'input 4'});
disp(T);
writetable(T, 'motor_test_results.txt', 'WriteRowNames', true, 'Delimiter', '\t');
save_system(mdl);          % motor_test_mdl.slx, can be opened to show the test
close_system(mdl);


%% Local function: builds the test model (4 From Workspace -> drone -> To Workspace)
function build_model(mdl)
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
set_param(mdl, 'StopTime', 'T_end', 'ReturnWorkspaceOutputs', 'on');
add_block('simulink/Ports & Subsystems/Subsystem Reference', [mdl '/UAV'], ...
          'ReferencedSubsystem', 'UAV_model', 'Position', [300 40 500 640]);
set_param([mdl '/UAV'], 'ContentPreviewEnabled', 'off');   % never show the inside of UAV_model
for j = 1:4
    b = sprintf('u%d', j);
    add_block('simulink/Sources/From Workspace', [mdl '/' b], 'VariableName', b, ...
              'Position', [80 60 + 80 * (j - 1) 180 90 + 80 * (j - 1)]);
    add_line(mdl, [b '/1'], sprintf('UAV/%d', j));
end
% drone outputs (as wired in EINT_system_simulation.slx): 3 z, 7 phi, 8 theta, 9 psi, 12 wz
logged = {'z', 3; 'phi', 7; 'theta', 8; 'psi', 9; 'wz', 12};
for k = 1:size(logged, 1)
    b = ['log_' logged{k, 1}];
    add_block('simulink/Sinks/To Workspace', [mdl '/' b], 'VariableName', logged{k, 1}, ...
              'SaveFormat', 'Timeseries', 'Position', [650 40 + 60 * (k - 1) 750 70 + 60 * (k - 1)]);
    add_line(mdl, sprintf('UAV/%d', logged{k, 2}), [b '/1']);
end
for p = setdiff(1:19, [logged{:, 2}])      % unused outputs
    b = sprintf('term%d', p);
    add_block('simulink/Sinks/Terminator', [mdl '/' b], 'Position', [900 20 * p 920 20 * p + 15]);
    add_line(mdl, sprintf('UAV/%d', p), [b '/1']);
end
end
