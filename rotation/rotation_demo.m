%% rotation_demo.m - The drone takes off to 0.5 m, then makes one full turn on itself.
%  Builds rotation_demo_mdl.slx from standard blocks (Gain, Sum, Integrator...), simulates it and plots.
%  Run without the GUI (lighter):  matlab -batch "rotation_demo"
%  Do NOT double-click the UAV block (it is UAV_model, the solution of the subject).
%
%  Two loops, both with the measured outputs of the drone fed back:
%  - altitude, PID:  u0 = 0.386 + (P + I - D) / K_AZ             -> the same on the 4 motors
%  - yaw, PD:        du = (P_psi - D_psi) / K_PSI                -> +du on motors 1 and 3, -du on 2 and 4
%  Motors 1 and 3 turn the drone one way, 2 and 4 the other way (motor_test): +du / -du makes it turn
%  without changing the total thrust, so the altitude is not disturbed by the turn.
%  Hypothesis: the angles of the model are in radians (one turn = 2*pi).

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(fullfile(here, '..', '..', 'Project_subject_student_folder', 'Matlab files'));

random_init;
Ts = 0.05;
g_noise = 0;
T_end = 12;

% altitude PID (same as hover_test, plus the integral term)
U_BASE = 0.386;            % hover command found with the FNN
Z_REF = 0.5;               % altitude reference [m], from t = 1 s
Kp = 4; Ki = 1.5; Kd = 4;  % gains in m/s^2 (per m, per m.s, per m/s)
K_AZ = 20;                 % az [m/s^2] per unit of command (slope of the FNN static map)

% yaw PD
Kp_psi = 4; Kd_psi = 4;    % gains in rad/s^2 (per rad, per rad/s)
R = readtable('motor_test_results.txt', 'Delimiter', '\t', 'ReadRowNames', true);
% motor test: +0.05 on ONE motor -> psi = 0.5 * acc * 4^2 after 4 s; here du acts on the 4 motors
K_PSI = 4 * 2 * mean(abs(R.d_psi)) / (16 * 0.05);    % yaw angular acceleration [rad/s^2] per unit of du
t_ref = (0:Ts:T_end)';
psi_ref = timeseries(2 * pi * min(max((t_ref - 5) / 4, 0), 1), t_ref);   % one turn between t = 5 s and 9 s
fprintf('K_PSI = %.0f rad/s^2 per unit of command\n', K_PSI);

mdl = 'rotation_demo_mdl';
fprintf('building %s.slx...\n', mdl);
build_model(mdl);
save_system(mdl);
fprintf('simulating %g s... ', T_end); tic; out = sim(mdl); fprintf('%.0f s\n', toc);

t = out.z.Time; z = squeeze(out.z.Data); psi = squeeze(out.psi.Data);
phi = squeeze(out.phi.Data); theta = squeeze(out.theta.Data); U = squeeze(out.u.Data);
if size(U, 1) == 2 && size(U, 2) ~= 2, U = U'; end
figure('Name', 'Rotation demo', 'Position', [100 100 900 750]);
subplot(4, 1, 1); plot(t, Z_REF * (t >= 1), '--', t, z, 'LineWidth', 1.3); grid on;
ylabel('z [m]'); legend('reference', 'z', 'Location', 'best'); title('Take-off, then one full turn on itself');
subplot(4, 1, 2); plot(psi_ref.Time, squeeze(psi_ref.Data), '--', t, psi, 'LineWidth', 1.3); grid on;
ylabel('\psi [rad]'); legend('reference', '\psi (yaw)', 'Location', 'best');
subplot(4, 1, 3); plot(t, phi, t, theta, 'LineWidth', 1.3); grid on;
ylabel('angle [rad]'); legend('\phi (roll)', '\theta (pitch)', 'Location', 'best');
subplot(4, 1, 4); plot(out.u.Time, U, 'LineWidth', 1.2); grid on;
ylabel('commands'); xlabel('t [s]'); legend('motors 1 and 3', 'motors 2 and 4', 'Location', 'best');
saveas(gcf, 'rotation_demo.png');
fprintf('final yaw %.2f rad (target %.2f), final z %.2f m (target %.1f)\n', psi(end), 2 * pi, z(end), Z_REF);
close_system(mdl);


%% Local function: builds the model with standard blocks
function build_model(mdl)
if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile([mdl '.slx']), delete([mdl '.slx']); end
new_system(mdl);
set_param(mdl, 'StopTime', 'T_end', 'ReturnWorkspaceOutputs', 'on');
B = @(name) [mdl '/' name];
blk = @(lib, name, pos, varargin) add_block(lib, B(name), 'Position', pos, varargin{:});
wire = @(a, b) add_line(mdl, a, b, 'autorouting', 'on');      % 'block/out port' -> 'block/in port'

% the drone
blk('simulink/Ports & Subsystems/Subsystem Reference', 'UAV', [900 40 1100 640], 'ReferencedSubsystem', 'UAV_model');
set_param(B('UAV'), 'ContentPreviewEnabled', 'off');

% altitude PID:  u0 = U_BASE + (Kp*e + Ki*int(e) - Kd*vz) / K_AZ,   e = z_ref - z
blk('simulink/Sources/Step', 'z_ref', [40 60 70 90], 'Time', '1', 'Before', '0', 'After', 'Z_REF');
blk('simulink/Math Operations/Sum', 'error_z', [120 65 140 85], 'Inputs', '+-');
blk('simulink/Math Operations/Gain', 'P_z', [200 20 260 50], 'Gain', 'Kp/K_AZ');
blk('simulink/Math Operations/Gain', 'I_z_gain', [200 80 260 110], 'Gain', 'Ki/K_AZ');
blk('simulink/Continuous/Integrator', 'I_z', [290 80 320 110]);
blk('simulink/Math Operations/Gain', 'D_z', [200 150 260 180], 'Gain', 'Kd/K_AZ');
blk('simulink/Sources/Constant', 'u_base', [350 200 410 230], 'Value', 'U_BASE');
blk('simulink/Math Operations/Sum', 'u0', [450 60 470 220], 'Inputs', '+++-');
wire('z_ref/1', 'error_z/1');
wire('UAV/3', 'error_z/2');                     % feedback: z (drone output 3)
wire('error_z/1', 'P_z/1');
wire('error_z/1', 'I_z_gain/1');
wire('I_z_gain/1', 'I_z/1');
wire('UAV/6', 'D_z/1');                         % feedback: vz (drone output 6)
wire('P_z/1', 'u0/1');
wire('I_z/1', 'u0/2');
wire('u_base/1', 'u0/3');
wire('D_z/1', 'u0/4');

% yaw PD:  du = (Kp_psi*(psi_ref - psi) - Kd_psi*wz) / K_PSI
blk('simulink/Sources/From Workspace', 'psi_ref', [20 400 90 430], 'VariableName', 'psi_ref');
blk('simulink/Math Operations/Sum', 'error_psi', [120 405 140 425], 'Inputs', '+-');
blk('simulink/Math Operations/Gain', 'P_psi', [200 395 260 435], 'Gain', 'Kp_psi/K_PSI');
blk('simulink/Math Operations/Gain', 'D_psi', [200 480 260 520], 'Gain', 'Kd_psi/K_PSI');
blk('simulink/Math Operations/Sum', 'du', [320 420 340 500], 'Inputs', '+-');
wire('psi_ref/1', 'error_psi/1');
wire('UAV/9', 'error_psi/2');                   % feedback: psi (drone output 9)
wire('error_psi/1', 'P_psi/1');
wire('UAV/12', 'D_psi/1');                      % feedback: wz (drone output 12)
wire('P_psi/1', 'du/1');
wire('D_psi/1', 'du/2');

% motors: 1 and 3 get u0 + du, 2 and 4 get u0 - du
blk('simulink/Math Operations/Sum', 'motors_1_3', [650 150 670 210], 'Inputs', '++');
blk('simulink/Math Operations/Sum', 'motors_2_4', [650 420 670 480], 'Inputs', '+-');
wire('u0/1', 'motors_1_3/1');
wire('du/1', 'motors_1_3/2');
wire('u0/1', 'motors_2_4/1');
wire('du/1', 'motors_2_4/2');
wire('motors_1_3/1', 'UAV/1');                  % drone inputs u1 and u3
wire('motors_1_3/1', 'UAV/3');
wire('motors_2_4/1', 'UAV/2');                  % drone inputs u2 and u4
wire('motors_2_4/1', 'UAV/4');

% logging (for the figure) and scopes (to look at the signals in Simulink)
blk('simulink/Signal Routing/Mux', 'mux_u', [720 300 725 360], 'Inputs', '2');
wire('motors_1_3/1', 'mux_u/1');
wire('motors_2_4/1', 'mux_u/2');
blk('simulink/Sinks/To Workspace', 'log_u', [760 315 840 345], 'VariableName', 'u', 'SaveFormat', 'Timeseries');
wire('mux_u/1', 'log_u/1');
logged = {'z', 3; 'phi', 7; 'theta', 8; 'psi', 9};
for k = 1:size(logged, 1)
    y = 60 + 120 * (k - 1);
    blk('simulink/Sinks/To Workspace', ['log_' logged{k, 1}], [1250 y 1330 y + 30], ...
        'VariableName', logged{k, 1}, 'SaveFormat', 'Timeseries');
    blk('simulink/Sinks/Scope', ['scope_' logged{k, 1}], [1250 y + 45 1280 y + 75]);
    wire(sprintf('UAV/%d', logged{k, 2}), ['log_' logged{k, 1} '/1']);
    wire(sprintf('UAV/%d', logged{k, 2}), ['scope_' logged{k, 1} '/1']);
end
end
