%% plot_tr5.m - Réponse quasi continue de az à un échelon et temps de réponse à 5 %
%  Justification : Ts = tr5% / 10
%  A lancer depuis le dossier "Matlab files".

clear; close all; clc;
mdl = 'EINT_system_simulation';
load_system(mdl);

random_init;              % crée les graines de bruit
g_noise = 0;              % sans bruit
Ts = 1e-3;                % pas très fin : réponse quasi continue
col_az = 9;               % az dans simout (x y z vx vy vz ax ay az ...)

out = sim(mdl, 'StopTime', '5');
t  = out.simout.time;
az = out.simout.signals.values(:, col_az);

%% Temps de réponse à 5 % sur le premier échelon (t = 1 s)
t_step = 1;
w  = t >= t_step & t < 2.9;                 % fenêtre avant le 2e échelon (t = 3 s)
a0   = az(find(t < t_step, 1, 'last'));     % valeur initiale
aInf = az(find(t < 2.9, 1, 'last'));        % valeur finale
band = 0.05 * abs(aInf - a0);               % bande de +/- 5 %

tw = t(w);  aw = az(w);
idx_out = find(abs(aw - aInf) > band, 1, 'last');   % dernier instant hors de la bande
t5  = tw(idx_out + 1);
tr5 = t5 - t_step;
Ts_choisi = tr5 / 10;

fprintf('Valeur initiale : %.2f m/s^2, valeur finale : %.2f m/s^2\n', a0, aInf);
fprintf('Temps de réponse à 5 %% : tr5 = %.3f s\n', tr5);
fprintf('Ts = tr5/10 = %.3f s  ->  Ts retenu = 0.05 s\n', Ts_choisi);

%% Figure
figure('Name', 'Temps de réponse à 5 %', 'Color', 'w'); hold on; grid on;
plot(t, az, 'LineWidth', 1.5);
yline(aInf + band, 'k--'); yline(aInf - band, 'k--');
xline(t_step, ':', 'échelon de u', 'LabelVerticalAlignment', 'bottom');
xline(t5, 'r--', sprintf('t_{r5%%} = %.2f s', tr5), 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.2);
plot(t5, az(find(t >= t5, 1)), 'ro', 'MarkerFaceColor', 'r');
xlim([0.5 2.5]);
xlabel('t [s]'); ylabel('a_z [m/s^2]');
title(sprintf('Réponse de a_z à un échelon de u (T_s = 1 ms, sans bruit) : t_{r5%%} = %.2f s  \\Rightarrow  T_s = t_{r5%%}/10 = %.3f s', tr5, Ts_choisi));
legend('a_z', 'bande \pm5 %', '', 'Location', 'southeast');
