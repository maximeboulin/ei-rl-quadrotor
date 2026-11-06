%% check_Ts.m - Justification de la période d'échantillonnage Ts
%  A lancer depuis n'importe où (le script se place dans 01_data et ajoute matlab/ au chemin).
%  Compare la réponse de az à un échelon de commande pour plusieurs Ts.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(fullfile(here, '..', 'matlab'));   % provided Simulink models
mdl = 'EINT_system_simulation';
load_system(mdl);

random_init;            % crée Ts, g_noise et les graines (seed_*) dans le workspace
col_az = 9;             % az = 9e colonne de simout (ordre du Mux : x y z vx vy vz ax ay az phi theta psi wx wy wz w_dot_x w_dot_y w_dot_z)

%% 1) Réponse "quasi continue", sans bruit : mesure de la constante de temps
g_noise = 0;
Ts = 1e-3;
out = sim(mdl, 'StopTime', '5');
t  = out.simout.time;
Y  = out.simout.signals.values;
fprintf('all_outputs : %d signaux, %d échantillons\n', size(Y, 2), size(Y, 1));
az = Y(:, col_az);

% Echelon de commande à t = 1 s : on cherche le temps pour atteindre 63 % de la variation
i0 = find(t >= 1, 1);  i1 = find(t >= 2.9, 1);     % fenêtre [1 s ; 2.9 s] (2e échelon à 3 s)
a0 = az(i0 - 1);  aInf = az(i1);
i63 = i0 - 1 + find((az(i0:i1) - a0) / (aInf - a0) >= 0.63, 1);
tau = t(i63) - 1;
fprintf('az : %.2f -> %.2f m/s^2, constante de temps tau ~ %.3f s\n', a0, aInf, tau);
fprintf('temps de réponse à 5 %% ~ 3*tau = %.2f s, bande passante ~ 1/(2*pi*tau) = %.2f Hz\n', 3*tau, 1/(2*pi*tau));
fprintf('Règle usuelle tau/10 <= Ts <= tau/4  ->  %.3f s <= Ts <= %.3f s\n', tau/10, tau/4);

%% 2) Même expérience pour plusieurs Ts (sans bruit)
Ts_list = [1e-3 0.01 0.05 0.1 0.2];
figure('Name', 'Effet de Ts sur la réponse de az'); hold on; grid on;
for Ts = Ts_list
    out = sim(mdl, 'StopTime', '5');
    stairs(out.simout.time, out.simout.signals.values(:, col_az), 'LineWidth', 1.2);
end
xlabel('t [s]'); ylabel('a_z [m/s^2]');
legend(compose('Ts = %g s', Ts_list), 'Location', 'best');
title(sprintf('Réponse de a_z (sans bruit), \\tau \\approx %.2f s', tau));

%% 3) Effet de Ts sur le bruit (g_noise = 0.01, valeur par défaut)
%  Les blocs "Band-Limited White Noise" ont une puissance fixe : la variance du bruit
%  échantillonné vaut Cov/Ts, donc elle augmente quand Ts diminue.
g_noise = 0.01;
fprintf('\nEcart-type du bruit sur az (régime établi 4,0-4,9 s), g_noise = %g :\n', g_noise);
for Ts = [1e-4 1e-3 0.01 0.05 0.1]
    out = sim(mdl, 'StopTime', '5');
    t = out.simout.time;  az = out.simout.signals.values(:, col_az);
    w = t >= 4 & t <= 4.9;
    fprintf('  Ts = %-7g : std = %.4f m/s^2  (%d échantillons simulés)\n', Ts, std(detrend(az(w))), numel(t));
end
