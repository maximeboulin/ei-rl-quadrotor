%% replay_data.m - Rejoue dans Simulink les commandes u de quelques expériences du CSV
%  But : vérifier que la sortie des CSV est bien az (accélération verticale) et estimer le bruit.
%  A lancer depuis le dossier "Matlab files". Durée : environ 1 à 3 min.
%  Produit : replay_az_g0.csv et replay_az_g001.csv (une ligne par expérience rejouée)
%            + replay_rows.csv (numéros des lignes rejouées, numérotation à partir de 0 comme en Python).

clear; close all; clc;

rows = [0 1 2 500 1000 2000 2999];   % lignes du CSV à rejouer (0-based, comme en Python)
N    = 2000;                         % échantillons par expérience

%% 1) Lecture des commandes u
fprintf('Lecture de bdd_in_mat_05.csv (peut prendre ~30 s)...\n');
U = readmatrix(fullfile('..', 'bdd_in_mat_05.csv'));
U = U(rows + 1, :);                  % +1 : MATLAB numérote à partir de 1

%% 2) Construction d'un petit modèle : From Workspace -> drone (4 moteurs) -> To Workspace
random_init;                         % graines de bruit (et Ts, écrasé juste après)
Ts = 0.05;
mdl = 'replay_model';
if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile([mdl '.slx']), delete([mdl '.slx']); end
new_system(mdl);
load_system('EINT_system_simulation');

add_block('simulink/Sources/From Workspace', [mdl '/u_in'], ...
    'VariableName', 'u_ts', 'SampleTime', 'Ts', 'Interpolate', 'off', ...
    'OutputAfterFinalValue', 'Holding final value');
add_block('EINT_system_simulation/Subsystem Reference', [mdl '/UAV']);   % même drone que l'exemple
add_block('simulink/Sinks/To Workspace', [mdl '/out'], ...
    'VariableName', 'simout', 'SaveFormat', 'Structure With Time', 'SampleTime', 'Ts', 'MaxDataPoints', 'inf');
for k = 1:4
    add_line(mdl, 'u_in/1', sprintf('UAV/%d', k));   % même commande aux 4 moteurs
end
add_line(mdl, 'UAV/19', 'out/1');                    % sortie all_outputs (az = colonne 9)

% Mêmes réglages de solveur que le modèle d'exemple
cs = attachConfigSetCopy(mdl, getActiveConfigSet('EINT_system_simulation'), true);
setActiveConfigSet(mdl, cs.Name);
set_param(mdl, 'StopTime', num2str((N - 1) * Ts));
save_system(mdl);

%% 3) Simulations : sans bruit (g_noise = 0) puis avec le bruit par défaut (g_noise = 0.01)
t = (0:N-1)' * Ts;
for g = [0 0.01]
    g_noise = g;
    AZ = zeros(numel(rows), N);
    for i = 1:numel(rows)
        u_ts = [t, U(i, :)'];
        out = sim(mdl);
        az = out.simout.signals.values(:, 9);
        AZ(i, :) = az(1:N)';
        fprintf('g_noise = %-5g  ligne %4d : OK\n', g, rows(i));
    end
    fname = sprintf('replay_az_g%s.csv', strrep(num2str(g), '.', ''));
    writematrix(AZ, fname);
    fprintf('-> %s écrit\n', fname);
end
writematrix(rows, 'replay_rows.csv');
fprintf('\nTerminé. Fichiers replay_az_g0.csv, replay_az_g001.csv et replay_rows.csv dans "Matlab files".\n');
