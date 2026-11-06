function export_matlab_figures
%EXPORT_MATLAB_FIGURES  Runs the MATLAB tests of the project and saves their figures for the slides.
%   Run from anywhere:  export_matlab_figures   (in slides/). Takes a few minutes (Simulink runs).
%   Figures are saved in slides/figures/ with a light theme and a white background.
%   Each test runs in its own workspace, so the "clear" at the top of the scripts does not stop the loop.
root = fileparts(fileparts(mfilename('fullpath')));
out = fullfile(root, 'slides', 'figures');
if ~isfolder(out), mkdir(out); end
jobs = {
    fullfile(root, '03_altitude_rl', 'rnn_robustness', 'EINT_RL_eval.m'), 'rnn_rl_eval'
    fullfile(root, '03_altitude_rl', 'lstm_sac', 'EINT_compare.m'),       'lstm_rl_vs_pid'
    fullfile(root, '03_altitude_rl', 'lstm_sac', 'EINT_robustness.m'),    'lstm_robustness'
    fullfile(root, '04_rotation', 'rotation_demo.m'),                     'rotation'
    fullfile(root, '05_landing_car', 'lstm_2d', 'eval_landing.m'),        'landing_2d'
    };
p0 = path;
for j = 1:size(jobs, 1)
    fprintf('\n===== %s\n', jobs{j, 1});
    close all force
    try
        run_one(jobs{j, 1}, root);
        figs = flip(findall(0, 'Type', 'figure'));
        for i = 1:numel(figs)
            f = figs(i);
            try, theme(f, 'light'); catch, end          % R2025a+: dark desktop theme -> light figure
            set(f, 'Color', 'w');
            name = fullfile(out, sprintf('%s_%d.png', jobs{j, 2}, i));
            exportgraphics(f, name, 'Resolution', 200, 'BackgroundColor', 'white');
            fprintf('saved %s\n', name);
        end
    catch err
        fprintf(2, 'FAILED (%s): %s\n', jobs{j, 2}, err.message);
    end
    close all force
    path(p0);                                            % each test adds its own folders
    bdclose all
end
cd(fullfile(root, 'slides'));
fprintf('\nDone: figures in %s\n', out);
end

function run_one(script, root)
addpath(fullfile(root, 'matlab'));                       % random_init, UAV_model, quadcopter_model
run(script);
end
