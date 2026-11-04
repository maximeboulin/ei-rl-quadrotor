%% simplify_pd.m - Simplifies the PD feedback loop of hover_test_mdl.slx (in place).
%  Removed: zoh_z, zoh_vz (sampling), to_command (1/K_AZ), saturation, delay.
%  The factor 1/K_AZ goes into the gains: P = Kp/K_AZ, D = Kd/K_AZ (same computation, fewer blocks).
%  Result:   z_ref - z -> P \
%                             (P - D) + 0.386 -> the 4 motors
%            vz        -> D /
%  Nothing else is touched (constant, scopes, logs, UAV). Run once:  matlab -batch "simplify_pd"

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(fullfile(here, '..', '..', 'Project_subject_student_folder', 'Matlab files'));

mdl = 'hover_test_mdl';
load_system(mdl);
B = @(name) [mdl '/' name];
if isempty(find_system(mdl, 'SearchDepth', 1, 'Name', 'delay'))
    fprintf('the loop is already simplified\n');
else
    % 1. remove the lines of the blocks that go away (single lines, safe to delete by port)
    delete_line(mdl, 'zoh_z/1', 'error/2');
    delete_line(mdl, 'zoh_vz/1', 'D/1');
    delete_line(mdl, 'pd/1', 'to_command/1');
    delete_line(mdl, 'to_command/1', 'command/1');
    delete_line(mdl, 'command/1', 'saturation/1');
    delete_line(mdl, 'saturation/1', 'delay/1');
    lh = get_param(B('delay'), 'LineHandles');
    delete_line(lh.Outport);                                     % delay -> the 4 motors (branched line)
    % 2. remove the blocks, then the branches left unconnected (UAV z -> zoh_z, UAV vz -> zoh_vz)
    for b = {'zoh_z', 'zoh_vz', 'to_command', 'saturation', 'delay'}
        delete_block(B(b{1}));
    end
    loose = find_system(mdl, 'FindAll', 'on', 'SearchDepth', 1, 'Type', 'line', 'DstBlockHandle', -1);
    for h = loose(:)'
        if ishandle(h), delete_line(h); end
    end
    % 3. 1/K_AZ goes into the gains
    set_param(B('P'), 'Gain', 'Kp/K_AZ');
    set_param(B('D'), 'Gain', 'Kd/K_AZ');
    % 4. new, shorter feedback loop
    add_line(mdl, 'UAV/3', 'error/2', 'autorouting', 'on');      % feedback: z
    add_line(mdl, 'UAV/6', 'D/1', 'autorouting', 'on');          % feedback: vz
    add_line(mdl, 'pd/1', 'command/1', 'autorouting', 'on');
    for j = 1:4
        add_line(mdl, 'command/1', sprintf('UAV/%d', j), 'autorouting', 'on');
    end
    % 5. check that the scopes / logs are still connected to the drone
    outs = {'z', 3; 'vz', 6; 'az', 15};
    for k = 1:size(outs, 1)
        for prefix = {'scope_', 'log_'}
            name = [prefix{1} outs{k, 1}];
            if ~isempty(find_system(mdl, 'SearchDepth', 1, 'Name', name))
                ph = get_param(B(name), 'LineHandles');
                if ph.Inport(1) == -1
                    add_line(mdl, sprintf('UAV/%d', outs{k, 2}), [name '/1'], 'autorouting', 'on');
                end
            end
        end
    end
    save_system(mdl);
    fprintf('feedback loop simplified in %s.slx\n', mdl);
end
close_system(mdl);

hover_test;                % simulate and plot
