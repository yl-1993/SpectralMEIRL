% Add necessary paths for sub-directories
function configPath(alg, bAdd)

if bAdd
    cmd = @(x)addpath(x);
else
    cmd = @(x)rmpath(x);
end

cmd('MDP');
cmd('MDP/Generators');
cmd('MDP/Solvers');
cmd('MDP/Utils');
cmd('MDP/Simulator');
cmd('Params');
cmd('Utils');
cmd('Utils2');

if nargin > 0 && ~isempty(alg)
    if strcmp(alg, 'MAP_BIRL')
        cmd('BIRL');

    elseif strcmp(alg, 'MWAL')
        cmd('MWAL');
        
    elseif strcmp(alg, 'EM_IRL')
        cmd('BIRL');
        cmd('EM_IRL');
        
    elseif ~isempty(strfind(alg, 'DPM_BIRL'))
        cmd('BIRL');
        cmd('DPM_BIRL');
        
    elseif strcmp(alg, 'Ind_BIRL')
        cmd('BIRL');
        cmd('Ind_BIRL');
    
    elseif strcmp(alg, 'SPECTRAL_IRL')
        cmd('Spectral')

    elseif strcmp(alg, 'RANDOM_IRL')
        cmd('Random')
        
    end
end
