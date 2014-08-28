function algname = getAlgName(alg)

prior = alg.priorType(1:2);
if strcmp(alg.name, 'MAP_BIRL') || strcmp(alg.name, 'Ind_BIRL')
    algname = alg.llhType;
    algname = sprintf('%s(%s)', algname, prior);
    
elseif strcmp(alg.name, 'EM_IRL')
    algname = strcat('EM_', alg.llhType);
    if isfield(alg, 'nClust') && ~isempty(alg.nClust)
        algname = sprintf('%s(%s,%d)', algname, prior, alg.nClust);
    else
        algname = sprintf('%s(%s)', algname, prior);
    end
    
elseif strcmp(alg.name, 'DPM_BIRL_MH')
    algname = strcat('DPM_', alg.llhType);
    algname = sprintf('%s_MH(%s)', algname, prior);
    
elseif strcmp(alg.name, 'DPM_BIRL_MH2')
    algname = strcat('DPM_', alg.llhType);
    algname = sprintf('%s_MH2(%s)', algname, prior);
    
elseif strcmp(alg.name, 'DPM_BIRL_MH3')
    algname = strcat('DPM_', alg.llhType);
    algname = sprintf('%s_MH3(%s)', algname, prior);
    
elseif strcmp(alg.name, 'DPM_BIRL_MH4')
    algname = strcat('DPM_', alg.llhType);
    algname = sprintf('%s_MH4(%s)', algname, prior);
    
elseif strcmp(alg.name, 'DPM_BIRL_MHL')
    algname = strcat('DPM_', alg.llhType);
    algname = sprintf('%s_MHL(%s)', algname, prior);
    
elseif strcmp(alg.name, 'DPM_BIRL_Gibbs')
    algname = strcat('DPM_', alg.llhType);
    algname = sprintf('%s_Gibbs(%s)', algname, prior);
    
elseif strcmp(alg.name, 'SPECTRAL_IRL')
    algname = strcat('SEPCC_', alg.llhType);

elseif strcmp(alg.name, 'RANDOM_IRL')
    algname = strcat('RANDOM_', alg.llhType);

end
