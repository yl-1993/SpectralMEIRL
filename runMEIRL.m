% Script for run the experiments of IRL algorithms with multiple experts

close all;
clear all;
LASTN = maxNumCompThreads(1);

probName = 'gridworld';
probName = 'highway3';

alg.name    = 'Ind_BIRL';
alg.llhType   = 'BIRL';
alg.priorType = 'Uniform';

%alg.name    = 'EM_IRL';
alg.nClust  = 1;
alg.llhType   = 'MLIRL';
alg.priorType = 'Uniform';

%alg.name    = 'DPM_BIRL_MHL';
%alg.llhType   = 'BIRL';
% alg.priorType = 'NG';
alg.priorType = 'Uniform';

%alg.name = 'SPECTRAL_IRL';

configPath(alg.name, true);
problem = problemParamsME(probName);

irlOpts = paramsMEIRL(alg.name, alg.llhType, alg.priorType);
if isfield(alg, 'nClust')
    irlOpts.nClust = problem.nExperts * alg.nClust;
end

fprintf('****************************************\n');
fprintf('%s\n', getAlgName(alg));
fprintf('%s\n', getProblemName(problem));
fprintf('****************************************\n');

hst1 = testMEIRL(problem, irlOpts);

% alg2{1} = alg;
% data = getResults(hst1);
% printResults(alg2, data, problem);

fprintf('****************************************\n');
fprintf('%s\n', getAlgName(alg));
fprintf('%s\n', getProblemName(problem));
fprintf('****************************************\n');

algName  = getAlgName(alg);
probName = getProblemName(problem);
outdir  = strcat('Results_', datestr(now, 'yymmdd'));
outpath = sprintf('./%s/%s', outdir, probName);
%if ~isdir(outpath)
%    fprintf('Mkdir %s !!!\n\n', outpath);
%    mkdir(outpath);
%end
%outfname = sprintf('%s/%s_hst1.mat', outpath, algName);
%save(outfname, 'hst1', '-v7.3');
%fprintf('Write to %s\n', outfname);

%[evd, fsc, nmi, nrCl] = get_results(outdir);
%[evd, fsc, nmi, nrCl] = get_results(outfname);

configPath(alg.name, false);
maxNumCompThreads(LASTN);
