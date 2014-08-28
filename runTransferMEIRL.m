% Script for run the experiments of transferring the results of IRL algorithms with multiple experts

close all;
clear all;
LASTN = maxNumCompThreads(1);

probName = 'gridworld';
% probName = 'highway3';

% alg.name    = 'Ind_BIRL';
% alg.llhType   = 'BIRL';
% alg.priorType = 'Uniform';

alg.name    = 'EM_IRL';
alg.nClust  = 1;
alg.llhType   = 'MLIRL';
alg.priorType = 'Uniform';

% % alg.name    = 'DPM_BIRL_MH';
% alg.name    = 'DPM_BIRL_Gibbs';
% alg.llhType   = 'BIRL';
% alg.priorType = 'NG';

configPath(alg.name, true);
problem = problemParamsME(probName);

[res, hst2, hst3] = testTransfer(problem, alg);

fprintf('****************************************\n');
fprintf('%s\n', getAlgName(alg));
fprintf('%s\n', getProblemName(problem));
fprintf('****************************************\n');

algName = getAlgName(alg);
probName = getProblemName(problem);
outdir  = strcat('Results_', datestr(now, 'yymmdd'));
outpath = sprintf('./%s/%s', outdir, probName);
if ~isdir(outpath)
    fprintf('Mkdir %s !!!\n\n', outpath);
    mkdir(outpath);
end
outfname = sprintf('%s/%s_hst2.mat', outpath, algName);
save(outfname, 'hst2', '-v7.3');
outfname = sprintf('%s/%s_hst3.mat', outpath, algName);
save(outfname, 'hst3', '-v7.3');
outfname = sprintf('%s/%s_res.mat', outpath, algName);
save(outfname, 'res', '-v7.3');

% [hst1, hst2] = testMEIRL(problem, irlOpts);
% 
% trainResults = getResults(hst1);
% if problem.newExps > 0, transferResults = getResults(hst2); end
% 
% fprintf('****************************************\n');
% fprintf('%s\n', getAlgName(alg));
% fprintf('%s\n', getProblemName(problem));
% fprintf('****************************************\n');
% 
% outpath = 'Results_MEIRL';
% alg2{1} = alg;
% fprintf('Training\n');
% printResults(alg2, trainResults, problem, [], [], outpath, hst1);
% if problem.newExps > 0
%     fprintf('Transfer\n');
%     printResults(alg2, transferResults, problem, [], [], outpath, [], hst2);
% end

configPath(alg.name, false);
maxNumCompThreads(LASTN);
