% Script for run the experiments of IRL algorithms with a single expert

close all;
clear all;
LASTN = maxNumCompThreads(1);

alg.name = 'MAP_BIRL';

alg.llhType = 'BIRL'; 

alg.priorType = 'Uniform';

probName = 'gridworld';
% probName = 'highway3';

configPath(alg.name, true);

irlOpts = paramsSEIRL(alg.name, alg.llhType, alg.priorType);
problem = problemParamsSE(probName);

fprintf('****************************************\n');
fprintf('%s\n', getAlgName(alg));
fprintf('%s\n', getProblemName(problem));
fprintf('****************************************\n');

% profile clear;
% profile on;
hst = testSEIRL(problem, irlOpts);
% profile report;
% profile off;

alg2{1} = alg;
data = getResults(hst);
printResults(alg2, data, problem);

configPath(alg.name, false);
maxNumCompThreads(LASTN);
