function [evd, fsc, nmi, nrCl] = get_results(outpath)

configPath([], true);

probName = 'gridworld';

k = 1;
% alg{k}.name    = 'Ind_BIRL';
% alg{k}.llhType   = 'BIRL';
% alg{k}.priorType = 'Uniform';
% alg{k}.lw = 1;
% alg{k}.lt = ':';
% k = k + 1;
% 
% alg{k}.name    = 'EM_IRL';
% alg{k}.nClust  = 1;
% alg{k}.llhType   = 'MLIRL';
% alg{k}.priorType = 'Uniform';
% alg{k}.lw = 2;
% alg{k}.lt = '--';
% k = k + 1;
% 
% alg{k}.name    = 'EM_IRL';
% alg{k}.nClust  = 2;
% alg{k}.llhType   = 'MLIRL';
% alg{k}.priorType = 'Uniform';
% alg{k}.lw = 2;
% alg{k}.lt = '--';
% k = k + 1;
% 
% alg{k}.name    = 'EM_IRL';
% alg{k}.nClust  = 3;
% alg{k}.llhType   = 'MLIRL';
% alg{k}.priorType = 'Uniform';
% alg{k}.lw = 2;
% alg{k}.lt = '--';
% k = k + 1;
%
% alg{k}.name    = 'DPM_BIRL_MH';
% alg{k}.llhType   = 'BIRL';
% alg{k}.priorType = 'Un';
% alg{k}.lw = 2;
% alg{k}.lt = '--';
% k = k + 1;

alg{k}.name    = 'DPM_BIRL_MHL';
alg{k}.llhType   = 'BIRL';
alg{k}.priorType = 'Un';
alg{k}.lw = 2;
alg{k}.lt = '-';
k = k + 1;

problem  = problemParamsME(probName);
nAlgs    = length(alg);
nExps    = problem.nExps;

evd.mu  = nan(nAlgs, 1);
evd.se  = nan(nAlgs, 1);
fsc.mu  = nan(nAlgs, 1);
fsc.se  = nan(nAlgs, 1);
nmi.mu  = nan(nAlgs, 1);
nmi.se  = nan(nAlgs, 1);
nrCl.mu = nan(nAlgs, 1);
nrCl.se = nan(nAlgs, 1);
for k = 1:nAlgs
    algName = getAlgName(alg{k});
    
    probName = getProblemName(problem);
    outpath2 = sprintf('./%s/%s', outpath, probName);
    outfname = sprintf('%s/%s_hst1.mat', outpath2, algName);
    fprintf('Read %s\n', outfname);
    load(outfname);
    
    expData.evd  = nan(nExps, 1);
    expData.fsc  = nan(nExps, 1);
    expData.nmi  = nan(nExps, 1);
    expData.nrCl = nan(nExps, 1);
    for i = 1:nExps
        fprintf('Compute EVD for exp. %d\n', i);
        problem.seed = problem.initSeed + i;
        mdp = generateProblem(problem, problem.initSeed, problem.discount);
        
        input  = hst1{i}.data;
        nT = size(input.trajSet, 1);
        vE = zeros(nT, 1);
        for m = 1:nT
            kE = input.trajId(m);
            wE = input.weight(:, kE);
            pE = input.policy(:, kE);
            [VE, HE] = evaluate(pE, mdp, wE);
            vE(m) = full(wE'*HE'*mdp.start);
        end
        
        sol  = hst1{i}.sol;
        cl.b = sol.belongTo;
        cl.w = sol.weight;
        expData.evd(i)  = calEVD(input, vE, cl, mdp);
        expData.fsc(i)  = calFscore(input, sol.belongTo);
        expData.nmi(i)  = calNMI(input, sol.belongTo);
        expData.nrCl(i) = countClust(sol.belongTo);
    end
    evd.mu(k)  = mean(expData.evd);
    evd.se(k)  = sqrt(var(expData.evd)./nExps);
    fsc.mu(k)  = mean(expData.fsc);
    fsc.se(k)  = sqrt(var(expData.fsc)./nExps);
    nmi.mu(k)  = mean(expData.nmi);
    nmi.se(k)  = sqrt(var(expData.nmi)./nExps);
    nrCl.mu(k) = mean(expData.nrCl);
    nrCl.se(k) = sqrt(var(expData.nrCl)./nExps);
end


for k = 1:nAlgs
    algName = getAlgName(alg{k});
    fprintf('== %s\n', algName);
    fprintf('EVD      : %8.4f %8.4f\n', evd.mu(k), evd.se(k));
    fprintf('F-score  : %8.4f %8.4f\n', fsc.mu(k), fsc.se(k));
    fprintf('NMI      : %8.4f %8.4f\n', nmi.mu(k), nmi.se(k));
    fprintf('Clusters : %8.4f %8.4f\n', nrCl.mu(k), nrCl.se(k));
    fprintf('\n');
end

fprintf('== EVD\n');
for k = 1:nAlgs
    algName = getAlgName(alg{k});
    fprintf('%s \t', algName);
    fprintf('%8.4f %8.4f\n', evd.mu(k), evd.se(k));
end
fprintf('\n');

fprintf('== F-score\n');
for k = 1:nAlgs
    algName = getAlgName(alg{k});
    fprintf('%s \t', algName);
    fprintf('%8.4f %8.4f\n', fsc.mu(k), fsc.se(k));
end
fprintf('\n');

fprintf('== NMI\n');
for k = 1:nAlgs
    algName = getAlgName(alg{k});
    fprintf('%s \t', algName);
    fprintf('%8.4f %8.4f\n', nmi.mu(k), nmi.se(k));
end
fprintf('\n');

fprintf('== # of clusters\n');
for k = 1:nAlgs
    algName = getAlgName(alg{k});
    fprintf('%s \t', algName);
    fprintf('%8.4f %8.4f\n', nrCl.mu(k), nrCl.se(k));
end

% fprintf('Plot results\n');
% 
% scrsz = get(0,'ScreenSize');
% fig   = figure('Position',[1 scrsz(4)/2 1600 400]);
% bErr  = true;
% xstr  = '# of trajectories per expert';
% 
% makeSubplot(1, 4, 1);
% plotResults(trajs, evd.mu, evd.se, alg, xstr, 'EVD', probName, bErr);
% 
% makeSubplot(1, 4, 2);
% plotResults(trajs, fsc.mu, fsc.se, alg, xstr, 'F-score', [], bErr);
% 
% makeSubplot(1, 4, 3);
% plotResults(trajs, nmi.mu, nmi.se, alg, xstr, 'NMI', [], bErr);
% 
% makeSubplot(1, 4, 4);
% plotResults(trajs, nrCl.mu, nrCl.se, alg, xstr, '# of clusters', [], bErr, true);
% 
% inpath = 'Results';
% infname = sprintf('./%s/%s.fig', inpath, probName);
% saveas(fig, infname);

configPath([], false);

end
