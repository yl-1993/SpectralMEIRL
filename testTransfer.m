% Transfer information of IRL with multiple experts to new trajectory
%
function [res, hst2, hst3] = testTransfer(problem, alg)

algName = getAlgName(alg);
probName = getProblemName(problem);

outpath = 'Results_140825';
outpath = sprintf('./%s/%s', outpath, probName);
outfname = sprintf('%s/%s_hst1.mat', outpath, algName);
fprintf('Read %s\n', outfname);
load(outfname);

irlOpts = paramsMEIRL(alg.name, alg.llhType, alg.priorType);
if isfield(alg, 'nClust')
    irlOpts.nClust = problem.nExperts * alg.nClust;
end

hst2 = cell(length(problem.iters), problem.newExps);
hst3 = cell(length(problem.iters), 1);

for iter1 = problem.iters
    fprintf('## %d ##\n', iter1);
    
    % generate training data
    problem.seed = problem.initSeed + iter1;
    mdp  = generateProblem(problem, problem.initSeed, problem.discount);
    data = hst1{iter1}.data;
    if strcmp(irlOpts.alg, 'Ind_BIRL')
        sol = hst1{iter1}.sol;
    else
        sol = hst1{iter1}.sol;
        %hst = hst1{iter1}.hst;
    end
    
    % testing new trajectory
    expData = [];
    for iter2 = 1:problem.newExps
        fprintf('++ %d ++\n', iter2);
        
        % generate new data
        problem.seed = problem.initSeed + iter2;
        newdata = generateDemonstration(mdp, problem, data.weight);
        
        % transfer clustered information
        RandStream.setGlobalStream(RandStream.create('mrg32k3a', ...
            'NumStreams', 1, 'Seed', problem.seed));
        if ~isempty(strfind(irlOpts.alg, 'DPM_BIRL'))
            wL = DPM_BIRL_transfer(hst, newdata.trajSet, mdp, irlOpts);
        else
            wL = feval(sprintf('%s_transfer', irlOpts.alg), ...
                sol, newdata.trajSet, mdp, irlOpts);
        end
        
        hst2{iter1, iter2}.data = newdata;
        hst2{iter1, iter2}.wL   = wL;
        
        % evaluate solution
        if ~isempty(strfind(irlOpts.alg, 'DPM_BIRL'))
            res1 = evalSEIRL(wL.mean, newdata.weight, mdp);
            res2 = evalSEIRL(wL.max, newdata.weight, mdp);
            hst2{iter1, iter2}.meanRes = res1;
            hst2{iter1, iter2}.maxRes  = res2;
            expData.mean(iter2) = res1.valueDiff;
            expData.max(iter2)  = res2.valueDiff;
            fprintf('- Mean : [R] %f  [P] %f  [V] %f\n', ...
                res1.rewardDiff, res1.policyDiff, res1.valueDiff);
            fprintf('- MAP  : [R] %f  [P] %f  [V] %f\n\n', ...
                res2.rewardDiff, res2.policyDiff, res2.valueDiff);
        else
            res = evalSEIRL(wL, newdata.weight, mdp);
            hst2{iter1, iter2}.res = res;
            expData(iter2) = res.valueDiff;
            fprintf('-      : [R] %f  [P] %f  [V] %f\n', ...
                res.rewardDiff, res.policyDiff, res.valueDiff);
        end
    end
    
    if ~isempty(strfind(irlOpts.alg, 'DPM_BIRL'))
        hst3{iter1}.mean.mu = mean(expData.mean);
        hst3{iter1}.mean.se = sqrt(var(expData.mean)/problem.newExps);
        hst3{iter1}.max.mu = mean(expData.max);
        hst3{iter1}.max.se = sqrt(var(expData.max)/problem.newExps);
        fprintf('# Mean : %12.4f %12.4f\n', ...
            hst3{iter1}.mean.mu, hst3{iter1}.mean.se);
        fprintf('# Map  : %12.4f %12.4f\n\n', ...
            hst3{iter1}.max.mu, hst3{iter1}.max.se);
    else
        hst3{iter1}.mu = mean(expData);
        hst3{iter1}.se = sqrt(var(expData)/problem.newExps);
        fprintf('#      : %12.4f %12.4f\n\n', hst3{iter1}.mu, hst3{iter1}.se);
    end
end

if ~isempty(strfind(irlOpts.alg, 'DPM_BIRL'))
    data.mean = [];
    data.max  = [];
    for iter1 = problem.iters
        data.mean(iter1) = hst3{iter1}.mean.mu;
        data.max(iter1)  = hst3{iter1}.max.mu;
    end
    res.mean.mu = mean(data.mean);
    res.mean.se = sqrt(var(data.mean)/length(problem.iters));
    res.max.mu  = mean(data.max);
    res.max.se  = sqrt(var(data.max)/length(problem.iters));
    fprintf('+ Mean result : %12.4f %12.4f\n', res.mean.mu, res.mean.se);
    fprintf('+ MAP result  : %12.4f %12.4f\n', res.max.mu, res.max.se);
else
    data = [];
    for iter1 = problem.iters
        data(iter1) = hst3{iter1}.mu;
    end
    res.mu = mean(data);
    res.se = sqrt(var(data)/length(problem.iters));
    fprintf('+ result : %12.4f %12.4f\n', res.mu, res.se);
end

end