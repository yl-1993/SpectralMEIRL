% test IRL algorithms for a single expert
%
function hst = testSPECCIRL(problem, irlOpts)

hst = cell(length(problem.iters), 1);
t1 = clock;
for iter = problem.iters
    fprintf('## %d ##\n', iter);
    
    % generate data
    problem.seed = problem.initSeed + iter;
    mdp  = generateProblem(problem, problem.seed, problem.discount);
    data = generateDemonstration(mdp, problem);
    
    % IRL
    RandStream.setGlobalStream(RandStream.create('mrg32k3a', ...
        'NumStreams', 1, 'Seed', problem.seed));
    tic;
    fprintf('alg:%s\n', irlOpts.alg);
    wL = feval(irlOpts.alg, data.trajSet, mdp, irlOpts);
    elapsedTime = toc;
    
    % evaluate solution
    results = evalSEIRL(wL, data.weight, mdp);
    fprintf('- SEIRL results: [R] %f  [P] %f  [V] %f : %.2f sec\n\n', ...
        results.rewardDiff, results.policyDiff, results.valueDiff, elapsedTime);
    hst{iter}      = results;
    hst{iter}.wL   = wL;
    hst{iter}.data = data;
    hst{iter}.mdp  = mdp;
    wL - data.weight
end
disp(['total time:',num2str(etime(clock,t1))]);

end