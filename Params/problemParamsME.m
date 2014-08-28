function problem = problemParamsME(name)

problem.name      = name;
problem.iters     = 1:1;
problem.discount  = 0.9;
problem.nExps     = length(problem.iters);
problem.nExperts  = 3;    % # of experts
problem.nTrajs    = 5;    % # of trajectories
problem.nSteps    = 40;   % # of steps in each trajectory
problem.initSeed  = 1;    % initial random seed

if strcmp(name, 'gridworld')
    problem.gridSize  = 8;
    problem.blockSize = 2;
    problem.noise     = 0.4;
    problem.filename = sprintf('%s_%dx%d', name, ...
        problem.gridSize, problem.blockSize);
    
elseif strcmp(name, 'highway3')
    problem.filename = name;
    
end

problem.newExpertProb = 0.0;
problem.newExps       = 20;
problem.newTrajSteps  = problem.nSteps;
