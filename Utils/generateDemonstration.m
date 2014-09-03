% Generate demonstration
function data = generateDemonstration(mdp, problem, weights)

data  = struct('weight', [], 'policy', [], 'trajId', [], 'trajSet', []);
nF    = mdp.nFeatures;

if nargin == 3 && ~isempty(weights)
    nW = size(weights, 2);
    RandStream.setGlobalStream(RandStream.create('mrg32k3a', ...
        'NumStreams', 1, 'Seed', problem.seed));
    if rand() < problem.newExpertProb
        fprintf('- Generate new weight\n');
        w = sampleWeight(problem.name, nF, nW + 1);
    else
        ix = randi(nW);
        fprintf('- Select %d-th weight\n', ix);
        w  = weights(:, ix);
    end
    mdp = convertW2R(w, mdp);
    problem.nTrajs  = 1;
    problem.nSteps  = problem.newTrajSteps;
    [trajs, policy] = generateTrajectory(mdp, problem);
    data.weight  = w;
    data.policy  = policy;
    data.trajId  = 1;
    data.trajSet = trajs;
else
    for i = 1:problem.nExperts
        % generate weight and reward
        fprintf('- Generate %d-th weight\n', i);
        RandStream.setGlobalStream(RandStream.create('mrg32k3a', ...
            'NumStreams', 1, 'Seed', problem.seed + i));
        w   = sampleWeight(problem.name, nF, i);
        mdp = convertW2R(w, mdp);
        
        RandStream.setGlobalStream(RandStream.create('mrg32k3a', ...
            'NumStreams', 1, 'Seed', problem.seed + i));
        [trajs, policy] = generateTrajectory(mdp, problem);
        
        data.weight  = cat(2, data.weight, w);
        data.policy  = cat(2, data.policy, policy);
        data.trajId  = cat(1, data.trajId, repmat(i, problem.nTrajs, 1));
        data.trajSet = cat(1, data.trajSet, trajs);
    end
end

end


function [trajs, policy] = generateTrajectory(mdp, problem)

% compute the optimal policy
fprintf('  Policy iteration : ');
[policy, value] = policyIteration(mdp);
optValue = full(mdp.start'*value);
fprintf('%.4f\n', optValue);

% Sample trajectories (make sure sampling is good)
meanThreshold = 1;
varThreshold = 1;
fprintf('  Sample %d trajectories : ', problem.nTrajs);
while 1
    [trajs, trajVmean, trajVvar] = sampleTrajectories(problem.nTrajs, ...
        problem.nSteps, policy, mdp);
    if abs(optValue - trajVmean) < meanThreshold && trajVvar < varThreshold
        break;
    end
end
fprintf('%.4f (%.4f)\n', trajVmean, trajVvar);

end


function w = sampleWeight(name, nF, i)

w = zeros(nF, 1);
if strcmp(name, 'highway')
    if i == 1   % fast driver avoids collisions and prefers left-lane
        w(1) = -1;
        w(2) = -0.1;
        w(3) = 0.1;
        w(6) = -0.1;
        w(end) = 1;
    elseif i == 2   % safe driver avoids collisions and off-roads and prefers right-lane
        w(1) = -1;
        w(2) = -0.5;
        w(5) = 0.1;
        w(6) = -0.5;
    elseif i == 3   % demolition prefers collisions and high-speed
        w(1) = 1;
        w(end) = 1;
    elseif i == 4   % nasty driver prefers collisions and off-roads
        w(1) = 1;
        w(2) = 0.5;
        w(6) = 0.5;
    else            % driver prefers randomly chosen lane and speed
        w1 = zeros(3, 1);
        w1(randi(3)) = 0.2;
        w2 = zeros(3, 1);
        w2(randi(3)) = 0.5;
        w = [-1; -0.1; w1; -0.1; w2];
    end
    
elseif strcmp(name, 'highway2')
    if i == 1   % fast driver avoids collisions and prefers left-lane
        w(1) = -1;  % collision
        w(2) = 0.1; % left-lane
        w(end) = 1; % high-speed
    elseif i == 2   % safe driver avoids collisions and prefers right-lane
        w(1) = -1;  % collision
        w(4) = 0.1; % right-lane
        w(5) = 1;   % slow-speed
    elseif i == 3   % demolition prefers collisions and high-speed
        w(1) = 1;   % collision
        w(end) = 1; % high-speed
    else            % driver prefers randomly chosen lane and speed
        w1 = zeros(3, 1);
        w1(randi(3)) = 0.2;
        w2 = zeros(3, 1);
        w2(randi(3)) = 0.5;
        w = [-1; w1; w2];
    end
    
elseif strcmp(name, 'highway3')
    if i == 1               % fast driver avoids collisions and prefers high speed
        w(1)   = -1;        % collision
        w(end) = 0.1;       % high-speed
    elseif i == 2           % safe driver avoids collisions and prefers right-most lane
        w(1)       = -1;    % collision
        w(end - 2) = 0.1;   % right-most lane
    elseif i == 3           % demolition prefers collisions and high-speed
        w(1)   = 1;         % collision
        w(end) = 0.1;       % high-speed
    end
    
elseif strcmp(name, 'gridworld')
    l = randperm(nF);
    k = ceil(0.3*nF);
    idx = l(1:k);
    w(idx) = rand(k, 1) * 2 - 1;
    
%     l = randperm(nF - 1);
%     k = ceil(0.3*nF);
%     idx = l(1:k);
%     w(idx) = rand(k, 1) - 1;
%     w(end) = 1;
    
elseif strcmp(name, 'gridworld2')
    w = -rand(nF, 1);
    
end

end