% Generate gridworld problem [Abbeel & Ng, ICML 2004]
function mdp = gridworld(gridSize, blockSize, noise, discount, bprint)

nS = gridSize^2;                % # of states
nA = 4;                         % actions: north, south, west, east
nF = (gridSize/blockSize)^2;    % # of features

% state transition: T(s',s,a) = P(s'|s,a)
T = zeros(nS, nS, nA);
for y = 1:gridSize
    for x = 1:gridSize
        s = loc2s(x, y, gridSize);
        ns    = zeros(4, 1);
        ns(1) = loc2s(x, y - 1, gridSize);
        ns(2) = loc2s(x, y + 1, gridSize);
        ns(3) = loc2s(x - 1, y, gridSize);
        ns(4) = loc2s(x + 1, y, gridSize);
        for a = 1:nA
            for a2 = 1:nA
                T(ns(a2), s, a) = T(ns(a2), s, a) + noise/nA;
            end
            T(ns(a), s, a) = T(ns(a), s, a) + 1 - noise;
        end
    end
end

% check transition probability
for s = 1:nS
    for a = 1:nA
        err = abs(sum(T(:, s, a)) - 1);
        if err > 1e-6 || nnz(T(:, s, a) < 0) > 0 || nnz(T(:, s, a) > 1)
            fprintf('ERROR: %d %d %f \n', s, a, err);
        end
    end
end

% state feature
F = zeros(nS, nF);
for y = 1:gridSize
    for x = 1:gridSize
        s = loc2s(x, y, gridSize);
        i = ceil(x/blockSize);
        j = ceil(y/blockSize);
        f = loc2s(i, j, gridSize/blockSize);
        F(s, f) = 1;
    end
end

% initial state distribution
start = ones(nS, 1);
start = start./sum(start);

% l = randperm(nS);
% k = ceil(0.1*nS);
% % k = ceil(log(nF));
% idx = l(1:k);
% start = zeros(nS, 1);
% start(idx) = 1;
% start = start./sum(start);

% start = zeros(nS, 1);
% start(1) = 1;

% weight vector
weight = zeros(nF, 1);
l = randperm(nF - 1);
k = ceil(0.3*nF);
% k = ceil(log(nF));
idx = l(1:k);
weight(idx) = rand(k, 1) - 1;
weight(end) = 1;
%weight = weight./norm(weight, 1);

% generate blockSizeDP
mdp.name       = sprintf('gridworld_%dx%d', gridSize, blockSize);
mdp.gridSize   = gridSize;
mdp.blockSize  = blockSize;
mdp.nStates    = nS;
mdp.nActions   = nA;
mdp.nFeatures  = nF;
mdp.discount   = discount;
mdp.useSparse  = 1;
mdp.start      = start;
mdp.transition = T;
mdp.F          = repmat(F, nA, 1);
mdp.weight     = weight;
mdp.reward     = reshape(mdp.F*mdp.weight, nS, nA);

if mdp.useSparse
    mdp.F      = sparse(mdp.F);
    mdp.weight = sparse(mdp.weight);
    mdp.start  = sparse(mdp.start);    
    for a = 1:mdp.nActions
        mdp.transitionS{a} = sparse(mdp.transition(:, :, a));
        mdp.rewardS{a} = sparse(mdp.reward(:, a));
    end
end

if nargin >= 5 && bprint
    fprintf('solve %s\n', mdp.name);
    tic;
    [policy, value] = policyIteration(mdp);
    elapsedTime = toc;
    
    fprintf('== %s ==\n', mdp.name);
    actstr = 'NSWE';
    for y = 1:gridSize
        for x = 1:gridSize
            s = loc2s(x, y, gridSize);
            i = ceil(x/blockSize);
            j = ceil(y/blockSize);
            f = loc2s(i, j, gridSize/blockSize);
            a = actstr(policy(s));
            w = full(mdp.weight(f));
            fprintf('%2d %1s[%2d: %5.2f] ', s, a, f, w);
        end
        fprintf('\n');
    end
    fprintf('\n');
    
    nTrajs = 2;
    nSteps = gridSize*4;
    [trajs, trajVmean, trajVvar] = ...
        sampleTrajectories(nTrajs, nSteps, policy, mdp);
    
    for t = 1:nSteps
        for m = 1:nTrajs
            s = trajs(m, t, 1);
            a = trajs(m, t, 2);
            x = mod(s - 1, gridSize) + 1;
            y = floor((s - 1)/gridSize) + 1;
            f = loc2s(ceil(x/blockSize), ceil(y/blockSize), gridSize/blockSize);
            fprintf('%4d:: %2d | %2d %2d [%2d] : %s : %5.2f  | ', ...
                t, s, x, y, f, actstr(a), mdp.reward(s, a));
        end
        fprintf('\n');
    end    
    fprintf('\ntraj. value    : %.4f (%.4f)', trajVmean, trajVvar);
    fprintf('\nopt. value     : %.4f %.2f sec\n\n', ...
        full(mdp.start'*value), elapsedTime);
end

end


function s = loc2s(x, y, gridSize)

x = max(1, min(gridSize, x));
y = max(1, min(gridSize, y));
s = (y - 1)*gridSize + x;

end