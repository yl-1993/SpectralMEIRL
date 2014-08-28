% Generate highway problem [Abbeel & Ng, ICML 2004]
% speed: 2~4
% lanes: 3
function mdp = highway3(discount, bprint)

appearanceProb = [0.4, 0.6, 0.8];   % prob. of other car appearing on each lane
succProb = [0.8, 0.4; 1.0, 0.8];
carSize  = 2;
nSpeeds  = 2;
nLanes   = 3;
nGrids   = 8;
nS       = nSpeeds*nLanes*nGrids^nLanes;

nA = 5;                     % nop, move left, move right, speed up, speed down
nF = 1 + nLanes + nSpeeds;  % collision, lanes, speeds
T  = zeros(nS, nS, nA);     % state transition probability
F  = zeros(nS, nF);         % state feature
for s = 1:nS
    [spd, myx, y1, y2, y3] = sid2info(s, nSpeeds, nLanes, nGrids);
    
    nX = zeros(nA, 2);
    nX(1, :) = [spd, myx];                     % nop
    nX(2, :) = [spd, max(1, myx - 1)];         % move left 
    nX(3, :) = [spd, min(nLanes, myx + 1)];    % move right
    nX(4, :) = [min(nSpeeds, spd + 1), myx];   % speed up
    nX(5, :) = [max(1, spd - 1), myx];         % speed down
    
    % location of other cars
    Y    = [y1, y2, y3];  
    idx1 = find(Y > 1);
    idx2 = find(Y == 1);
    
    nY = [];
    for i = idx2
        Y2    = Y;
        Y2(i) = Y2(i) + 1;
        nY    = cat(1, nY, Y2);
    end
    Y2 = Y;
    nY = cat(1, nY, Y2);
    nY(:, idx1) = nY(:, idx1) + spd;
    nY(nY > nGrids) = 1;
    
    nY = cat(2, nY, zeros(size(nY, 1), 1));
    for i = 1:size(nY, 1)
        p = 1;
        for j = 1:nLanes
            if Y(j) == 1 && nY(i, j) == 2
                p = p*appearanceProb(j);
            elseif Y(j) == 1 && nY(i, j) == 1
                p = p*(1 - appearanceProb(j));
            end
        end
        nY(i, nLanes + 1) = p;
    end
    p = 1 - sum(nY(:, nLanes + 1));
    nY(end, nLanes + 1) = nY(end, nLanes + 1) + p;

    for a = 1:nA
        % calculate transition probability
        for i = 1:size(nY, 1)
            ns = info2sid(nX(a, 1), nX(a, 2), nY(i, 1), nY(i, 2), nY(i, 3), ...
                nSpeeds, nLanes, nGrids);
            if a == 2 || a == 3
                pr = succProb(1, spd)^(spd - 1);
                T(ns, s, a) = T(ns, s, a) + nY(i, 4) * pr;
                ns2 = info2sid(spd, myx, nY(i, 1), nY(i, 2), nY(i, 3), ...
                    nSpeeds, nLanes, nGrids);
                T(ns2, s, a) = T(ns2, s, a) + nY(i, 4) * (1 - pr);
            elseif a == 4 || a == 5
                pr = succProb(2, spd)^(spd - 1);
                T(ns, s, a) = T(ns, s, a) + nY(i, 4) * pr;
                ns2 = info2sid(spd, myx, nY(i, 1), nY(i, 2), nY(i, 3), ...
                    nSpeeds, nLanes, nGrids);
                T(ns2, s, a) = T(ns2, s, a) + nY(i, 4) * (1 - pr);
            else
                T(ns, s, a) = nY(i, 4);
            end 
        end
    end
    
    % calculate feature
    f = zeros(nF, 1);
    
    % check collision
    f(1) = Y(myx) > nGrids - carSize*2 && Y(myx) < nGrids;
    f(1 + myx) = 1;             % lane
    f(1 + nLanes + spd) = 1;    % speed
    F(s, :) = f';
end

% check transition probability
for a = 1:nA
    for s = 1:nS
        err = abs(sum(T(:, s, a)) - 1);
        if err > 1e-6 || nnz(T(:, s, a) < 0) > 0 || nnz(T(:, s, a) > 1)
            fprintf('ERROR: %d %d %f\n', s, a, sum(T(:, s, a)));
        end
    end
end

% initial state distribution
start     = zeros(nS, 1);
s0        = info2sid(1, 2, 1, 1, 1, nSpeeds, nLanes, nGrids);
start(s0) = 1;

% weight for reward
w = zeros(nF, 1);
% fast driver avoids collisions and prefers high speed
w(1) = -1;      % collision
w(end) = 0.1;   % high speed

% % safe driver avoids collisions and prefers right-most lane
% w(1) = -1;
% w(1 + nLanes) = 0.1;

% % demolition prefers collisions and high-speed
% w(1) = 1;
% w(end) = 0.1;

% generate MDP
mdp.name       = 'highway';
mdp.nSpeeds    = nSpeeds;
mdp.nLanes     = nLanes;
mdp.nGrids     = nGrids;
mdp.carSize    = carSize;
mdp.appearProb = appearanceProb;
mdp.succProb   = succProb;
mdp.nStates    = nS;
mdp.nActions   = nA;
mdp.nFeatures  = nF;
mdp.discount   = discount;
mdp.useSparse  = 1;
mdp.start      = start;
mdp.F          = repmat(F, nA, 1);
mdp.transition = T;
mdp.weight     = w;
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

if nargin >= 2 && bprint
    fprintf('solve %s\n', mdp.name);
    tic;    
    [policy, value, ~, H] = policyIteration(mdp);
    elapsedTime = toc;
    featOcc = full(H'*mdp.start);
    
    seed = 1;    
    RandStream.setDefaultStream(RandStream.create('mrg32k3a', ...
        'NumStreams', 1, 'Seed', seed));
    fprintf('sample trajectory\n');
    nTrajs = 1;
    nSteps = 5000;
    [trajs, trajVmean, trajVvar] = ...
        sampleTrajectories(nTrajs, nSteps, policy, mdp);
    
    nFeatures = zeros(nF, 1);
    for t = 1:nSteps
        s = trajs(1, t, 1);
        a = trajs(1, t, 2);
        f = mdp.F((a - 1)*nS + s, :);
        nFeatures = nFeatures + f';
    end
    fprintf('\n# of collisions: %5d', nFeatures(1));
    fprintf('\n# of lanes     : ');
    for i = 1:nLanes
        fprintf('%5d ', nFeatures(1 + i));
    end
    fprintf('\n# of speeds    : ');
    for i = 1:nSpeeds
        fprintf('%5d ', nFeatures(1 + nLanes + i));
    end    
    fprintf('\n');
    
    fprintf('\nweight         : ');
    for i = 1:mdp.nFeatures
        fprintf('%5.2f ', full(mdp.weight(i)));
    end
    fprintf('\nfeat. occupancy: ');
    for i = 1:nF
        fprintf('%5.2f ', featOcc(i));
    end
    fprintf('\ntraj. value    : %.4f (%.4f)', trajVmean, trajVvar);
    fprintf('\nopt. value     : %.4f %.2f sec\n\n', ...
        full(mdp.start'*value), elapsedTime);
end

end
