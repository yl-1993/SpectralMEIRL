% Metropolis-Hastings algorithm for IRL with multiple experts
%
function [sol, logPost, hst] = DPM_BIRL_MH(trajSet, mdp, opts)

fprintf('- DPM-BIRL using MH\n');

% build distribution for 1d reward value
opts = getRewardDist(opts);

% trajectory data
tr.set = trajSet;          % trajectories
tr.n   = size(tr.set, 1);  % number of trajectories
tr.cnt = cell(tr.n, 1);    % information of trajectories
for i = 1:tr.n
    info = getTrajInfo(tr.set(i, :, :), mdp);
    tr.cnt{i} = info.cnt;
end

% initialize cluster
startT = tic;
cl = init(tr, mdp, opts);
elapsedT = toc(startT);
cl = relabelClustAssign(cl);
pr = calDPMLogPost(tr.set, cl.b, cl.w, cl.p, mdp, opts);
fprintf('+    init : %10.2f %10s %8.2f sec ', pr, '', elapsedT);
printClustAssign(cl.b);

maxCl.logPost = -inf;
[maxCl, hst, ~] = saveHist(cl, pr, maxCl, [], 1, elapsedT);

for iter = 1:opts.maxIters
    for m = randperm(tr.n)      % random permutation of ordering of trajectories
        cl = updateClustAssign(m, tr.cnt, cl, mdp, opts);
    end    
    cl = relabelClustAssign(cl);    
    
    for k = randperm(max(cl.b)) % random permutation of ordering of clusters
        cl = updateWeight(k, tr.cnt, cl, mdp, opts);
    end
    
    elapsedT = toc(startT);
    pr = calDPMLogPost(tr.set, cl.b, cl.w, cl.p, mdp, opts);
    [maxCl, hst, bUpdate] = saveHist(cl, pr, maxCl, hst, iter + 1, elapsedT);
    
    if iter == 1 || mod(iter, 100) == 0 || bUpdate || elapsedT > opts.maxTime
        fprintf('+ %4d-th : %10.2f %10.2f %8.2f sec ', ...
            iter, pr, maxCl.logPost, elapsedT);
        printClustAssign(maxCl.belongTo);
    end
    if elapsedT > opts.maxTime
        fprintf('# time out\n');
        break;
    end
end
sol     = maxCl;
logPost = maxCl.logPost;
fprintf('\n\n');

end


function [maxCl, hst, bUpdate] = saveHist(cl, pr, maxCl, hst, i, elapsedT)

bUpdate = false;
if maxCl.logPost < pr
    maxCl.belongTo = cl.b;
    maxCl.weight   = cl.w;
    maxCl.policy   = cl.p;
    maxCl.value    = cl.v;
    maxCl.logPost  = pr;
    bUpdate = true;
end
hst.cl{i}      = cl;
hst.logPost(i) = pr;
hst.maxLogPost(i) = maxCl.logPost;
hst.elapsedT(i)   = elapsedT;

end


% Perform Metropolis-Hastings update for cluster assignment of m-th
% trajectory
function cl = updateClustAssign(m, trajCnt, cl, mdp, opts)

for iiter = 1:opts.clIters;
    c = cl.b(m);
    w = cl.w(:, c);
    v = cl.v(:, c);
    
    N = max(cl.b);
    prior = zeros(N + 1, 1);
    for k = 1:N
        prior(k) = sum(cl.b == k);
    end
    prior(c)   = prior(c) - 1;
    prior(end) = opts.alpha;
    
    c2 = sampleMultinomial(prior);
    if c2 > N
        [w2, p2, v2] = newWeight(mdp, opts);
    else
        w2 = cl.w(:, c2);
        p2 = cl.p(:, c2);
        v2 = cl.v(:, c2);
    end
    
    ratio = calAcceptRatio(trajCnt{m}, w2, v2, w, v, mdp, opts);
    if rand < ratio
        cl.b(m) = c2;
        if c2 > N
            cl.w(:, c2) = w2;
            cl.p(:, c2) = p2;
            cl.v(:, c2) = v2;
        end
    end
end

end


% Perform Metropolis-Hastings update for weight of k-th cluster
function cl = updateWeight(k, trajCnt, cl, mdp, opts)

for iiter = 1:opts.wIters;
    trajs = trajCnt(cl.b == k);
    w = cl.w(:, k);
    v = cl.v(:, k);
    
    [w2, p2, v2] = newWeight(mdp, opts);
    
    ratio = calAcceptRatio(trajs, w2, v2, w, v, mdp, opts);
    if rand < ratio
        cl.w(:, k) = w2;
        cl.p(:, k) = p2;
        cl.v(:, k) = v2;
    end
end

end



% Relabel cluster assignment
function cl = relabelClustAssign(cl)

tmpId = [];
W = [];
P = [];
V = [];

bRelabel = false;
for k = 1:max(cl.b)
    if sum(cl.b == k) > 0
        W = cat(2, W, cl.w(:, k));
        P = cat(2, P, cl.p(:, k));
        V = cat(2, V, cl.v(:, k));
        tmpId   = cat(1, tmpId, [k, size(W, 2)]);
        bRelabel = true;
    end
end

if bRelabel
    B = zeros(length(cl.b), 1);
    for i = 1:size(tmpId, 1)
        B(cl.b == tmpId(i, 1)) = tmpId(i, 2);
    end
    cl.b = B;
    cl.w = W;
    cl.p = P;
    cl.v = V;
end

end


% Sample new weight and compute its policy and value
function [w, p, v] = newWeight(mdp, opts)

w      = sampleNewWeight(mdp.nFeatures, opts);
mdp    = convertW2R(w, mdp);
[p, v] = policyIteration(mdp);

end


% Initialize
function cl = init(tr, mdp, opts)
% 
% cl.b = [];
% cl.w = [];
% cl.p = [];
% cl.v = [];
% for i = 1:tr.n
%     w = MAP_BIRL(tr.set(i, :, :), [], mdp, opts);
%     mdp    = convertW2R(w, mdp);
%     [p, v] = policyIteration(mdp);
%     cl.b(:, i) = i;
%     cl.w(:, i) = w;
%     cl.p(:, i) = p;
%     cl.v(:, i) = v;
% end

cl.b = randi(tr.n, tr.n, 1); %ones(tr.n, 1); %

N = max(cl.b);
cl.w = [];
cl.p = [];
cl.v = [];
for i = 1:N
    [w, p, v]  = newWeight(mdp, opts);
    cl.w(:, i) = w;
    cl.p(:, i) = p;
    cl.v(:, i) = v;
end

end
