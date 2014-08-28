% Metropolis-Hastings algorithm for IRL with multiple experts using
% Langevin algorithm
%
function [sol, logPost, hst] = DPM_BIRL_MHL(trajSet, mdp, opts)

fprintf('- DPM-BIRL using MH (Langevin algorithm)\n');

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
        cl = updateClustAssign(m, tr, cl, mdp, opts);
    end    
    cl = relabelClustAssign(cl);    
    
    for k = randperm(max(cl.b)) % random permutation of ordering of clusters
        cl = updateWeight(k, tr, cl, mdp, opts);
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


% Perform Metropolis-Hastings update for cluster assignment of m-th
% trajectory
function cl = updateClustAssign(m, tr, cl, mdp, opts)

for iiter = 1:opts.clIters;
    c = cl.b(m);
    w = cl.w(:, c);
    p = cl.p(:, c);
    v = cl.v(:, c);
    cl.llh(c)      = nan;
    cl.prior(c)    = nan;
    cl.gradL(:, c) = nan(size(w));
    cl.gradP(:, c) = nan(size(w));
    
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
    
    ratio = calAcceptRatio(tr.cnt{m}, w2, v2, w, v, mdp, opts);
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
function cl = updateWeight(k, tr, cl, mdp, opts)

sigma = 1e-2;
for iiter = 1:opts.wIters;    
    trajInfo = getTrajInfo(tr.set(cl.b == k, :, :), mdp);
    w = cl.w(:, k);
    p = cl.p(:, k);
    v = cl.v(:, k);
    if isnan(cl.llh(k))
        [llh, gradL]   = calLogLLH(w, trajInfo, p, [], [], mdp, opts);
        [prior, gradP] = calLogPrior(w, opts);
        cl.llh(k)      = llh;
        cl.prior(k)    = prior;
        cl.gradL(:, k) = gradL;
        cl.gradP(:, k) = gradP;
    end
    logP = cl.llh(k) + cl.prior(k);
    grad = cl.gradL(:, k) + cl.gradP(:, k);
        
    eps = randn(mdp.nFeatures, 1);
    w2  = w + sigma.^2*grad./2 + sigma.*eps;
    w2  = max(opts.lb, min(opts.ub, w2));
    mdp = convertW2R(w2, mdp);
    [p2, v2] = policyIteration(mdp);    
    [llh2, gradL2]   = calLogLLH(w2, trajInfo, p2, [], [], mdp, opts);
    [prior2, gradP2] = calLogPrior(w2, opts);
    logP2 = llh2 + prior2;
    grad2 = gradL2 + gradP2;
    
    a = eps + sigma/2*(grad + grad2);
    a = exp(-0.5*sum(a.^2))*exp(logP2);
    b = exp(-0.5*sum(eps.^2))*exp(logP);
    
    ratio = a/b;
    if rand < ratio
        cl.w(:, k) = w2;
        cl.p(:, k) = p2;
        cl.v(:, k) = v2;
        cl.llh(k)      = llh2;
        cl.prior(k)    = prior2;
        cl.gradL(:, k) = gradL2;
        cl.gradP(:, k) = gradP2;
    end
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
cl.llh   = nan(N, 1);
cl.prior = nan(N, 1);
cl.gradL = nan(size(cl.w));
cl.gradP = nan(size(cl.w));

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
