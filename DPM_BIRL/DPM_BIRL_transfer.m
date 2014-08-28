% Information transfer to new trajectory using Metropolis-Hastings 
% sampling and the results from DPM-BIRL
%
function wL = DPM_BIRL_transfer(hst, traj, mdp, opts)

% build distribution for 1d reward value
opts = getRewardDist(opts);

% count state-action pairs
trajInfo = getTrajInfo(traj, mdp);

% using all samples
n = length(hst.cl);
N = ceil(0.8*n);
cl = hst.cl(end - N + 1:end);

% % using maximum a posterior estimate
% [~, i] = max(hst.logPost);
% cl = hst.cl(i);
% N = 1;

tmpW = [];
tmpP = [];
tmpZ = [];
for i = 1:N
    for k = 1:max(cl{i}.b)
        tmpW = cat(2, tmpW, cl{i}.w(:, k));
        tmpP = cat(2, tmpP, cl{i}.p(:, k));
        z = sum(cl{i}.b == k);
        tmpZ = cat(2, tmpZ, z);        
    end
end

% find duplicated reward
W = [];
P = [];
Z = [];
while ~isempty(tmpW)
    w = tmpW(:, 1);
    p = tmpP(:, 1);
    x = bsxfun(@minus, tmpW, w); 
    b = sum(x.^2) == 0;
    z = sum(tmpZ(b));
    W = cat(2, W, w);
    P = cat(2, P, p);
    Z = cat(2, Z, z);
    tmpW = tmpW(:, ~b);
    tmpP = tmpP(:, ~b);
    tmpZ = tmpZ(:, ~b);
end
Z = Z./N;

% pre-compute prior distribution for sampling
w0  = MAP_BIRL(traj, [], mdp, opts);
mdp = convertW2R(w0, mdp);
p0  = policyIteration(mdp);
llh0   = calLogLLH(w0, trajInfo, p0, [], [], mdp, opts);
prior0 = opts.alpha*exp(calLogPrior(w0, opts));
dist.llh   = llh0;
dist.prior = prior0;
dist.w     = w0;

for i = 1:size(W, 2)
    w = W(:, i);
    prior = Z(i);
    if isequal(w0, w)
        fprintf('w0 == w at %d\n', i);
        prior = prior + prior0;
        llh   = llh0;
        dist.prior(1) = 0;
    else
        llh = calLogLLH(w, trajInfo, P(:, i), [], [], mdp, opts);
    end
    dist.llh   = cat(2, dist.llh, llh);
    dist.prior = cat(2, dist.prior, prior);
    dist.w     = cat(2, dist.w, w);
end

% initialize
i = sampleMultinomial(dist.prior);
w   = dist.w(:, i);
llh = dist.llh(i);
post = llh + log(dist.prior(:, i));
S = w;
maxData.post = post;
maxData.w    = w;
maxData.i    = i;
fprintf('  init : %4d %12.4f\n', i, post);

for iter = 1:opts.newIters;
    i = sampleMultinomial(dist.prior);
    w2   = dist.w(:, i);
    llh2 = dist.llh(i);
    
    if rand < exp(llh2 - llh)
        w    = w2;
        llh  = llh2;
        post = llh + log(dist.prior(:, i));
        fprintf(' %5d : %4d %12.4f %12.4f\n', iter, i, post, maxData.post);
    end
    S = cat(2, S, w);
    if maxData.post < post
        maxData.post = post;
        maxData.w    = w;
        maxData.i    = i;
    end
end

wL.mean = mean(S, 2);
wL.max  = maxData.w;

end
