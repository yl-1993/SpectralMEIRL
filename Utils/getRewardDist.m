% Build distribution for 1d reward value
%
function opts = getRewardDist(opts)

rlist = [];
rdist = [];
if strcmp(opts.priorType, 'NG')      % normal-gamma dist for hyper-prior
    rlist = (opts.lb(1): opts.delta: opts.ub(1));
    rdist = 2*opts.gamma(2) + opts.beta.*rlist.^2./(1 + opts.beta);
    rdist = rdist.^(-opts.gamma(1)-0.5);
    rdist = rdist./sum(rdist);
    
elseif strcmp(opts.priorType, 'BG')  % beta-gamma dist for hyper-prior
    mlist = betarnd(opts.beta, 1 - opts.beta, [opts.nmu, 1]);
    vlist = gamrnd(opts.gamma(1), 1/opts.gamma(2), size(mlist));
    a = mlist.*vlist;
    b = (1 - mlist).*vlist;
    
    rlist = betarnd(a, b, size(a));
    rlist = rlist(rlist > opts.lb(1) & rlist < opts.ub(1));
    rdist = ones(size(rlist));
    rdist = rdist./sum(rdist);
    
    opts.beta1 = a;
    opts.beta2 = b;
end

opts.rlist = rlist;
opts.rdist = rdist;

end