% MAP inference for BIRL
% Compute maximum of posterior of reward function w
% Choi & Kim, MAP inference for Bayesian inverse reinforcement learning,
% NIPS 2011
%
function [wL, logPost] = MAP_BIRL(trajs, w0, mdp, opts)

% build distribution for 1d reward value
if strcmp(opts.priorType, 'NG') || strcmp(opts.priorType, 'BG')
    if ~isfield(opts, 'rlist') || isempty(opts.rlist) ...
        || ~isfield(opts, 'rdist') || isempty(opts.rdist)
        opts = getRewardDist(opts);
    end
end

nF = mdp.nFeatures;
lb = repmat(opts.lb(1), nF, 1);
ub = repmat(opts.ub(1), nF, 1);
trajInfo = getTrajInfo(trajs, mdp);
objFunc  = @(x)calNegLogPost(x, [], [], [], trajInfo, mdp, opts);

if opts.showMsg
    disp = 'iter';
else
    warning off all;
    disp = 'off';
end
options = optimset('Display', disp, 'algorithm', 'sqp', ...
    'GradObj', 'on', 'TolX', 1e-8, 'TolFun', 1e-4);

sol.w = [];
sol.v = -inf;
for iter = 1:opts.restart
    if isempty(w0)
        w0 = init(objFunc, mdp, opts);
    end
    
    [w, val] = fmincon(objFunc, w0, [], [], [], [], lb, ub, [], options);
    
    if sol.v < -val
        sol.v = -val;
        sol.w = w;
    end
    w0 = [];
end
wL      = sol.w;
logPost = sol.v;

end


% Initialize weight vector
function w0 = init(objFunc, mdp, opts)

w0 = sampleNewWeight(mdp.nFeatures, opts);
v0 = objFunc(w0);
k  = 0;
while isnan(v0) || isinf(v0)
    w0 = sampleNewWeight(mdp.nFeatures, opts);
    v0 = objFunc(w0);
    k  = k + 1;
end
if k > 0
    fprintf('ERROR: obj. function returns NaN or INF during initialization\n');
end

end