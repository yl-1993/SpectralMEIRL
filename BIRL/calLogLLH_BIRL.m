% BIRL likelihood
% Ramachandran & Amir, Bayesian inverse reinforcement learning, IJCAI 2007
%
function [llh, grad] = calLogLLH_BIRL(w, eta, ...
    trajInfo, piL, H, dQ, mdp)

mdp = convertW2R(w, mdp);
if isempty(piL) || isempty(H)
    [piL, ~, QL, ~] = policyIteration(mdp);
    if nargout == 2 && isempty(dQ)
        dQ = calGradQ(piL, mdp);
    end
else
    VL = H*w;
    QL = QfromV(VL, mdp);
end

nF = mdp.nFeatures;
nS = mdp.nStates;
nA = mdp.nActions;

BQ    = eta.*QL;
BQsum = log(sum(exp(BQ), 2));
NBQ   = bsxfun(@minus, BQ, BQsum);

llh = 0;
for i = 1:size(trajInfo.cnt, 1)
    s = trajInfo.cnt(i, 1);
    a = trajInfo.cnt(i, 2);
    n = trajInfo.cnt(i, 3);
    llh = llh + NBQ(s, a)*n;
end

if nargout == 2
    % compute soft-max policy
    pi_sto = exp(BQ);
    pi_sto = bsxfun(@rdivide, pi_sto, sum(pi_sto, 2));
    
    % calculate dlogPi/dw
    dlogPi = zeros(nF, nS*nA);
    for f = 1:nF
        x = reshape(dQ(f, :), nS, nA);
        y = sum(pi_sto.*x, 2);
        z = eta.*bsxfun(@minus, x, y);
        dlogPi(f, :) = reshape(z, 1, nS*nA);
    end
    
    % calculate gradient of reward function
    grad = 0;
    for i = 1:size(trajInfo.cnt, 1)
        s = trajInfo.cnt(i, 1);
        a = trajInfo.cnt(i, 2);
        n = trajInfo.cnt(i, 3);
        j = (a-1)*nS+s;
        grad = grad+n*dlogPi(:, j);
    end
end

end
