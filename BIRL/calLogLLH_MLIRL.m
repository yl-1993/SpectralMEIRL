% MLIRL likelihood
% Babes et al., Apprenticeship learning about multiple intentions, ICML
% 2011
%
function [llh, grad] = calLogLLH_MLIRL(w, eta, trajInfo, mdp)

% Compute stochastic policy
MAX_ITERS = 100;
EPS = 1e-12;

nS = mdp.nStates;
nA = mdp.nActions;
nF = mdp.nFeatures;
gamma = mdp.discount;
mdp = convertW2R(w, mdp);

Q = zeros(nS, nA);
for t = 1:MAX_ITERS
    BQ  = exp(eta.*Q);
    BQS = sum(BQ, 2);
    NBQ = gamma.*sum(bsxfun(@rdivide, Q.*BQ, BQS), 2);

    oldQ = Q;
    for a = 1:nA
        if mdp.useSparse
            Q(:, a) = mdp.rewardS{a} + mdp.transitionS{a}(:, :)'*NBQ;
        else
            Q(:, a) = mdp.reward(:, a) + squeeze(mdp.transition(:, :, a)')*NBQ;
        end
    end
    if approxeq(Q, oldQ, EPS), break; end
end

BQ  = exp(eta.*Q);
BQS = sum(BQ, 2);
piL = bsxfun(@rdivide, BQ, BQS);

% Compute likelihood
llh = 0;
for i = 1:size(trajInfo.cnt, 1)
    s = trajInfo.cnt(i, 1);
    a = trajInfo.cnt(i, 2);
    n = trajInfo.cnt(i, 3);
    llh = llh + log(piL(s, a))*n;
end

if nargout >= 2
    % calculate dQ/dw    
    dQ = calGradQ(piL, mdp);
    
    % calculate dlogPi/dw
    dlogPi = zeros(nF, nS*nA);
    for f = 1:nF
        x = reshape(dQ(f, :), nS, nA);
        y = sum(piL.*x, 2);
        z = eta.*bsxfun(@minus, x, y);
        dlogPi(f, :) = reshape(z, 1, nS*nA);
    end    
    
    % calculate gradient of reward function
    grad = 0;
    for i = 1:size(trajInfo.cnt, 1)
        s = trajInfo.cnt(i, 1);
        a = trajInfo.cnt(i, 2);
        n = trajInfo.cnt(i, 3);
        j = (a - 1)*nS + s;
        grad = grad + n*dlogPi(:, j);
    end
end

end
