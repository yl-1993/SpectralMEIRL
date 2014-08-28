% Policy-matching likelihood
% Neu & Szepesvari, Apprenticeship learning using inverse reinforcement
% learning and gradient methods, UAI 2007
%
function [LLH, grad] = calLogLLH_PM(w, trajInfo, mdp, irlOpts)

nS = mdp.nStates;
nA = mdp.nActions;
nF = mdp.nFeatures;
eta = irlOpts.eta;
mdp = convertW2R(w, mdp);
[detP, ~, QL] = policyIteration(mdp);

% compute soft-max policy
BQ = eta.*QL;
stoP = exp(BQ);
stoP = bsxfun(@rdivide, stoP, sum(stoP, 2));

% calculate dP/dw
dQ = calGradQ(detP, mdp);
dP = zeros(nF, nS*nA);
for f = 1:nF
    x = reshape(dQ(f, :), nS, nA);
    y = sum(stoP.*x, 2);
    z = stoP.*bsxfun(@minus, x, y);
    dP(f, :) = eta*reshape(z, 1, nS*nA);
end

% calculate cost J and dJ/dP
z = (stoP - trajInfo.pi).^2;
J = sum(trajInfo.mu'*sum(z, 2));
LLH = -J;

if nargout >= 2
    % calculate gradient
    dJ = 2*bsxfun(@times, stoP - trajInfo.pi, trajInfo.mu);
    dJ = reshape(dJ, nS*nA, 1);
    grad = dP*dJ;
    
    % calculate natural gradient
    if isfield(irlOpts, 'natural') && irlOpts.natural
        G = dPi*dPi';
        grad = pinv(G)*grad;
    end
    grad = -grad;
end

end