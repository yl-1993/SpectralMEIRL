% MMP likelihood without loss function
% Ratliff et al., Maximum margin planning, ICML 2006
%
function [llh, grad] = calLogLLH_MMP(w, trajInfo, mdp, irlOpts)

lambda = 1/irlOpts.sigma^2;
if isfield(irlOpts, 'slackPenalty') && irlOpts.slackPenalty == 2
    slackPenalty = 2;
else
    slackPenalty = 1;
end

mdp = convertW2R(w, mdp);
[~, ~, ~, H] = policyIteration(mdp);
featExp  = H'*mdp.start;
featExpE = trajInfo.featExp;

cost = (w'*(featExp - featExpE))^slackPenalty;
cost = cost + lambda/2*(w'*w);
llh  = -cost;
if nargout == 2
    grad = calMMPGrad(w, featExp, featExpE, lambda, slackPenalty);
    grad = -grad;
end

end


function grad = calMMPGrad(w, featExp, featExpE, lambda, slackPenalty)

if slackPenalty == 2
    C = irlOpts.slackPenalty*w'*(featExp - featExpE);
else
    C = 1;
end
grad = C*(featExp - featExpE) + lambda*w;

end