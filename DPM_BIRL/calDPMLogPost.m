% Calculate log-posterior for DPM-IRL
%
function [logPost, logDPprior, logLLH, logPrior] = ...
    calDPMLogPost(trajSet, belongTo, wSet, policySet, mdp, opts)

logDPprior = log(calClustAssignProb(belongTo, opts.alpha));
logLLH     = 0;
logPrior   = 0;
for k = 1:max(belongTo)
    w = wSet(:, k);    
    if isempty(policySet)
        mdp    = convertW2R(w, mdp);
        policy = policyIteration(mdp);
    else
        policy = policySet(:, k);
    end
        
    X = getTrajInfo(trajSet(belongTo == k, :, :), mdp);
    llh   = calLogLLH(w, X, policy, [], [], mdp, opts);
    prior = calLogPrior(w, opts);
    logLLH   = logLLH + llh;
    logPrior = logPrior + prior;    
end
logPost = logDPprior + logLLH + logPrior;

end