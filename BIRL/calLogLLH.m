% Compute log likelihood and gradient of trajectories
% given reward function w and inverse temperature eta
%   log p(X | w, opts.eta)
%
function [llh, grad] = calLogLLH(w, trajInfo, piL, H, dQ, mdp, opts)

if strcmp(opts.llhType, 'BIRL')
    if nargout == 1
        llh = calLogLLH_BIRL(w, opts.eta, ...
            trajInfo, piL, H, dQ, mdp);
    else
        [llh, grad] = calLogLLH_BIRL(w, opts.eta, ...
            trajInfo, piL, H, dQ, mdp);
    end
    
elseif strcmp(opts.llhType, 'MaxEnt')
    if nargout == 1
        llh = calLogLLH_MaxEnt(w, trajInfo, mdp);
    else
        [llh, grad] = calLogLLH_MaxEnt(w, trajInfo, mdp);
    end
    
elseif strcmp(opts.llhType, 'PM')
    if nargout == 1
        llh = calLogLLH_PM(w, trajInfo, mdp, opts);
    else
        [llh, grad] = calLogLLH_PM(w, trajInfo, mdp, opts);
    end
    
elseif strcmp(opts.llhType, 'MMP')
    if nargout == 1
        llh = calLogLLH_MMP(w, trajInfo, mdp, opts);
    else
        [llh, grad] = calLogLLH_MMP(w, trajInfo, mdp, opts);
    end
    
elseif strcmp(opts.llhType, 'MLIRL')
    if nargout == 1
        llh = calLogLLH_MLIRL(w, opts.eta, trajInfo, mdp);
    else
        [llh, grad] = calLogLLH_MLIRL(w, opts.eta, trajInfo, mdp);
    end
end

end
