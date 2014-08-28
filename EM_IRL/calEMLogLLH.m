% Compute log likelihood and gradient of trajectories
% given reward function w and inverse temperature eta
% log p(X | w, eta) for EM_IRL
%
function [llh, grad] = calEMLogLLH(z, w, ...
    trajInfo, piL, H, dQ, mdp, opts)

mdp = convertW2R(w, mdp);
if isempty(piL) || isempty(H)
    [piL, ~, ~, H] = policyIteration(mdp);
    if nargout == 2 && isempty(dQ)
        dQ = calGradQ(piL, mdp);
    end
end

if nargout == 1
    llh = 0;
    for m = 1:length(trajInfo)
        if z(m) > 0
            x = calLogLLH(w, trajInfo{m}, piL, H, dQ, mdp, opts);
            llh = llh + z(m) * x;
        end
    end
    % Negate for using gradien descent method
    llh = -llh;
    
else
    llh  = 0;
    grad = 0;
    for m = 1:length(trajInfo)
        if z(m) > 0
            [x, y] = calLogLLH(w, trajInfo{m}, piL, H, dQ, mdp, opts);
            llh  = llh + z(m) * x;
            grad = grad + z(m) * y;
        end
    end
    % Negate for using gradien descent method
    llh  = -llh;
    grad = -grad;
end

end
