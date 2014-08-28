% Compute negative log posterior in order to perform gradient descent
%
function [post, grad, prior, llh] = calNegLogPost(w, ...
    piL, H, dQ, trajInfo, mdp, opts)

if nargout == 1
    llh   = calLogLLH(w, trajInfo, piL, H, dQ, mdp, opts);
    prior = calLogPrior(w, opts);
else
    [llh, grad1]   = calLogLLH(w, trajInfo, piL, H, dQ, mdp, opts);
    [prior, grad2] = calLogPrior(w, opts);
    grad = grad1 + grad2;
    grad = -grad;
%     fprintf('%8.2f %8.2f : %8.2f %8.2f\n', llh, prior, min(grad2), max(grad2));
end
post = prior + llh;
post = -post;


if isinf(post)
    fprintf('ERROR: prior: %f, llh:%f, eta:%f, w:%f %f \n', ...
        prior, llh, opts.eta, full(min(w)), full(max(w)));
end

end