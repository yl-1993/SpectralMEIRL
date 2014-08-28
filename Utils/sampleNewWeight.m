% Sample new weight from the given prior
%
function w = sampleNewWeight(dim, opts)

lb    = opts.lb;
ub    = opts.ub;

if strcmp(opts.priorType, 'NG') || strcmp(opts.priorType, 'BG')
    w = zeros(dim, 1);
    for d = 1:dim
        ix   = sampleMultinomial(opts.rdist);
        w(d) = opts.rlist(ix);
    end

elseif strcmp(opts.priorType, 'Gaussian')
    mu    = opts.mu;
    sigma = opts.sigma;
    w = mu + randn(dim, 1).*sigma;
    w = max(lb, min(ub, w));
    
else % Uniform
    w = rand(dim, 1) * (ub - lb) + lb;
end

end