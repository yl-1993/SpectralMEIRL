% Compute log prior and gradient of reward function w
%
function [prior, grad] = calLogPrior(w, opts)

if strcmp(opts.priorType, 'NG')      % normal-gamma dist for hyper prior
    prior = sum(log(2*opts.gamma(2)+opts.beta.*w.^2./(opts.beta+1)));
    prior = -(opts.gamma(1)+0.5)*prior;
    if nargout == 2
        grad = 2*(opts.beta+1)*opts.gamma(2)+opts.beta.*w.^2;
        grad = w./grad;
        grad = -2*opts.beta*(opts.gamma(1)+0.5)*grad;
    end
    
elseif strcmp(opts.priorType, 'BG')  % beta-gamma dist for hyper prior
    if sum(w <= 0 | w >= 1) > 0
        prior = -inf;
        grad  = nan(size(w));
    else
%         mlist = betarnd(opts.beta, 1 - opts.beta, [opts.nmu, 1]);
%         vlist = gamrnd(opts.gamma(1), 1/opts.gamma(2), [opts.nmu, 1]);
%         opts.beta1 = mlist.*vlist;
%         opts.beta2 = (1 - mlist).*vlist;
        
        n = length(opts.beta1);
        z = nan(n, length(w));
        for d = 1:length(w)
            z(:, d) = betapdf(w(d), opts.beta1, opts.beta2);
        end
        x = sum(z);
        prior = sum(log(x./n));
        if nargout == 2
            grad = nan(size(w));
            for d = 1:length(w)
                p = (opts.beta1 - 1)./w(d) - (opts.beta2 - 1)./(1 - w(d));
                grad(d) = sum(z(:, d).*p);
            end
            grad = grad./x';
            %grad = grad*1e-2;
        end
    end
    
elseif strcmp(opts.priorType, 'Uniform')
    prior = log(1);
    grad  = zeros(size(w));
    
elseif strcmp(opts.priorType, 'Gaussian')
    x = w - opts.mu;
    prior = sum(-(x'*x)./(2*opts.sigma.^2));
    grad  = -x./(opts.sigma.^2);
end

end
