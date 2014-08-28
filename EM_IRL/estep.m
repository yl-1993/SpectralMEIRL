% Expectation step
function [z, L] = estep(data, trajInfo, mdp, opts)

nTrajs = length(trajInfo);
nrCl   = opts.nClust;

logLLH = zeros(nTrajs, nrCl);
for k = 1:nrCl
    w = data.weight(:, k);
    for m = 1:nTrajs
        logLLH(m, k) = calLogLLH(w, trajInfo{m}, ...
            [], [], [], mdp, opts);
    end
end

z = zeros(nTrajs, nrCl);
for m = 1:nTrajs
    for k = 1:nrCl
        z(m, k) = data.rho(k) * exp(logLLH(m, k));
    end
    if sum(z(m, :)) > 0
        z(m, :) = z(m, :)./sum(z(m, :));
    end
end

L = 0;
for m = 1:nTrajs
    for k = 1:nrCl
        if z(m, k) > 0
            L = L + (log(data.rho(k)) + logLLH(m, k)) * z(m, k);
        end
    end
end

end