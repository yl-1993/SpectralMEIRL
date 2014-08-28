% Maximization step
function [rho, weight] = mstep(data, trajInfo, mdp, opts)

nTrajs = length(trajInfo);
nrCl   = opts.nClust;

lb = repmat(opts.lb, mdp.nFeatures, 1);
ub = repmat(opts.ub, mdp.nFeatures, 1);
if opts.showMsg
    display = 'iter';
else
    warning off all;
    display = 'off';
end
options = optimset('Display', display, 'algorithm', 'sqp', ...
    'GradObj', 'on', 'TolX', 1e-8, 'TolFun', 1e-4);

rho = zeros(nrCl, 1);
for k = 1:nrCl
    rho(k) = sum(data.z(:, k));
end
rho = rho / nTrajs;

weight = nan(mdp.nFeatures, nrCl);
for k = 1:nrCl;
    if rho(k) > 0
        objFunc = @(x)calEMLogLLH(data.z(:, k), x, ...
            trajInfo, [], [], [], mdp, opts);        
        
        sol.w = [];
        sol.v = -inf;
        for iter = 1:opts.restart
            w0 = data.weight(:, k);
            [w, val] = fmincon(objFunc, w0, [], [], [], [], lb, ub, [], options);
            if sol.v < -val
                sol.v = -val;
                sol.w = w;
            end
        end
        weight(:, k) = sol.w;
    else
        weight(:, k) = data.weight(:, k);
    end
end

end