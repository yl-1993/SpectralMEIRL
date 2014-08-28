% Expectation Maximization algorithm for IRL with multiple experts
function [sol, logPost, hst] = EM_IRL(trajSet, mdp, opts)

fprintf('- EM for IRL with multiple experts -\n');
dim   = mdp.nFeatures;
nTrajs   = size(trajSet, 1);
nrCl     = opts.nClust;
maxIters = opts.maxIters;

% count state-action pairs
trajInfo = cell(1, nTrajs);
for m = 1:nTrajs
    trajInfo{m} = getTrajInfo(trajSet(m, :, :), mdp);
end

% initialize
startT      = tic;
data.L      = -inf;
data.rho    = ones(nrCl, 1) / nrCl;
data.weight = nan(dim, nrCl);
for k = 1:nrCl
    data.weight(:, k) = sampleNewWeight(dim, opts);
end

[z, L]   = estep(data, trajInfo, mdp, opts);
elapsedT = toc(startT);
data.z = z;
data.L = L;
fprintf('%3d-th : %12.4f %6.2f sec\n', 0, L, elapsedT);
hst = saveHist(data, [], 1, elapsedT);

for iter = 1:maxIters
    [rho, weight]  = mstep(data, trajInfo, mdp, opts);
    newdata.rho    = rho;
    newdata.weight = weight;
    
    [z, L]    = estep(newdata, trajInfo, mdp, opts);
    elapsedT  = toc(startT);
    newdata.z = z;
    newdata.L = L;
    fprintf('%3d-th : %12.4f %6.2f sec\n', ...
        iter, newdata.L, elapsedT);
    hst = saveHist(data, hst, iter + 1, elapsedT);
    
    delta = newdata.L - data.L;
    if delta > 0
        data = newdata;
        if delta < 1e-4, break; end
    else
        break;
    end
    if elapsedT > opts.maxTime
        fprintf('# time out\n');
        break; 
    end
end

sol = data;
[~, sol.belongTo] = max(data.z, [], 2);
logPost = data.L;
fprintf('  '); 
printClustAssign(sol.belongTo);
fprintf('\n\n');

end



function hst = saveHist(data, hst, i, elapsedT)

[~, b] = max(data.z, [], 2);
hst.cl{i}.b     = b;
hst.cl{i}.w     = data.weight;
hst.logPost(i)  = data.L;
hst.elapsedT(i) = elapsedT;

end

