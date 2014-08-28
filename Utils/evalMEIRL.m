function results = evalMEIRL(sol, data, mdp, problem)

nExperts = problem.nExperts;
nTrajs   = size(data.trajSet, 1);
% results  = struct('rewardDiff', {}, 'policyDiff', {}, 'valueDiff', {}, ...
%     'recall', {}, 'precision', {}, 'fscore', {}, ...
%     'nClusters', {}, 'bts', {});

rdiff = nan(nTrajs, 1);
pdiff = nan(nTrajs, 1);
vdiff = nan(nTrajs, 1);
for m = 1:nTrajs
    kE = data.trajId(m);
    wE = data.weight(:, kE);
    
    kL = sol.belongTo(m);
    wL = sol.weight(:, kL);
    
    r = evalSEIRL(wL, wE, mdp);
    rdiff(m) = r.rewardDiff;
    pdiff(m) = r.policyDiff;
    vdiff(m) = r.valueDiff;
end
results.rewardDiff = mean(rdiff);
results.policyDiff = mean(pdiff);
results.valueDiff  = mean(vdiff);


% compute precision and recall
nrCl = max(sol.belongTo);
tp = zeros(nExperts, 1);
fp = zeros(nExperts, 1);
fn = zeros(nExperts, 1);
tn = zeros(nExperts, 1);

for k = 1:nExperts
    idx = data.trajId == k;
    tmp = sol.belongTo(idx);
    cnt = zeros(nrCl, 1);
    for m = 1:nrCl
        cnt(m) = nnz(tmp == m);
    end
    [v, i] = max(cnt);
    tp(k) = v;
    fp(k) = nnz(sol.belongTo == i) - v;
    fn(k) = nnz(idx) - v;
    tn(k) = nTrajs - (tp(k) + fp(k) + fn(k));
end

results.precision = sum(tp) / (sum(tp) + sum(fp));
results.recall    = sum(tp) / (sum(tp) + sum(fn));

% compute f-score
results.fscore = 2*results.precision*results.recall;
results.fscore = results.fscore / (results.precision + results.recall);


% compute normlized mutual information (NMI)
nrCl = max(sol.belongTo);

nc = zeros(nExperts, 1);
for j = 1:nExperts
    nc(j) = nnz(data.trajId == j);
end

nw = zeros(nrCl, 1);
for k = 1:nrCl
    nw(k) = nnz(sol.belongTo == k);
end

nwc = zeros(nrCl, nExperts);
for k = 1:nrCl
    idx = sol.belongTo == k;
    for j = 1:nExperts
        x = nnz(data.trajId(idx) == j);
        if x > 0
            nwc(k, j) = x/nTrajs*log(nTrajs*x/nw(k)/nc(j));
        end
    end
end

x = nw./nTrajs;
x = x(nw > 0);
entw = -sum(x.*log(x));

x = nc./nTrajs;
entc = -sum(x.*log(x));

results.nmi = 2*sum(nwc(:))/(entw + entc);

    
% count # of clusters
L = zeros(nrCl, 1);
for k = 1:nrCl
    L(k) = nnz(sol.belongTo == k);
end
results.nClusters = nnz(L);


% between-task similarity record
bts = zeros(nTrajs, nTrajs);
for i = 1:nTrajs
    idx = sol.belongTo == sol.belongTo(i);
    bts(i, idx) = bts(i, idx) + 1;
end
results.bts = bts;
        

