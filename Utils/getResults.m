function results = getResults(hist)

[MR, VR, ER] = getStat(hist, 'rewardDiff');
[MP, VP, EP] = getStat(hist, 'policyDiff');
[MV, VV, EV] = getStat(hist, 'valueDiff');

results = struct('MR', MR, 'VR', VR, 'ER', ER, ...
    'MP', MP, 'VP', VP, 'EP', EP, ...
    'MV', MV, 'VV', VV, 'EV', EV);

if isfield(hist{1}, 'logPost')
    [m, v, e] = getStat(hist, 'logPost');
    results.MK = m;
    results.VK = v;
    results.EK = e;
end

if isfield(hist{1}, 'precision')
    [m, v, e] = getStat(hist, 'precision');
    results.MPRC = m;
    results.VPRC = v;
    results.EPRC = e;
end

if isfield(hist{1}, 'recall')
    [m, v, e] = getStat(hist, 'recall');
    results.MRCL = m;
    results.VRCL = v;
    results.ERCL = e;
end

if isfield(hist{1}, 'fscore')
    [m, v, e] = getStat(hist, 'fscore');
    results.MF = m;
    results.VF = v;
    results.EF = e;
end

if isfield(hist{1}, 'nmi')
    [m, v, e] = getStat(hist, 'nmi');
    results.MNMI = m;
    results.VNMI = v;
    results.ENMI = e;
end

if isfield(hist{1}, 'nClusters')
    [m, v, e] = getStat(hist, 'nClusters');
    results.MNCL = m;
    results.VNCL = v;
    results.ENCL = e;
end

end



function [m, v, e] = getStat(hist, fname)

n = length(hist);
tmp = nan(n, 1);
for i = 1:n
    tmp(i) = getfield(hist{i}, fname);
end
m = mean(tmp);
v = var(tmp);
e = sqrt(v/n);

end