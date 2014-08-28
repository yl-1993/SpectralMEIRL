function [evd, fsc, nmi, nrCl, tm, expData] = plot_results_cputime()

configPath([], true);

probName = 'gridworld';
outpath  = 'Results_140822';
MAX_TIME = inf; %1000;
INTERVAL = 5;

k = 1;
% alg{k}.name    = 'Ind_BIRL';
% alg{k}.llhType   = 'BIRL';
% alg{k}.priorType = 'Uniform';
% alg{k}.lw = 1;
% alg{k}.lt = ':';
% k = k + 1;
% 
alg{k}.name    = 'EM_IRL';
alg{k}.nClust  = 1;
alg{k}.llhType   = 'MLIRL';
alg{k}.priorType = 'Uniform';
alg{k}.lw = 2;
alg{k}.lt = '--';
k = k + 1;

alg{k}.name    = 'EM_IRL';
alg{k}.nClust  = 2;
alg{k}.llhType   = 'MLIRL';
alg{k}.priorType = 'Uniform';
alg{k}.lw = 2;
alg{k}.lt = '--';
k = k + 1;

alg{k}.name    = 'EM_IRL';
alg{k}.nClust  = 3;
alg{k}.llhType   = 'MLIRL';
alg{k}.priorType = 'Uniform';
alg{k}.lw = 2;
alg{k}.lt = '--';
k = k + 1;

% alg{k}.name    = 'DPM_BIRL_MH';
% alg{k}.llhType   = 'BIRL';
% alg{k}.priorType = 'NG';
% alg{k}.lw = 2;
% alg{k}.lt = '-';
% k = k + 1;

alg{k}.name    = 'DPM_BIRL_MHL';
alg{k}.llhType   = 'BIRL';
alg{k}.priorType = 'NG';
alg{k}.lw = 2;
alg{k}.lt = '-';

problem  = problemParamsME(probName);
probName = getProblemName(problem);
outpath  = sprintf('./%s/%s', outpath, probName);

nAlgs   = length(alg);
nExps   = problem.nExps;
expData = cell(nAlgs, nExps);
minTime = inf;
maxTime = -inf;
for k = 1:nAlgs
    algName = getAlgName(alg{k});
    
    outfname = sprintf('%s/%s_hst1.mat', outpath, algName);
    fprintf('Read %s\n', outfname);
    load(outfname);
    
    for i = 1:nExps
        fprintf('Compute EVD for exp. %d\n', i);
        problem.seed = problem.initSeed + i;
        mdp = generateProblem(problem, problem.initSeed, problem.discount);
        
        input  = hst1{i}.data;
        nTrajs = size(input.trajSet, 1);
        vE = zeros(nTrajs, 1);
        for m = 1:nTrajs
            kE = input.trajId(m);
            wE = input.weight(:, kE);
            pE = input.policy(:, kE);
            [VE, HE] = evaluate(pE, mdp, wE);
            vE(m) = full(wE'*HE'*mdp.start);
        end
        
        if strcmp(alg{k}.name, 'Ind_BIRL')
            sol  = hst1{i}.sol;
            cl.b = sol.belongTo;
            cl.w = sol.weight;
            expData{k, i}.evd  = calEVD(input, vE, cl, mdp);
            expData{k, i}.fsc  = calFscore(input, sol.belongTo);
            expData{k, i}.nmi  = calNMI(input, sol.belongTo);
            expData{k, i}.nrCl = countClust(sol.belongTo);
            expData{k, i}.tm   = 0;
        else            
            tmp   = hst1{i}.hst;
            maxPr = -inf;
            T     = length(tmp.elapsedT);
            expData{k, i}.evd  = zeros(T, 1);
            expData{k, i}.fsc  = zeros(T, 1);
            expData{k, i}.nmi  = zeros(T, 1);
            expData{k, i}.nrCl = zeros(T, 1);
            for t = 1:T
                if maxPr < tmp.logPost(t)
                    maxPr = tmp.logPost(t);
                    expData{k, i}.evd(t)  = calEVD(input, vE, tmp.cl{t}, mdp);
                    expData{k, i}.fsc(t)  = calFscore(input, tmp.cl{t}.b);
                    expData{k, i}.nmi(t)  = calNMI(input, tmp.cl{t}.b);
                    expData{k, i}.nrCl(t) = countClust(tmp.cl{t}.b);
                else
                    expData{k, i}.evd(t)  = expData{k, i}.evd(t - 1);
                    expData{k, i}.fsc(t)  = expData{k, i}.fsc(t - 1);
                    expData{k, i}.nmi(t)  = expData{k, i}.nmi(t - 1);
                    expData{k, i}.nrCl(t) = expData{k, i}.nrCl(t - 1);
                end                
            end
            expData{k, i}.tm = tmp.elapsedT';
            minTime = min(minTime, min(tmp.elapsedT));
            maxTime = max(maxTime, max(tmp.elapsedT));
        end
    end
end

fprintf('Time interval: %f - %f\n', minTime, maxTime);
maxTime = min(maxTime, MAX_TIME);
tm = minTime: INTERVAL: maxTime;
nTimes = length(tm);

fprintf('Compute mean and standard error\n');
evd.mu = nan(nAlgs, nTimes);
evd.se = nan(nAlgs, nTimes);
fsc.mu = nan(nAlgs, nTimes);
fsc.se = nan(nAlgs, nTimes);
nmi.mu = nan(nAlgs, nTimes);
nmi.se = nan(nAlgs, nTimes);
nrCl.mu = nan(nAlgs, nTimes);
nrCl.se = nan(nAlgs, nTimes);

for k = 1:nAlgs    
    for j = 1:nTimes
        tmp.evd  = nan(problem.nExps, 1);
        tmp.fsc  = nan(problem.nExps, 1);
        tmp.nmi  = nan(problem.nExps, 1);
        tmp.nrCl = nan(problem.nExps, 1);
        tmp.cnt  = 0;
        for i = 1:problem.nExps
            m = find(expData{k, i}.tm < tm(j), 1, 'last');
            if ~isempty(m)
                tmp.evd(i)  = expData{k, i}.evd(m);
                tmp.fsc(i)  = expData{k, i}.fsc(m);
                tmp.nmi(i)  = expData{k, i}.nmi(m);
                tmp.nrCl(i) = expData{k, i}.nrCl(m);
                tmp.cnt     = tmp.cnt + 1;
            end
        end
        evd.mu(k, j)  = mean(tmp.evd);
        evd.se(k, j)  = sqrt(var(tmp.evd)./tmp.cnt);
        fsc.mu(k, j)  = mean(tmp.fsc);
        fsc.se(k, j)  = sqrt(var(tmp.fsc)./tmp.cnt);
        nmi.mu(k, j)  = mean(tmp.nmi);
        nmi.se(k, j)  = sqrt(var(tmp.nmi)./tmp.cnt);
        nrCl.mu(k, j) = mean(tmp.nrCl);
        nrCl.se(k, j) = sqrt(var(tmp.nrCl)./tmp.cnt);
    end
end

fprintf('Plot results\n');

scrsz = get(0,'ScreenSize');
fig   = figure('Position',[1 scrsz(4)/2 1600 400]);
bErr  = true;
xstr  = 'cpu time (sec)';

makeSubplot(1, 4, 1);
plotResults(tm, evd.mu, evd.se, alg, xstr, 'EVD', probName, bErr);

makeSubplot(1, 4, 2);
plotResults(tm, fsc.mu, fsc.se, alg, xstr, 'F-score', [], bErr);

makeSubplot(1, 4, 3);
plotResults(tm, nmi.mu, nmi.se, alg, xstr, 'NMI', [], bErr);

makeSubplot(1, 4, 4);
plotResults(tm, nrCl.mu, nrCl.se, alg, xstr, '# of clusters', [], bErr, true);

inpath = 'Results';
infname = sprintf('./%s/%s.fig', inpath, probName);
saveas(fig, infname);

configPath([], false);

end


% function makeSubplot(r, c, k)
% 
% subaxis(r, c, k, 'SpacingHoriz', 0.01, 'SpacingVert', 0, ...
%     'PaddingRight', 0.02, 'PaddingLeft', 0.02, ...
%     'PaddingTop', 0.0, 'PaddingBottom', 0.0, ...
%     'MarginRight', 0.02, 'MarginLeft', 0.02, ...
%     'MarginTop', 0.08, 'MarginBottom', 0.1);
% 
% end
% 
% function plotResults(X, mu, se, alg, ystr, tstr, bErr, bLegend)
% 
% nAlgs = length(alg);
% mm = '+o*xsd^v><ph.';
% cc = hsv(nAlgs);
% hold on;
% 
% % idx = 1:10:length(X);
% % for k = 1:nAlgs
% %     plot(X(idx), mu(idx, k), 'Color', cc(k, :), 'LineStyle', mm(k));
% % end
% for k = 1:nAlgs
%     plot(X, mu(:, k), 'Color', cc(k, :), 'LineWidth', 2);
% end
% 
% if nargin > 7 && bLegend
%     maxStrLen = 0;
%     buf = [];
%     for k = 1:nAlgs
%         tmp = getAlgName(alg{k});
%         tmp(tmp == '_') = '-';
%         buf{k} = tmp;
%         maxStrLen = max(maxStrLen, length(buf{k}));
%     end
%     legendStr = [];
%     for k = 1:nAlgs
%         str = buf{k};
%         n   = length(buf{k});
%         str(n + 1:maxStrLen) = ' ';
%         legendStr = cat(1, legendStr, str);
%     end
%     legend(legendStr, 'Location', 'NorthEast'); %'SouthEast');
% end
% 
% % for k = 1:nAlgs
% %     plot(X, mu(:, k), 'Color', cc(k, :));
% % end
% 
% if nargin > 6 && bErr
%     err1 = mu + se;
%     err2 = mu - se;
%     for k = 1:nAlgs
%         plot(X, err1(:, k), 'Color', cc(k, :), 'LineStyle', '--');
%         plot(X, err2(:, k), 'Color', cc(k, :), 'LineStyle', '--');
%     end
% end
% 
% xlabel('cpu time (sec)');
% ylabel(ystr);
% tstr(tstr == '_') = ' ';
% title(tstr);
% 
% end
% 
% 
% function nrCl = countClust(belongTo)
% 
% N = max(belongTo);
% L = zeros(N, 1);
% for k = 1:N
%     L(k) = nnz(belongTo == k);
% end
% nrCl = nnz(L);
% 
% end
% 
% 
% function evd = calEVD(input, vE, cl, mdp)
% 
% M = length(cl.b);
% vdiff = nan(M, 1);
% for m = 1:M
%     kE = input.trajId(m);
%     wE = input.weight(:, kE);
%     
%     kL = cl.b(m);
%     wL = cl.w(:, kL);
%     if isfield(cl, 'p') && ~isempty(cl.p)
%         pL = cl.p(:, kL);
%     else
%         mdp = convertW2R(wL, mdp);
%         pL  = policyIteration(mdp);
%     end
%     [VL, HL] = evaluate(pL, mdp, wL);
%     vL = full(wE'*HL'*mdp.start);
%     
%     vdiff(m) = vE(m) - vL;
% end
% evd = mean(vdiff);
% 
% end
% 
% 
% function fsc = calFscore(input, belongTo)
% 
% nExperts = size(input.weight, 2);
% nrCl = max(belongTo);
% M  = length(belongTo);
% tp = zeros(nExperts, 1);
% fp = zeros(nExperts, 1);
% fn = zeros(nExperts, 1);
% tn = zeros(nExperts, 1);
% 
% for u = 1:nExperts
%     idx = input.trajId == u;
%     tmp = belongTo(idx);
%     cnt = zeros(nrCl, 1);
%     for m = 1:nrCl
%         cnt(m) = nnz(tmp == m);
%     end
%     [v, w] = max(cnt);
%     tp(u) = v;
%     fp(u) = nnz(belongTo == w) - v;
%     fn(u) = nnz(idx) - v;
%     tn(u) = M - (tp(u) + fp(u) + fn(u));
% end
% 
% precision = sum(tp) / (sum(tp) + sum(fp));
% recall    = sum(tp) / (sum(tp) + sum(fn));
% fsc       = 2*precision*recall/(precision + recall);
% 
% end
% 
% 
% function nmi = calNMI(input, belongTo)
% 
% nExperts = size(input.weight, 2);
% nrCl = max(belongTo);
% M    = length(belongTo);
% 
% nc = zeros(nExperts, 1);
% for j = 1:nExperts
%     nc(j) = nnz(input.trajId == j);
% end
% 
% nw = zeros(nrCl, 1);
% for u = 1:nrCl
%     nw(u) = nnz(belongTo == u);
% end
% 
% nwc = zeros(nrCl, nExperts);
% for u = 1:nrCl
%     idx = belongTo == u;
%     for w = 1:nExperts
%         q = nnz(input.trajId(idx) == w);
%         if q > 0
%             nwc(u, w) = q/M*log(M*q/nw(u)/nc(w));
%         end
%     end
% end
% 
% q = nw./M;
% q = q(nw > 0);
% entw = -sum(q.*log(q));
% 
% q = nc./M;
% entc = -sum(q.*log(q));
% 
% nmi = 2*sum(nwc(:))/(entw + entc);
% 
% end
