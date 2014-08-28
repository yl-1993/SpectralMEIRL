function [post, llh, prior] = calDPMTestLogPost(w, testTrajCnt, ...
    VL, szCl, wSet, mdp, opts)

alpha = opts.alpha;
eta   = opts.eta;
dim   = length(w);
nData = sum(szCl);

priorR = exp(calLogPrior(w, opts));
% priorR = 1;
% if strcmp(opts.priorType, 'HBIRL')
%     % Compute log prior and gradient of reward function w for HBIRL
%     % p(w | beta, g1, g2) = C \prod_{i=1}^d (2 g2+{beta w(i)^2}/{beta+1})^(-g1-0.5)    
%     priorR = (2*opts.gamma(2)+opts.beta.*w.^2./(opts.beta+1));
%     priorR = priorR.^(-opts.gamma(1)-0.5);
%     priorR = prod(priorR);
% 
% elseif strcmp(opts.priorType, 'Gaussian')
%     for d = 1:dim
%         priorR = priorR*normpdf(w(d), opts.mu, opts.sigma);
%     end
%     
% elseif strcmp(opts.priorType, 'Uniform')
% end

prior = priorR*alpha/(alpha + nData);
for m = 1:length(szCl)
    if isequal(w, wSet(:, m))
        prior = prior + szCl(m)/(alpha + nData);
    end
end
prior = 1;

mdp = convertW2R(w, mdp);
if isempty(VL)
    [~, VL] = policyIteration(mdp);
end

QL  = QfromV(VL, mdp, w);
BQ  = eta.*QL;
BQS = log(sum(exp(BQ), 2));
NBQ = bsxfun(@minus, BQ, BQS);

llh = 0;
for i = 1:size(testTrajCnt, 1)
    s   = testTrajCnt(i, 1);
    a   = testTrajCnt(i, 2);
    n   = testTrajCnt(i, 3);
    llh = llh + NBQ(s, a)*n;
end
prior = log(prior);
post  = prior + llh;

end