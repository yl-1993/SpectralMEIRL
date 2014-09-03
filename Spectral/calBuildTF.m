function TF = calBuildTF(trajs, mdp)
%
% Compute occupancy measure and build trajectory*feature matrix.
%
nS = mdp.nStates;
nA = mdp.nActions;
nF = mdp.nFeatures;
TF.nTrajs = size(trajs, 1);
TF.nSteps = size(trajs, 2);
cnt = zeros(nS, nA);
occ = zeros(nS, nA);
featExp = zeros(TF.nTrajs, nF);
nSteps = 0;
for m = 1:TF.nTrajs
    occ_each = zeros(nS,nA);
    for h = 1:TF.nSteps
        s = trajs(m, h, 1);
        a = trajs(m, h, 2);
        if s == -1 && a == -1
            break;
        end
        cnt(s, a) = cnt(s, a)+1;
        occ(s, a) = occ(s, a)+mdp.discount^(h-1);
        occ_each(s, a) = occ_each(s, a)+mdp.discount^(h-1);
        nSteps    = nSteps + 1;
    end
    featExp(m,:) = mdp.F'*occ_each(:);
end
piL             = cnt./repmat(sum(cnt, 2), 1, nA);
piL(isnan(piL)) = 0;
mu  = sum(cnt, 2)/nSteps; %trajInfo.nTrajs/trajInfo.nSteps;
occ = occ./TF.nTrajs;

TF.v   = mdp.reward(:)'*occ(:);
TF.pi  = piL;                         % empirical estimate of policy
TF.mu  = mu;                          % state visitation
TF.occ = occ;                         % discounted state-action frequency
TF.featExp = featExp;
TF.stdFeatExp = mdp.F'*TF.occ(:);  % feature expectation
%trajInfo.featExp = mdp.F'*trajInfo.occ(:);  % feature expectation

N = nnz(cnt);
TF.cnt = zeros(N, 3);                 % vector of state-action count
i = 1;
for s = 1:nS
    for a = 1:nA
        if cnt(s, a) > 0
            TF.cnt(i, 1) = s;
            TF.cnt(i, 2) = a;
            TF.cnt(i, 3) = cnt(s, a);
            i = i+1;
        end
    end
end

end %end_of_calBuildTF