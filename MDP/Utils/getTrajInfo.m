function trajInfo = getTrajInfo(trajs, mdp)
%
% Compute occupancy measure and empirical policy for trajectories.
%
nS = mdp.nStates;
nA = mdp.nActions;

trajInfo.nTrajs = size(trajs, 1);
trajInfo.nSteps = size(trajs, 2);
cnt = zeros(nS, nA);
occ = zeros(nS, nA);
nSteps = 0;
for m = 1:trajInfo.nTrajs
    for h = 1:trajInfo.nSteps
        s = trajs(m, h, 1);
        a = trajs(m, h, 2);
        if s == -1 && a == -1
            break;
        end
        cnt(s, a) = cnt(s, a)+1;
        occ(s, a) = occ(s, a)+mdp.discount^(h-1);
        nSteps    = nSteps + 1;
    end
end
piL             = cnt./repmat(sum(cnt, 2), 1, nA);
piL(isnan(piL)) = 0;
mu  = sum(cnt, 2)/nSteps; %trajInfo.nTrajs/trajInfo.nSteps;
occ = occ./trajInfo.nTrajs;

trajInfo.v   = mdp.reward(:)'*occ(:);
trajInfo.pi  = piL;                         % empirical estimate of policy
trajInfo.mu  = mu;                          % state visitation
trajInfo.occ = occ;                         % discounted state-action frequency
trajInfo.featExp = mdp.F'*trajInfo.occ(:);  % feature expectation

N = nnz(cnt);
trajInfo.cnt = zeros(N, 3);                 % vector of state-action count
i = 1;
for s = 1:nS
    for a = 1:nA
        if cnt(s, a) > 0
            trajInfo.cnt(i, 1) = s;
            trajInfo.cnt(i, 2) = a;
            trajInfo.cnt(i, 3) = cnt(s, a);
            i = i+1;
        end
    end
end

end %end_of_getTrajInfo