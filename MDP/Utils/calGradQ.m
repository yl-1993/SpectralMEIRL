% Compute gradient of Q-function

function dQ = calGradQ(piL, mdp)

nS = mdp.nStates;
nA = mdp.nActions;

% calculate dQ/dw
Epi = sparse(nS, nS*nA);
if size(piL, 2) == 1    % deterministic policy
    idx = (piL-1)*nS+(1:nS)';
    idx = (idx-1)*nS+(1:nS)';
    Epi(idx) = 1;
else                    % stochastic policy
    for s = 1:nS
        for a = 1:nA
            Epi(s, (a - 1)*nS + s) = piL(s, a);
        end
    end
end

if mdp.nStates < 2^10
    dQ = (eye(nS*nA)-mdp.T*Epi)\mdp.F;
else
    EPS       = 1e-12;
    MAX_ITERS = 10^4;
    dQ = zeros(nS*nA, mdp.nFeatures);
    T  = mdp.T*Epi;
    for iter = 1:MAX_ITERS
        dQ_old = dQ;
        dQ = mdp.F + (T*dQ);
        done = approxeq(dQ, dQ_old, EPS);
        if done, break; end
    end
end
dQ = dQ';


end