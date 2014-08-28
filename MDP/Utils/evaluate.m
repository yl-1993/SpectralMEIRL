function [V, H] = evaluate(piL, mdp, w)
%
% Evaluate policy measured on the given reward weight w
% V = (I-gamma*T^piL)^-1 R^piL
%
if nargin < 3 || isempty(w)
    w = mdp.weight;
end

nS = mdp.nStates;
nA = mdp.nActions;

if mdp.useSparse
    I = speye(nS);
    Tpi = sparse(nS, nS);
    for a = 1:nA
        idx = find(piL == a);
        if ~isempty(idx)
            Tpi(idx, :) = mdp.transitionS{a}(:, idx)';
        end
    end
    idx = (piL-1)*nS+(1:nS)';
    idx = (idx-1)*nS+(1:nS)';
    Epi = sparse(nS, nS*nA);
    Epi(idx) = 1;
else
    I = eye(nS);
    Tpi = zeros(nS, nS);
    for a = 1:nA
        idx = find(piL == a);
        if ~isempty(idx)
            Tpi(idx, :) = squeeze(mdp.transition(:, idx, a)');
        end
    end
    idx = (piL-1)*nS+(1:nS)';
    idx = (idx-1)*nS+(1:nS)';
    Epi = zeros(nS, nS*nA);
    Epi(idx) = 1;
end

if nS < 2^14
    H = (I-mdp.discount.*Tpi)\(Epi*mdp.F);
else
    EPS       = 1e-12;
    MAX_ITERS = 10^4;
    EF = Epi*mdp.F;
    H  = EF;
    for iter = 1:MAX_ITERS
        H_old = H;
        H = EF + mdp.discount*(Tpi*H);
        done = approxeq(H, H_old, EPS);
        if done, break; end
        if sum(isnan(H(:))) > 0, break; end
    end
end
V = H*w;

end