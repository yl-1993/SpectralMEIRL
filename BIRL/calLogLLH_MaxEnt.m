% MaxEnt likelihood
% Ziebart et al., Maximum entropy inverse reinforcement learning, AAAI 2008
%
function [LLH, grad] = calLogLLH_MaxEnt(w, trajInfo, mdp)

[D, Da] = calExpFreq(w, mdp);

a = trajInfo.mu'*mdp.start*exp(w'*trajInfo.featExp);
b = exp(w'*mdp.F'*Da);
LLH = log(a/b);

if nargout >= 2
    grad = trajInfo.featExp - mdp.F'*Da;
end

end


% Calculate expected frequency
function [D, Da] = calExpFreq(w, mdp)

EPS       = 1e-12;
MAX_ITERS = 10^5;

nS = mdp.nStates;
nA = mdp.nActions;
mdp = convertW2R(w, mdp);

% Value iteration for E[exp(\sum_t gamma^t R(s_t,a_t))]
V = zeros(nS, 1);
Q = zeros(nS, nA);
for t = 1:MAX_ITERS
    V_old = V;
    for a = 1:nA
        if mdp.useSparse
            Q(:, a) = mdp.rewardS{a} ...
                + mdp.discount*(mdp.transitionS{a}(:, :)'*V);
        else
            Q(:, a) = mdp.reward(:, a) ...
                + mdp.discount*(mdp.transitionS{a}(:, :)'*V);
        end
    end
    V = log(sum(exp(Q), 2));
    if approxeq(V, V_old, EPS), break; end
end

% Local action probability
Pa = exp(bsxfun(@minus, Q, V));

% Compute expected visitation counts
D = zeros(nS, 1);
T = sparse(reshape(mdp.transition, nS, nS*nA));
for t = 1:MAX_ITERS
    D_old = D;
    D = mdp.start + mdp.discount*T*reshape(bsxfun(@times, Pa, D), nS*nA, 1);
    if approxeq(D, D_old, EPS), break; end
end
Da = reshape(bsxfun(@times, Pa, D), nS*nA, 1);

end
