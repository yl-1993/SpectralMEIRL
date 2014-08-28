function Q = QfromV(V, mdp, w)

nS = mdp.nStates;
nA = mdp.nActions;

if nargin < 3 || isempty(w)
    w = mdp.weight;
else
    mdp = convertW2R(w, mdp);
end

if mdp.useSparse
    Q = sparse(nS, nA);
    for a = 1:nA
        Q(:, a) = mdp.rewardS{a} + mdp.discount*(mdp.transitionS{a}(:, :)'*V);
    end
else
    Q = zeros(nS, nA);
    for a = 1:nA
        Q(:, a) = mdp.reward(:, a) ...
            + mdp.discount*(squeeze(mdp.transition(:, :, a))'*V);
    end
end

end