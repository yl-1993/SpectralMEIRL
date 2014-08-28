function mdp = randomMDP(nStates, nActions, nFeatures, discount)

mdp.bFactored = false;
mdp.nStates   = nStates;
mdp.nActions  = nActions;
mdp.nFeatures = nFeatures;
mdp.useSparse = true;
mdp.discount  = discount;
DENSITY = min(2, log(nStates)/log(2))/nStates;

% transition probabilities T(s',s,a) = P(s'|s,a)
mdp.transition = zeros(nStates, nStates, nActions);
for a = 1:nActions
    for s = 1:nStates
        while sum(mdp.transition(:, s, a)) <= 0
            mdp.transition(:, s, a) = sprand(1, nStates, DENSITY);
        end
        mdp.transition(:, s, a) = mdp.transition(:, s, a)./sum(mdp.transition(:, s, a));
    end
end
mdp.transition = full(mdp.transition);

mdp.F = zeros(nStates*nActions, nFeatures);
for i = 1:nStates*nActions
    j = randi(nFeatures);
    mdp.F(i, j) = 1;
end

LB = 0.5;
UB = 1.5;
range      = min(UB, max(LB, randn(nFeatures, 1)+(LB+UB)/2));
ratio      = rand(nFeatures, 1);
mdp.weight = rand(nFeatures, 1);
mdp.lb     = mdp.weight-(range.*ratio);
mdp.ub     = mdp.lb+range;
mdp.reward = reshape(mdp.F*mdp.weight, nStates, nActions);
mdp.start = rand(nStates, 1);
mdp.start = mdp.start./sum(mdp.start);
mdp.name = sprintf('randomMdp_s%d_a%d_f%d', nStates, nActions, nFeatures);

mdp.F      = sparse(mdp.F);
mdp.weight = sparse(mdp.weight);
mdp.lb     = sparse(mdp.lb);
mdp.ub     = sparse(mdp.ub);
mdp.start  = sparse(mdp.start);

for a = 1:mdp.nActions
    mdp.transitionS{a} = sparse(mdp.transition(:, :, a));
    mdp.rewardS{a} = sparse(mdp.reward(:, a));
end

end