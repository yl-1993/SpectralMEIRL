function mdp = convertW2R(weight, mdp)

if mdp.useSparse
    mdp.weight = sparse(weight);
else
    mdp.weight = weight;
end

mdp.reward = reshape(mdp.F*weight, mdp.nStates, mdp.nActions);
if mdp.useSparse
    for a = 1:mdp.nActions
        mdp.rewardS{a} = sparse(mdp.reward(:, a));
    end
end

end