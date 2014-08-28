% IRL with multiple experts using MAP inference for BIRL on each trajectory
% independently
%
function [sol, logPost] = Ind_BIRL(trajSet, mdp, opts)

sol.weight   = [];
sol.belongTo = 1:size(trajSet, 1);
for m = 1:size(trajSet, 1)
    wL = MAP_BIRL(trajSet(m, :, :), [], mdp, opts);
    sol.weight = cat(2, sol.weight, wL);
end
logPost = [];

end

