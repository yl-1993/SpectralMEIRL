% Information transfer to new trajectory using  MAP inference for BIRL
%
function [wL, hist, maxData] = Ind_BIRL_transfer(sol, traj, mdp, opts)

wL = MAP_BIRL(traj, [], mdp, opts);
hist = [];
maxData = [];

end