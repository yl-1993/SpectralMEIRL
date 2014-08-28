function [wL] = RANDOM_IRL(trajs, mdp, irlOpts)

nF = mdp.nFeatures;
weight = zeros(nF, 1);
l = randperm(nF - 1);
k = ceil(0.3*nF);
% k = ceil(log(nF));
idx = l(1:k);
weight(idx) = rand(k, 1) - 1;
weight(end) = 1;

rand('state',sum(100*clock));
wL = 2*rand(nF, 1) - 1
%wL = weight;

end