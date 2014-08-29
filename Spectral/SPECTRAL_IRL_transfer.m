% Information transfer to new trajectory using the results from Spectral
%
function wL = SPECTRAL_IRL_transfer(sol, traj, mdp, opts)
% number of clusters
nClusters = sol.nClusters;
% feature expectation for new trajectory
tfVec = calBuildTF(traj, mdp);
% build new trajectory*feature matrix by new traj and merge cluster feature 
newTF = zeros(nClusters+1, size(tfVec.featExp,2));
newTF(1,:) = tfVec.featExp;
for i = 1:nClusters
    newTF(i+1,:) = sol.mFeatExp(i,:);
end
% project to SVD space
trajInfo = calSVD(newTF);
% get the distance of new traj to every clusters
dist = calDisToCluster(trajInfo.u);
% get a mix reward function
wL = sol.weight*(dist./sum(dist));

end
%trajInfo = getTrajInfo(traj, mdp);
% nrCl = length(sol.rho);
% dist = zeros(nrCl, 1);
% for k = 1:nrCl
%     if sol.rho(k) > 0
%         w = sol.weight(:, k);
%         llh = calLogLLH(w, trajInfo, [], [], [], mdp, opts);        
%         dist(k) = sol.rho(k)*exp(llh)
%     end
% end
% wL = sol.weight*(dist./sum(dist))
