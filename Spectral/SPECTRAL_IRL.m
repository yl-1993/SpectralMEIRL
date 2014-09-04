% Spectral method for Multiple Expert IRL
%
function [sol] = SPECTRAL_IRL(trajs, mdp, irlOpts)

%nClusters = irlOpts.nClusters;
nTrajs = size(trajs, 1);

sol.weight   = [];

TF = calBuildTF(trajs, mdp);
%[TF.featExp,W] = whiten(TF.featExp)
trajInfo = calSVD(TF.featExp,1);
%wL = trajInfo.v(1,:)'

%TODO: cluster using TF*v
%TF.featExp*trajInfo.v
[sol.belongTo, nClusters] = calCluster(trajInfo.u,0.1);
%[sol.belongTo, nClusters] = calCluster(TF.featExp*trajInfo.v,0.08);
sol.nClusters = nClusters;

% the following code give the right clusters
% nClusters = 3;
% clusterId  =[];
% trajNum = nTrajs/nClusters;
% for i  = 1:nClusters
%     clusterId  = cat(1, clusterId, repmat(i, trajNum, 1));
% end
% sol.belongTo = clusterId;
%

% merge feature expectation info
mFeatExp = calMergeFeatExp(trajInfo.featExp,sol.belongTo, nClusters);
% only use main energy of mFeatExp to estimate reward
% tmp = calSVD(mFeatExp,0.9);
% mFeatExp = tmp.featExp;
sol.mFeatExp = mFeatExp;
% prior matrix
% priorMatrix = eye(nClusters);
% p1 = 0.75;
% p2 = (1-p1)/(nClusters-1);
% for i = 1:nClusters
%     for j = 1:nClusters
%         if i == j
%             priorMatrix(i,j) = p1;
%         else
%             priorMatrix(i,j) = p2;
%         end
%     end
% end
priorMatrix = calPriorOfClusters(mFeatExp, nClusters);
%priorMatrix = [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8];
%trajRewawrd = priorMatrix*pinv(mFeatExp)';

trajRewawrd = calTrajReward(priorMatrix, mFeatExp);
for m = 1:nClusters
    wL = trajRewawrd(m,:)';
    sol.weight = cat(2, sol.weight, wL);
end

end