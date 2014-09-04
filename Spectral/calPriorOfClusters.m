function priorMatrix = calPriorOfClusters(mFeatExp, nClusters)
    priorMatrix = eye(nClusters);
    for i = 1:nClusters
        for j = 1:nClusters
            priorMatrix(j,i) = getVecDistance(mFeatExp(i,:),mFeatExp(j,:));
        end
    end
    for i = 1:nClusters
        priorMatrix(:,i) = priorMatrix(:,i)/sum(priorMatrix(:,i));
    end
end