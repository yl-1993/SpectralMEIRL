function [mFeatExp] = calMergeFeatExp(trajFeature, belongTo, nClusters)
    nTrajs = size(belongTo,2);
    nFeatures = size(trajFeature,2);
    mFeatExp = zeros(nClusters, nFeatures);
    for i = 1:nClusters
        mergeArr = (belongTo == i);
        for j = 1:nTrajs
            if mergeArr(j) == 1
                mFeatExp(i,:) = mFeatExp(i,:) + trajFeature(j,:);
            end
        end
        mFeatExp(i) = mFeatExp(i)/nnz(mergeArr); % works better without
        %doing such normalization
    end
end