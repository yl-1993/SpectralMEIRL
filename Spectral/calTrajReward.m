function trajReward = calTrajReward(priorMatrix, mFeatExp)
    % try to maximize the direction of each reward
    trajReward = priorMatrix*pinv(mFeatExp)'
    % Probabilistic Matrix Factorization
    [trajReward, rewardFeature] = calPMF(mFeatExp);
    trajReward = normalization(trajReward)
end

function trajReward = normalization(trajReward)
    len = size(trajReward,1);
    for i = 1:len
        trajReward(i,:) = trajReward(i,:)/sum(abs(trajReward(i,:)));
    end
end