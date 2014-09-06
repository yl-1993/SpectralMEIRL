function trajReward = calTrajReward(priorMatrix, mFeatExp)
    % normalize the mFeatMatrix
    mFeatExp = normMatrix(mFeatExp,'row');
    % try to maximize the direction of each reward
    trajReward = priorMatrix*pinv(mFeatExp)';
    % Probabilistic Matrix Factorization
    %mFeatExp = pinv(mFeatExp)';
    %[u, s, v] = svd(mFeatExp);
    %trajReward = v(1:size(mFeatExp,1),:);
    %trajReward = v(:,1:size(mFeatExp,1))';
    %[trajReward, rewardFeature] = calPMF(mFeatExp);
    %trajReward = trajReward*rewardFeature;
    %trajReward = pinv(trajReward)';
    %trajReward = softmax(trajReward);
    trajInfo = calSVD(trajReward, 0.85);
    trajReward = trajInfo.featExp;
    
    trajReward = normMatrix(trajReward,'row')
    %trajReward = softmax(trajReward);

end

function trajReward = normMatrix(trajReward, option)
    % based on row or col to normalize a matrix
    if option == 'row'
        len = size(trajReward,1);
        for i = 1:len
            trajReward(i,:) = trajReward(i,:)/max(abs(trajReward(i,:)));
        end
    elseif option == 'col'
        len = size(trajReward,2);
        for i = 1:len
            trajReward(:,i) = trajReward(:,i)/max(abs(trajReward(:,i)));
        end
    else
        trajReward = trajReward;
    end
end

function trajReward = softmax(trajReward)
    % reduce the value difference among reward weights
    % do softmax for each cluster reward weights
    [nr,nc] = size(trajReward);
    for i = 1:nr
        maxe = max(trajReward(i,:));
        for j = 1:nc
            trajReward(i,j) = exp(trajReward(i,j) - maxe);
        end
        trajReward(i,:) = trajReward(i,:)/sum(trajReward(i,:));
    end
end