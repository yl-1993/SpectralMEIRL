function [dist] = calDisToCluster(u)
    len = size(u,2);
    dist = zeros(len-1,1);
    newTraj = u(1,:);
    for i = 2:len
        dist(i-1) = getVecDistance(newTraj, u(i,:));
    end
end

function [dist] = getVecDistance(uArr,vArr)
    len = min(size(uArr,2),size(vArr,2));
    dist = 0;
    for i = 1:len
        dist = dist + getDistance(uArr(i), vArr(i));
    end
end