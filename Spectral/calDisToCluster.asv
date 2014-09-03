function [dist] = calDisToCluster(u)
    len = size(u,2);
    dist = zeros(len-1,1);
    newTraj = u(1,:);
    for i = 2:len
        dist(i-1) = getVecDistance(newTraj, u(i,:));
    end
end
