function [belongTo, nClusters] = calCluster(TR, rate)
	%nClusters = 3;
    %scale = 8; %TODO: real time
    belongTo = zeros(1,size(TR,1));
    i = 1;
    while isEmpty(belongTo)
        belongTo = getClusterFromCol(TR, belongTo, i, rate);
        i = i + 1;
    end
    nClusters = i - 1;
end

function [flag] = isEmpty(belongTo)
    len = size(belongTo,2);
    for i = 1:len
        if belongTo(i) == 0
            flag = 1;
            return;
        end
    end
    flag = 0
end

function [belongTo] = getClusterFromCol(TR, belongTo, nClust, rate)
    col = TR(:,nClust)
    len = size(col);
    dist.data = zeros(1,len);
    % remove the clustered point
    for i=1:len
        if belongTo(i) ~= 0
            col(i) = -realmax;
        end
    end
    % max value of remained points
    maxElement = max(col);
    maxDist = 0;
    minDist = realmax;
    % distance of all remained points
    for i = 1:len
        dist.index(i) = i;
        if belongTo(i) == 0
            dist.data(i) = getDistance(col(i),maxElement);
            if dist.data(i) > maxDist
                maxDist = dist.data(i);
            end
            if dist.data(i) < minDist && dist.data(i) > 0
                minDist = dist.data(i);
            end
        else
            dist.data(i) = realmax;
        end
    end
    % compute scale
    scale = maxDist/minDist;
    % TODO: HOW TO DEFINE 12
    % if scale < 12, all data should belong to the same group
    if scale < 20
        % the inner difference is not big enough
        for i = 1:len
          if belongTo(i) == 0
            belongTo(i) = nClust;
          end
        end
    else
        % sum of the remained point value
        dist_sum  =0;
        for i = 1:len
            if belongTo(i) == 0
                dist_sum = dist_sum + dist.data(i);
            end
        end  
        % sort by ascent
        dist= qSort(dist);
        % percentage of all energy
        dist_sum = dist_sum*rate
        tmp_sum = 0;
        k = 1;
        % get a small part of whole energe (e.g rate = 5%)
        for k = 1:len
            if tmp_sum + dist.data(k) < dist_sum
                tmp_sum = tmp_sum + dist.data(k);
            else
                break;
            end
        end
        tmp_sum
        % break out at k, so the first k-1 belong to the same cluster
        for j = 1:k-1
            index = dist.index(j);
            if belongTo(index) == 0
                belongTo(index) = nClust;
            end
        end
    end
end

