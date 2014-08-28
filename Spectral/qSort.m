function  dist = qSort(dist, low, high)
if nargin == 1
    low = 1;
    high = length(dist.data);
end
if(low<high)
    [dist pivokey] = qPartition(dist, low, high);
    dist = qSort( dist, low, pivokey - 1 );
    dist = qSort( dist, pivokey + 1, high );
end
% Partion
function [dist low] = qPartition(dist, low, high)
if nargin == 1
    low = 1;
    hight = length(dist.data);
end
pivokey=dist.data(low);
while low < high
    while low < high & dist.data(high) >= pivokey
        high = high - 1;
    end
    cd = dist.data(low); dd = dist.data(high);
    ci = dist.index(low); di = dist.index(high);
    dist.data(low) = dd; dist.data(high) = cd;
    dist.index(low) = di; dist.index(high) = ci;
    while low < high & dist.data(low) <= pivokey
        low = low + 1;
    end
    cd = dist.data(low); dd = dist.data(high);
    ci = dist.index(low); di = dist.index(high);
    dist.data(low) = dd; dist.data(high) = cd;
    dist.index(low) = di; dist.index(high) = ci;
end