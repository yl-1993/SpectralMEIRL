function [dist] = getVecDistance(uArr,vArr)
    len = min(size(uArr,2),size(vArr,2));
    dist = 0;
    % Euclidean distance
%     for i = 1:len
%         dist = dist + getDistance(uArr(i), vArr(i));
%     end
    % cosin distance
    uNorm = 0;
    vNorm = 0;
    for i = 1:len
        dist = dist + uArr(i)*vArr(i);
        uNorm = uNorm + uArr(i)*uArr(i);
        vNorm = vNorm + vArr(i)*vArr(i);
    end
    if uNorm ==0 || vNorm == 0 
        dist = 0;
    else
        dist = dist/(sqrt(uNorm)*sqrt(vNorm));
end