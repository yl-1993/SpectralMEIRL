% belongTo : cluster assignment
% z(i)     : # of cluster whose size is i
% szCl(i)  : size of i-th cluster
% z2c 

function [z, szCl, z2c] = count(belongTo)

N    = length(belongTo);    % # of data
nrCl = max(belongTo);       % # of total cluster
szCl = zeros(1, nrCl);
z    = zeros(1, N);
z2c  = cell(1, N);
for k = 1:nrCl
    szCl(k) = sum(belongTo == k);
end
for i = 1:N
    z2c{i} = [];
end
for i = 1:nrCl
    if szCl(i) > 0
        z(szCl(i)) = z(szCl(i)) + 1; 
        z2c{szCl(i)} = cat(2, z2c{szCl(i)}, i);
    end
end