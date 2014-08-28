% Calculate prior of cluster assignment c, pr(c|alpha)
%
function pr = calClustAssignProb(belongTo, alpha)

z  = count(belongTo);
ix = 1:length(z);
Z  = sum(z.*ix);

k  = ix.^z.*factorial(z);
pr = factorial(Z)/prod(alpha:alpha+Z-1)*(alpha^sum(z))/prod(k);

end
