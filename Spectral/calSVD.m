function trajInfo = calSVD(featExp)
%
% Compute Singular Value Decomposition for trajectory*feature matrix.
%
[u,singular,v] = svd(featExp);
trajInfo.u = u;
trajInfo.singular = singular;
trajInfo.v = v;

end %end_of_calSVD