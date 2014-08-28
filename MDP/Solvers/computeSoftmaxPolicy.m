function piL = computeSoftmaxPolicy(mdp, V, ETA)

if nargin < 3
    ETA = 2;
end

Q   = ETA.*QfromV(V, mdp);
piL = exp(Q);
piL = bsxfun(@rdivide, piL, sum(piL, 2));
if nnz(isnan(piL) | isinf(piL)) > 0
    fprintf('ERROR: compute softmax policy\n');
end
    
end
