function nmi = calNMI(input, belongTo)

nExperts = size(input.weight, 2);
nrCl = max(belongTo);
M    = length(belongTo);

nc = zeros(nExperts, 1);
for j = 1:nExperts
    nc(j) = nnz(input.trajId == j);
end

nw = zeros(nrCl, 1);
for u = 1:nrCl
    nw(u) = nnz(belongTo == u);
end

nwc = zeros(nrCl, nExperts);
for u = 1:nrCl
    idx = belongTo == u;
    for w = 1:nExperts
        q = nnz(input.trajId(idx) == w);
        if q > 0
            nwc(u, w) = q/M*log(M*q/nw(u)/nc(w));
        end
    end
end

q = nw./M;
q = q(nw > 0);
entw = -sum(q.*log(q));

q = nc./M;
entc = -sum(q.*log(q));

nmi = 2*sum(nwc(:))/(entw + entc);

end