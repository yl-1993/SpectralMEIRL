function nrCl = countClust(belongTo)

N = max(belongTo);
L = zeros(N, 1);
for k = 1:N
    L(k) = nnz(belongTo == k);
end
nrCl = nnz(L);

end