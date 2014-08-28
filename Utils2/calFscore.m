function fsc = calFscore(input, belongTo)

nExperts = size(input.weight, 2);
nrCl = max(belongTo);
M  = length(belongTo);
tp = zeros(nExperts, 1);
fp = zeros(nExperts, 1);
fn = zeros(nExperts, 1);
tn = zeros(nExperts, 1);

for u = 1:nExperts
    idx = input.trajId == u;
    tmp = belongTo(idx);
    cnt = zeros(nrCl, 1);
    for m = 1:nrCl
        cnt(m) = nnz(tmp == m);
    end
    [v, w] = max(cnt);
    tp(u) = v;
    fp(u) = nnz(belongTo == w) - v;
    fn(u) = nnz(idx) - v;
    tn(u) = M - (tp(u) + fp(u) + fn(u));
end

precision = sum(tp) / (sum(tp) + sum(fp));
recall    = sum(tp) / (sum(tp) + sum(fn));
fsc       = 2*precision*recall/(precision + recall);

end