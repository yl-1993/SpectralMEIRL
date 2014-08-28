function evd = calEVD(input, vE, cl, mdp)

M = length(cl.b);
vdiff = nan(M, 1);
for m = 1:M
    kE = input.trajId(m);
    wE = input.weight(:, kE);
    
    kL = cl.b(m);
    wL = cl.w(:, kL);
    if isfield(cl, 'p') && ~isempty(cl.p)
        pL = cl.p(:, kL);
    else
        mdp = convertW2R(wL, mdp);
        pL  = policyIteration(mdp);
    end
    [VL, HL] = evaluate(pL, mdp, wL);
    vL = full(wE'*HL'*mdp.start);
    
    vdiff(m) = vE(m) - vL;
end
evd = mean(vdiff);

end