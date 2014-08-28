function occ = computeOcc(piL, mdp)

nS = mdp.nStates;
nA = mdp.nActions;

if mdp.useSparse
    I = speye(nS);
    Tpi = sparse(nS, nS, nS);
    for a = 1:nA
        idx = find(piL == a);
        if ~isempty(idx)
            Tpi(idx, :) = mdp.transitionS{a}(:, idx)';
        end
    end
    idx = (piL-1)*nS+(1:nS)';
    idx = (idx-1)*nS+(1:nS)';
    Epi = sparse(nS, nS*nA);
    Epi(idx) = 1;
else  
    I = eye(nS);
    Tpi = zeros(nS, nS);
    for a = 1:nA
        idx = find(piL == a);
        if ~isempty(idx)
            Tpi(idx, :) = squeeze(mdp.transition(:, idx, a)');
        end
    end
    idx = (piL-1)*nS+(1:nS)';
    idx = (idx-1)*nS+(1:nS)';
    Epi = zeros(nS, nS*nA);
    Epi(idx) = 1;
end

occ = ((I-mdp.discount.*Tpi)\Epi)'*mdp.start;

end