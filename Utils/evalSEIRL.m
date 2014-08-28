function [results, piL] = evalSEIRL(wL, wE, mdp)

mdp = convertW2R(wE, mdp);
[piE, VE, QE, HE] = policyIteration(mdp);
vE = full(wE'*HE'*mdp.start);

mdp = convertW2R(wL, mdp);
[piL, VL, QL, HL] = policyIteration(mdp);
vL = full(wE'*HL'*mdp.start);

d  = zeros(mdp.nStates, 1);
for s = 1:mdp.nStates
    ixE = QE(s, :) == max(QE(s, :));
    ixL = QL(s, :) == max(QL(s, :));
    if isequal(ixE, ixL)
        d(s) = 0;
    else
        d(s) = 1;
    end
end

results.rewardDiff = norm(wE - wL);
results.policyDiff = sum(d)/mdp.nStates;
results.valueDiff  = vE - vL;
vE
vL
