% Sample trajectories by executing policy piL
function [trajs, Vmean, Vvar] = sampleTrajectories(nTrajs, nSteps, piL, mdp)

trajs  = zeros(nTrajs, nSteps, 2);
vList  = zeros(nTrajs, 1);
for m = 1:nTrajs
    s = sampleMultinomial(mdp.start); % sample start state
    v = 0;
    for h = 1:nSteps
        a = piL(s);
        r = mdp.reward(s, a);
        v = v + r*mdp.discount^(h-1);
        trajs(m, h, :) = [s, a];
        
        if mdp.useSparse
            s = sampleMultinomial(mdp.transitionS{a}(:, s));
        else
            s = sampleMultinomial(mdp.transition(:, s, a));
        end
    end
    vList(m) = v;
end
Vmean = mean(vList);
Vvar  = var(vList);

end