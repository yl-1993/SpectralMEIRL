% Information transfer to new trajectory using the results from EM
%
function wL = EM_IRL_transfer(sol, traj, mdp, opts)

trajInfo = getTrajInfo(traj, mdp);

nrCl = length(sol.rho);
dist = zeros(nrCl, 1);
for k = 1:nrCl
    if sol.rho(k) > 0
        w = sol.weight(:, k);
        llh = calLogLLH(w, trajInfo, [], [], [], mdp, opts);        
        dist(k) = sol.rho(k)*exp(llh);
    end
end
wL = sol.weight*(dist./sum(dist));
