function [V, H] = evaluateStoch(piL, mdp, w)
%
% Evaluate policy measured on the given reward weight w
% V = (I-gamma*T^piL)^-1 R^piL
%
if nargin == 3 && ~isempty(w)
    mdp = generateReward(w, mdp);
end

nS = mdp.nStates;
nA = mdp.nActions;

if mdp.useSparse    
    I = speye(nS);
    Rpi = sparse(nS, 1);
    for a = 1:nA
        Rpi = Rpi+piL(:, a).*mdp.rewardS{a};
    end
    Tpi = sparse(nS, nS);
    for s = 1:nS
        for a = 1:nA
            Tpi(:, s) = Tpi(:, s)+piL(:, a).*mdp.transitionS{a}(s, :)';
        end
    end
else
    I = eye(nS);
    Rpi = sum(piL.*mdp.reward, 2);
    Tpi2 = zeros(nS, nS);
    for s = 1:nS
        Tpi2(:, s) = sum(piL.*squeeze(mdp.transition(s, :, :)), 2);
    end
end

H = [];
V = (I-mdp.discount.*Tpi)\Rpi;

% EPS       = 1e-14;
% MAX_ITERS = 10^5;
% oldV = Rpi;
% for iter = 1:MAX_ITERS
%     V = Rpi+mdp.discount*(Tpi*oldV);
%     done = approxeq(V, oldV, EPS);
%     if done, break; end
%     oldV = V;
% end

end