function [V, Q, iter] = valueIteration(mdp, opts)

EPS       = 1e-12;
MAX_ITERS = 10^4;
SHOW_MSG  = 0;
if nargin > 1
    if ~isempty(opts.EPS)
        EPS = opts.EPS;
    end
    if ~isempty(opts.MAX_ITERS)
        MAX_ITERS = opts.MAX_ITERS;
    end
    if ~isempty(opts.SHOW_MSG)
        SHOW_MSG = opts.SHOW_MSG;
    end
end

nS = mdp.nStates;
nA = mdp.nActions;

if mdp.useSparse
    oldV = sparse(max(mdp.reward, [], 2));
    for iter = 1:MAX_ITERS
        Q = sparse(nS, nA);
        for a = 1:nA
            Q(:, a) = mdp.rewardS{a} ...
                + mdp.discount*(mdp.transitionS{a}(:, :)'*oldV);
        end
        V = max(Q, [], 2);
        done = approxeq(V, oldV, EPS);
        if SHOW_MSG == 1
            v0 = mdp.start'*V;
            fprintf('%d | %f %d\n', iter, full(v0), full(done));
        end
        if done, break; end
        oldV = V;
    end
else
    oldV = max(mdp.reward, [], 2);
    for iter = 1:MAX_ITERS
        Q = zeros(nS, nA);
        for a = 1:nA
            Q(:, a) = mdp.reward(:, a) ...
                + mdp.discount*(squeeze(mdp.transition(:, :, a)')*oldV);
        end
        V = max(Q, [], 2);
        done = approxeq(V, oldV, EPS);
        if SHOW_MSG == 1
            v0 = mdp.start'*V;
            fprintf('%d | %f %d\n', iter, v0, done);
        end
        if done, break; end
        oldV = V;
    end
end

end