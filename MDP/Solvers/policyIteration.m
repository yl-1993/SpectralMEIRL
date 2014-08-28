function [piL, V, Q, H] = policyIteration(mdp, oldpi, opts)

MAX_ITERS = 10^4;
EPS = 1e-12;
SHOW_MSG  = 0;
if nargin == 3
    if ~isempty(opts.MAX_ITERS)
        MAX_ITERS = opts.MAX_ITERS;
    end
    if ~isempty(opts.SHOW_MSG)
        SHOW_MSG = opts.SHOW_MSG;
    end
end

if nargin < 2 || isempty(oldpi)
    oldpi = ones(mdp.nStates, 1);
end

if mdp.useSparse
    oldV = sparse(mdp.nStates, 1);
else
    oldV = zeros(mdp.nStates, 1);
end

for iter = 1:MAX_ITERS
    [V, H]   = evaluate(oldpi, mdp);    
    Q        = QfromV(V, mdp);    
    [V, piL] = max(Q, [], 2);
    done     = isequal(piL, oldpi) || approxeq(V, oldV, EPS);
    
    if SHOW_MSG == 1
        v0   = full(mdp.start'*V);
        diff = norm(V-oldV);
        fprintf('%d | %f %d %.20f\n', iter, v0, full(done), full(diff));
    end
    if done, break; end
    
    oldpi = piL;
    oldV  = V;
end

end