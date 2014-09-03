% Generate MDP problems
function mdp = generateProblem(problem, seed, discount)

if nargin < 3 && isempty(discount)
    discount = 0.99;
end

RandStream.setGlobalStream(RandStream.create('mrg32k3a', ...
    'NumStreams', 1, 'Seed', seed));
if strcmp(problem.name, 'highway3')
    mdp = highway3(discount,1);
    %load('./MDP/highway3');
    mdp.discount = discount;
    
elseif strcmp(problem.name, 'gridworld')
    mdp = gridworld(problem.gridSize, problem.blockSize, ...
        problem.noise, discount);
end

% auxiliary matrices for computing gradient and so on.
nS = mdp.nStates;
nA = mdp.nActions;
if mdp.useSparse
    I = repmat(speye(nS), nA, 1);
    mdp.T = sparse(nS, nS*nA);
    for a = 1:nA
        i = (a-1)*nS+1;
        j = a*nS;
        mdp.T(:, i:j) = mdp.transitionS{a};
    end
else
    I = repmat(eye(nS), nA, 1);
    mdp.T = reshape(mdp.transition, nS, nS*nA);
end
mdp.T = mdp.discount.*mdp.T';
mdp.E = I-mdp.T;

end
