function params = paramsMEIRL(alg, llhType, priorType)

params.alg       = alg;
params.llhType   = llhType;
params.priorType = priorType;
params.restart   = 1;       % # of random restart
params.showMsg   = false; % true; %
params.maxTime   = 3600;


params.lb        = -1;      % lower bounds of reward
params.ub        = 1;       % upper bounds of reward

if strcmp(priorType, 'NG')  % normal-gamma dist for hyper-prior
    params.beta  = 1;       % hyper-parameter for mean of reward
    params.gamma = [1, 1];  % hyper-parameter for variance of reward
    params.delta = 1e-4;    % discretization level for reward value
    
elseif strcmp(priorType, 'BG')  % beta-gamma dist for hyper-prior    
    params.lb    = 0 + 1e-2;    % r \in [0,1]
    params.ub    = 1 - 1e-2;
    params.beta  = 0.5;     % hyper-parameter for mean of reward prior
    params.gamma = [1 1];   % hyper-parameter for precision of reward prior
    params.nmu   = 10000;   % number of samples for Monte-Carlo integration
    
elseif strcmp(priorType, 'Gaussian')
    params.mu    = 0.0;     % mean of reward function
    params.sigma = 0.1;     % std. dev. of reward function
    
elseif strcmp(priorType, 'Uniform')
    
end


if strcmp(llhType, 'BIRL')
    params.eta          = 10.0;     % inverse temperature
    
elseif strcmp(llhType, 'MMP')
    params.slackPenalty = 1;        % L1 or L2-norm slack penalty
    
elseif strcmp(llhType, 'PM')
    params.natural      = false;    % using natural gradient
    params.eta          = 1.0;      % inverse temperature
    
elseif strcmp(llhType, 'MaxEnt')
    
elseif strcmp(llhType, 'MLIRL')
    params.eta          = 10.0;      % inverse temperature
end


if strcmp(alg, 'EM_IRL')
    params.nClust     = 3;          % # of clusters
    params.maxIters   = 50;         % # of iterations for EM
    params.EMrestart  = 1;          % # of random restart for EM
  
elseif strcmp(alg, 'DPM_BIRL_MH')
    params.alpha      = 1;          % concentration parameter for DP
    params.maxIters   = 400;      % # of iterations for MH
    params.clIters    = 2;          % # of iterations for cluster assignment update
    params.wIters     = 10;         % # of iterations for reward update
    params.newIters   = 100;       % # of iterations for transfer
  
elseif strcmp(alg, 'DPM_BIRL_MHL')
    params.alpha      = 1;          % concentration parameter for DP
    params.maxIters   = 100;      % # of iterations for MH
    params.clIters    = 2;         % # of iterations for cluster assignment update
    params.wIters     = 10;         % # of iterations for reward update
    params.newIters   = 100;       % # of iterations for transfer

elseif strcmp(alg, 'SPECTRAL_IRL')
    params.nClusters     = 3;          % # of clusters
        
end
