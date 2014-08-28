function printResults(alg, data, problem, X, xstr, ...
    outpath, hist1, hist2)

if ~isempty(problem)
    fprintf('****************************************\n');
    fprintf('%s\n', getProblemName(problem));
    fprintf('****************************************\n');
end

if nargin > 3 && ~isempty(X)
    fprintf('-- %s\n', xstr);
    for i = 1:length(X)
        fprintf('%8.4f ', X(i));
    end
    fprintf('\n');
end

 fprintf('\n-- Reward difference\n');
 for k = 1:length(alg)
     fprintf('%12s ', getAlgName(alg{k}));
     for i = 1:size(data.MV, 2)
         fprintf('%8.4f (%6.4f) ', data.MR(k ,i), data.ER(k ,i));
     end
     fprintf('\n');
 end

 fprintf('\n-- Policy misprediction\n');
 for k = 1:length(alg)
     fprintf('%12s ', getAlgName(alg{k}));
     for i = 1:size(data.MP, 2)
         fprintf('%8.4f (%6.4f) ', data.MP(k ,i), data.EP(k ,i));
     end
     fprintf('\n');
 end

fprintf('\n-- Expected value difference\n');
for k = 1:length(alg)
    fprintf('%12s ', getAlgName(alg{k}));
    for i = 1:size(data.MV, 2)
        fprintf('%8.4f (%6.4f) ', data.MV(k ,i), data.EV(k ,i));
    end
    fprintf('\n');
end

if isfield(data, 'MF') && sum(isnan(data.MF(:))) == 0
    fprintf('\n-- F-score\n');
    for k = 1:length(alg)
        fprintf('%12s ', getAlgName(alg{k}));
        for i = 1:size(data.MV, 2)
            fprintf('%8.4f (%6.4f) ', data.MF(k ,i), data.EF(k ,i));
        end
        fprintf('\n');
    end
end

if isfield(data, 'MNMI') && sum(isnan(data.MNMI(:))) == 0
    fprintf('\n-- NMI\n');
    for k = 1:length(alg)
        fprintf('%12s ', getAlgName(alg{k}));
        for i = 1:size(data.MV, 2)
            fprintf('%8.4f (%6.4f) ', data.MNMI(k ,i), data.ENMI(k ,i));
        end
        fprintf('\n');
    end
end

if isfield(data, 'MNCL') && sum(isnan(data.MNCL(:))) == 0
    fprintf('\n-- # of clusters\n');
    for k = 1:length(alg)
        fprintf('%12s ', getAlgName(alg{k}));
        for i = 1:size(data.MV, 2)
            fprintf('%8.4f (%6.4f) ', data.MNCL(k ,i), data.ENCL(k ,i));
        end
        fprintf('\n');
    end
end
fprintf('\n');

if nargin > 5 && ~isempty(outpath)
    %problem.nTrajs = X;
    
    % set output file name
    probName = getProblemName(problem);
    outpath = sprintf('./%s/%s', outpath, probName);
    
    if ~isdir(outpath)
        fprintf('Mkdir %s !!!\n\n', outpath);
        mkdir(outpath);
    end
    
    if  problem.nExperts > 1 && problem.newExps > 0 ...
            && isfield(data, 'MF') && sum(isnan(data.MF(:))) == 0
        outfname = sprintf('%s/%s_new.mat', outpath, getAlgName(alg{k}));
    else
        outfname = sprintf('%s/%s.mat', outpath, getAlgName(alg{k}));
    end
    
    data.alg     = alg;
    data.X       = X;
    data.xstr    = xstr;
    data.problem = problem;
    save(outfname, 'data', '-v7.3');
    fprintf('save %s\n', outfname);
    
    if nargin > 7 && ~isempty(hist1)
        outfname = sprintf('%s/%s_hist.mat', outpath, getAlgName(alg{k}));
        save(outfname, 'hist1', '-v7.3');
    end
    if nargin > 8 && ~isempty(hist2)
        outfname = sprintf('%s/%s_newHist.mat', outpath, getAlgName(alg{k}));
        save(outfname, 'hist2', '-v7.3');
    end
end

end
