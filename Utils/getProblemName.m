function probName = getProblemName(problem)

if problem.nExperts > 1
    probName = sprintf('%s(%d)_%dx%dx%d', problem.filename, ...
        problem.nExps, problem.nExperts, problem.nTrajs, problem.nSteps);
    if problem.newExps > 0
        probName = sprintf('%s_%dx%d', probName, problem.newExps, ...
            problem.newTrajSteps);
        if problem.newExpertProb > 0
            probName = sprintf('%sx%d', probName, ...
                floor(problem.newExpertProb*10));
        end
    end
else
    probName = sprintf('%s(%d)_%dx%d', problem.filename, ...
        problem.nExps, problem.nTrajs, problem.nSteps);
end

