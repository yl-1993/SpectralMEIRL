function simulateHighway3(mdp, w, nSteps)

if nargin > 0 && ~isempty(mdp)
    fprintf('load highway problem\n');
else
    fprintf('generate highway problem\n');
    mdp = highway3(0.9);
end

if nargin == 2 && ~isempty(w)
    mdp = convertW2R(w, mdp);
end

fprintf('solve highway problem\n');
policy = policyIteration(mdp);

fprintf('sample trajectory\n');
nTrajs = 1;
if nargin < 3 || isempty(nSteps)
    nSteps = 1000;
end
trajs = sampleTrajectories(nTrajs, nSteps, policy, mdp);

nSpeeds = mdp.nSpeeds;
nLanes  = mdp.nLanes;
nGrids  = mdp.nGrids;
carSize = mdp.carSize;

gridw   = 10;
gridh   = 20;
nFrames = 12;
unitTime = 0.1;
act{1} = '';
act{2} = 'Move left';
act{3} = 'Move right';
act{4} = 'Accelerate';
act{5} = 'Deaccelerate';

fig = figure('Position', [1200, 600, gridw*nLanes*10, gridh*nGrids*2], ...
    'MenuBar', 'none');

nCollisions = 0;
for t = 1:(nSteps - 1)
    s  = trajs(1, t, 1);
    a  = trajs(1, t, 2);
    f  = full(mdp.F((a - 1)*mdp.nStates + s, :));
    ns = trajs(1, t + 1, 1);
    
    [spd, x, y1, y2, y3]      = sid2info(s, nSpeeds, nLanes, nGrids);
    [nspd, nx, ny1, ny2, ny3] = sid2info(ns, nSpeeds, nLanes, nGrids);
    nCollisions = nCollisions + f(1);
    Y  = [y1, y2, y3];
    nY = [ny1, ny2, ny3];
    
    z  = zeros(3, 1);
    nz = zeros(3, 1);
    for i = 1:3
        if Y(i) == 1 && nY(i) == 1
            z(i)  = -1;
            nz(i) = -1;
        elseif Y(i) == 1
            z(i)  = nY(i) - spd;
            nz(i) = nY(i);
        elseif nY(i) == 1
            z(i)  = Y(i);
            nz(i) = Y(i) + spd;
        else
            z(i)  = Y(i);
            nz(i) = nY(i);
        end
    end
    z  = nGrids - z;
    nz = nGrids - nz;
    
    clf(fig);
    hold on;
    
    for i = 0:nFrames - 1
        drawHighway3(nLanes, nGrids, gridw, gridh, spd*i/nFrames);
        for j = 1:3
            cy = z(j) + (nz(j) - z(j))*i/nFrames;
            drawCar(gridw*j, gridh*cy, 0, 'r', gridw, gridh*carSize);
        end
        cx = x + (nx - x)*i/nFrames;
        drawCar(gridw*cx, gridh*carSize, 0, 'g', gridw, gridh*carSize);
        
        text(gridw*0.1, -0.5*gridh, ...
            sprintf('# of collisions: %d', nCollisions), ...
            'FontWeight', 'bold');
        if spd == 1
            spdstr = 'Speed: low';
        else
            spdstr = 'Speed: high';
        end
        text(gridw*0.1, -1*gridh, spdstr, 'FontWeight', 'bold');

        text(gridw*3, -0.5*gridh, act{a}, 'FontWeight', 'bold');
        movFail = (a == 2 || a == 3) && (x == nx);
        accFail = (a == 4 || a == 5) && (spd == nspd);
        if a > 1 && (movFail || accFail)
            text(gridw*3, -1*gridh, 'Fail', 'FontWeight', 'bold');
        elseif a > 1
            text(gridw*3, -1*gridh, 'Success', 'FontWeight', 'bold');
        end
        text(gridw*0.1, (nGrids + 0.1)*gridh, sprintf('%d', t), 'FontWeight', 'bold');

        pause(unitTime/nFrames);
    end
end

end