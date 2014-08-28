function plotResults(X, mu, se, alg, xstr, ystr, tstr, bErr, bLegend)

nAlgs = length(alg);
cc = hsv(nAlgs);
mm = '+o*xsd^v><ph.';
hold on;

for k = 1:nAlgs    
    if length(X) < 10
        marker = mm(k);
    else
        marker = 'none';
    end
    plot(X, mu(k, :), 'Color', cc(k, :), 'LineWidth', alg{k}.lw, ...
        'LineStyle', alg{k}.lt, 'Marker', marker);
end

if nargin > 8 && bLegend
    maxStrLen = 0;
    buf = cell(nAlgs, 1);
    for k = 1:nAlgs
        tmp = getAlgName(alg{k});
        tmp(tmp == '_') = '-';
        buf{k} = tmp;
        maxStrLen = max(maxStrLen, length(buf{k}));
    end
    legendStr = [];
    for k = 1:nAlgs
        str = buf{k};
        n   = length(buf{k});
        str(n + 1:maxStrLen) = ' ';
        legendStr = cat(1, legendStr, str);
    end
    legend(legendStr, 'Location', 'NorthEast'); %'SouthEast');
end

if nargin > 7 && bErr && length(X) < 10
    for k = 1:nAlgs
        errorbar(X, mu(k, :), se(k, :), ...
            'LineStyle', 'none', 'Color', cc(k, :));
    end
    set(gca, 'xtick', X);
end

xlabel(xstr);
ylabel(ystr);
axis('tight');

if nargin > 5 && ~isempty(tstr)
    tstr(tstr == '_') = ' ';
    title(tstr);
end

end
