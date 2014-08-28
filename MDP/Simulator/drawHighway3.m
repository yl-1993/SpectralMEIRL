function drawHighway3(nLanes, nGrids, gridw, gridh, y0)

minh   = 0;
maxh   = gridh*nGrids;
height = maxh - minh;
rectangle('Position', [0 minh gridw height], ...
    'FaceColor', [0.5 0.8 0.5], 'EdgeColor' , [0.5 0.8 0.5]);
rectangle('Position', [gridw minh gridw*nLanes height], ...
    'FaceColor', [0.5 0.5 0.5], 'EdgeColor' , [0.5 0.5 0.5]);
rectangle('Position', [gridw*(nLanes + 1) minh gridw height], ...
    'FaceColor', [0.5 0.8 0.5], 'EdgeColor' , [0.5 0.8 0.5]);

x = [gridw gridw];
y = [minh maxh];
line(x, y, 'Color', 'w', 'LineWidth', 5, 'LineStyle', '-');

x = [gridw gridw]*(nLanes + 1);
line(x, y, 'Color', 'w', 'LineWidth', 5, 'LineStyle', '-');

for i = 2:nLanes
    x = [gridw*i gridw*i];
    line(x, y, 'Color', 'w', 'LineWidth', 2, 'LineStyle', '-');
end

x1 = [gridw*0.5 gridw];
x2 = [gridw*(nLanes + 1) gridw*(nLanes + 1.5)];
for j = 1:(nGrids + 2)
    y = j - y0;
    if y < minh
        y = y + nGrids; 
    end
    y = gridh*y;
        
    line(x1, [y y], 'Color', 'w', 'LineWidth', 3, 'LineStyle', '-');
    line(x2, [y y], 'Color', 'w', 'LineWidth', 3, 'LineStyle', '-');
end

axis([0 gridw*(nLanes + 2) minh maxh]);
set(gca, 'xtick', []);
set(gca, 'ytick', []);

end