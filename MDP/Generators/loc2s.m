function s = loc2s(x, y, gridSize)

x = max(1, min(gridSize, x));
y = max(1, min(gridSize, y));
s = (y - 1)*gridSize + x;

end