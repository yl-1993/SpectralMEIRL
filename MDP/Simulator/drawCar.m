function drawCar(x0, y0, dir, cl, gridw, gridh)

a1 = gridw*0.3;
a2 = gridw*0.35;
a3 = gridw*0.65;
a4 = gridw*0.7;
b1 = gridh*0.2;
b2 = gridh*0.75;
b3 = gridh*0.8;
x = x0 + [a1 a4 a4 a3 a2 a1 a1];
y = y0 + [b1 b1 b2 b3 b3 b2 b1];
if dir == 1
    t = (y0 + gridh/2) - y;
    y = y + 2*t;
end
fill(x, y, cl, 'EdgeColor', 'none');

a1 = gridw*0.35;
a2 = gridw*0.65;
b1 = gridh*0.25;
b2 = gridh*0.6;
x = x0 + [a1 a2 a2 a1 a1];
y = y0 + [b1 b1 b2 b2 b1];
if dir == 1
    t = (y0 + gridh/2) - y;
    y = y + 2*t;
end
fill(x, y, [0 0 1], 'EdgeColor', 'none');

a1 = gridw*0.38;
a2 = gridw*0.62;
b1 = gridh*0.28;
b2 = gridh*0.55;
x = x0 + [a1 a2 a2 a1 a1];
y = y0 + [b1 b1 b2 b2 b1];
if dir == 1
    t = (y0 + gridh/2) - y;
    y = y + 2*t;
end
fill(x, y, cl, 'EdgeColor', 'none');

end