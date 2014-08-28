% Sample according to the multinomial distribution
%
function id = sampleMultinomial(dist)

x = dist;
s = sum(x);
if s ~= 1, x = x./sum(dist); end
id = find(cumsum(x) > rand, 1);

end