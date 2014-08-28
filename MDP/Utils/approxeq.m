function p = approxeq(a, b, tol)
%
% Are a and b approximately equal (to within a specified tolerance)?
% 'tol' defaults to 1e-10.

if nargin < 3, tol = 1e-10; end

x = a(:);
y = b(:);

if length(x) ~= length(y)
    p = 0;
else
    p = norm(x - y, 'inf') < tol;
end

end
