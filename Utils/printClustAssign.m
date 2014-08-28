% Print cluster assignment
function printClustAssign(belongTo)

fprintf('[');
for i = 1:length(belongTo)
    fprintf('%2d ', belongTo(i));
end
fprintf(']\n');

end