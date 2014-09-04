function train_vec = convertM2V(train_matrix)
    nRow = size(train_matrix,1);
    nCol = size(train_matrix,2);
    train_vec = zeros(nRow*nCol,3);
    for i = 1:nRow
        for j = 1:nCol
            k = (i-1)*nCol+j;
            train_vec(k,1) = i;
            train_vec(k,2) = j;
            train_vec(k,3) = train_matrix(i,j);
        end
    end
end