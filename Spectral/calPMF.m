function [w1_P1, w1_M1] = calPMF(train_matrix)

  epsilon=5/1000*18; % Learning rate 
  lambda  = 10; % Regularization parameter 
  momentum=0.8; 

  epoch=1; 
  maxepoch=50; 

  numbatches= 2; % Number of batches  
  num_m = size(train_matrix,2);  % Number of features
  num_p = size(train_matrix,1);  % Number of clusters
  num_feat = num_m;           % Number of reward weights is the same as number of features 
  
  % convert matrix to vector
  train_vec = convertM2V(train_matrix);

  mean_rating = mean(train_vec(:,3)); 

  pairs_tr = length(train_vec); % training data 

  w1_M1     = 0.1*randn(num_m, num_feat); % Movie feature vectors
  w1_P1     = 0.1*randn(num_p, num_feat); % User feature vecators
  w1_M1_inc = zeros(num_m, num_feat);
  w1_P1_inc = zeros(num_p, num_feat);


    for epoch = epoch:maxepoch

      rr = randperm(pairs_tr);
      train_vec = train_vec(rr,:);
      clear rr 

      N=pairs_tr/numbatches; % number training triplets per batch (= totalNumber/batchNumber)
      for batch = 1:numbatches
        %fprintf(1,'epoch %d batch %d \r',epoch,batch);
       
        aa_p   = double(train_vec((batch-1)*N+1:batch*N,1));
        aa_m   = double(train_vec((batch-1)*N+1:batch*N,2));
        rating = double(train_vec((batch-1)*N+1:batch*N,3));

        rating = rating-mean_rating; % Default prediction is the mean rating. 

        %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
        pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
        f = sum( (pred_out - rating).^2 + ...
            0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));

        %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
        IO = repmat(2*(pred_out - rating),1,num_feat);
        Ix_m=IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
        Ix_p=IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);

        dw1_M1 = zeros(num_m,num_feat);
        dw1_P1 = zeros(num_p,num_feat);

        for ii=1:N
          dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
          dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
        end

        %%%% Update movie and user features %%%%%%%%%%%

        w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
        w1_M1 =  w1_M1 - w1_M1_inc;

        w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
        w1_P1 =  w1_P1 - w1_P1_inc;
      end 
    end 

end