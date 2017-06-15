function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
accuracy = 0.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
tmp_labels = zeros(size(labels,1),1);
K = size(labels,2);
% compute difference labels' likelihood, predict which labels match the
% case
for n = 1:N
    Log_P = log(P.c);    
    for k = 1:K
        if(ndims(G)==2)
            GT = G;
        else
            GT = G(:,:,k);
        end
        for i = 1:10
            if(GT(i,1)==0)% only have one parent
                Py = lognormpdf(dataset(n,i,1),P.clg(i).mu_y(k),P.clg(i).sigma_y(k));
                Px = lognormpdf(dataset(n,i,2),P.clg(i).mu_x(k),P.clg(i).sigma_x(k));
                Pa = lognormpdf(dataset(n,i,3),P.clg(i).mu_angle(k),P.clg(i).sigma_angle(k));
                Log_P(k) = Log_P(k)+Py+Px+Pa;
            else
                tmp = [1 reshape(dataset(n,GT(i,2),:),1,3)]';
                Py = lognormpdf(dataset(n,i,1),P.clg(i).theta(k,1:4)*tmp,P.clg(i).sigma_y(k));
                Px = lognormpdf(dataset(n,i,2),P.clg(i).theta(k,5:8)*tmp,P.clg(i).sigma_x(k));
                Pa = lognormpdf(dataset(n,i,3),P.clg(i).theta(k,9:12)*tmp,P.clg(i).sigma_angle(k));
                Log_P(k) = Log_P(k)+Py+Px+Pa;
            end
        end
    end
    if(Log_P(1)>Log_P(2))% just compare the first column
        tmp_labels(n,1) = 1;    
    end
end
result = sum(tmp_labels(:,1)==labels(:,1));
accuracy = result/N;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Accuracy: %.2f\n', accuracy);