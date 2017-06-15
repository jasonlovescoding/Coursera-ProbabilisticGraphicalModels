function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
P.c = mean(labels,1);
for i = 1:size(G,1)
    if(G(i,1)==0)% only have one parent
        temp = squeeze(dataset(:,i,:));
        idx1 = find(labels(:,1)==1);
        idx2 = find(labels(:,2)==1);
        [P.clg(i).mu_y(1) P.clg(i).sigma_y(1)] = FitGaussianParameters(temp(idx1,1));
        [P.clg(i).mu_y(2) P.clg(i).sigma_y(2)] = FitGaussianParameters(temp(idx2,1));
        [P.clg(i).mu_x(1) P.clg(i).sigma_x(1)] = FitGaussianParameters(temp(idx1,2));
        [P.clg(i).mu_x(2) P.clg(i).sigma_x(2)] = FitGaussianParameters(temp(idx2,2));
        [P.clg(i).mu_angle(1) P.clg(i).sigma_angle(1)] = FitGaussianParameters(temp(idx1,3));
        [P.clg(i).mu_angle(2) P.clg(i).sigma_angle(2)] = FitGaussianParameters(temp(idx2,3));
        P.clg(i).theta = [];
    else
        temp = squeeze(dataset(:,G(i,2),:));
        idx1 = find(labels(:,1)==1);
        idx2 = find(labels(:,2)==1);
        U_temp1 = temp(idx1,:);
        U_temp2 = temp(idx2,:);        
        [Beta P.clg(i).sigma_y(1)] = FitLinearGaussianParameters(dataset(idx1,i,1), U_temp1);
        P.clg(i).theta(1,1) = Beta(4);
        P.clg(i).theta(1,2:4) = Beta(1:3);
        [Beta P.clg(i).sigma_y(2)] = FitLinearGaussianParameters(dataset(idx2,i,1), U_temp2);
        P.clg(i).theta(2,1) = Beta(4);
        P.clg(i).theta(2,2:4) = Beta(1:3);        
        [Beta P.clg(i).sigma_x(1)] = FitLinearGaussianParameters(dataset(idx1,i,2), U_temp1);
        P.clg(i).theta(1,5) = Beta(4);
        P.clg(i).theta(1,6:8) = Beta(1:3);        
        [Beta P.clg(i).sigma_x(2)] = FitLinearGaussianParameters(dataset(idx2,i,2), U_temp2);
        P.clg(i).theta(2,5) = Beta(4);
        P.clg(i).theta(2,6:8) = Beta(1:3);        
        [Beta P.clg(i).sigma_angle(1)] = FitLinearGaussianParameters(dataset(idx1,i,3), U_temp1);
        P.clg(i).theta(1,9) = Beta(4);
        P.clg(i).theta(1,10:12) = Beta(1:3);
        [Beta P.clg(i).sigma_angle(2)] = FitLinearGaussianParameters(dataset(idx2,i,3), U_temp2);
        P.clg(i).theta(2,9) = Beta(4);
        P.clg(i).theta(2,10:12) = Beta(1:3);        
    end
end
loglikelihood = ComputeLogLikelihood(P, G, dataset);
% These are dummy lines added so that submit.m will run even if you 
% have not started coding. Please delete them.
%P.clg.sigma_x = 0;
%P.clg.sigma_y = 0;
%P.clg.sigma_angle = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('log likelihood: %f\n', loglikelihood);

