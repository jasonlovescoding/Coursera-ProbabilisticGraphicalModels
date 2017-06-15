function [P G loglikelihood] = LearnGraphAndCPDs(dataset, labels)

% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha) 
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

G = zeros(10,2,K); % graph structures to learn
% initialization
for k=1:K
    G(2:end,:,k) = ones(9,2);
end

% estimate graph structure for each class
for k=1:K
    % fill in G(:,:,k)
    % use ConvertAtoG to convert a maximum spanning tree to a graph G
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%
    [A ~] = LearnGraphStructure(dataset(labels(:,k)==1,:,:));
    G(:,:,k) = ConvertAtoG(A);
end

% estimate parameters

P.c = zeros(1,K);
% compute P.c
P.c = mean(labels,1);
% the following code can be copied from LearnCPDsGivenGraph.m
% with little or no modification
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
P.c = mean(labels,1);
for i = 1:size(G,1)
    if(G(i,1)==0)
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
        temp1 = squeeze(dataset(:,G(i,2,1),:));
        temp2 = squeeze(dataset(:,G(i,2,2),:));
        idx1 = find(labels(:,1)==1);
        idx2 = find(labels(:,2)==1);
        U_temp1 = temp1(idx1,:);
        U_temp2 = temp2(idx2,:);        
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('log likelihood: %f\n', loglikelihood);