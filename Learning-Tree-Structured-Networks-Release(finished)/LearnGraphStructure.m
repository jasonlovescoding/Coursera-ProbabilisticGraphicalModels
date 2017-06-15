function [A W] = LearnGraphStructure(dataset)

% Input:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% A: maximum spanning tree computed from the weight matrix W
% W: 10 x 10 weight matrix, where W(i,j) is the mutual information between
%    node i and j. 
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1);
K = size(dataset,3);

W = zeros(10,10);
% Compute weight matrix W
% set the weights following Eq. (14) in PA description
% you don't have to include M since all entries are scaled by the same M
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE   
for i = 1:size(W,1)
    for j = i:size(W,2)        
        temp1 = squeeze(dataset(:,i,:));
        temp2 = squeeze(dataset(:,j,:));
        W(i,j) = GaussianMutualInformation(temp1,temp2);
    end
end
for i = 2:size(W,1)
    for j = 1:i-1
        W(i,j) = W(j,i);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute maximum spanning tree
A = MaxSpanningTree(W);