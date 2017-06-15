function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
for n = 1:N
    LogP = log(P.c);
    for k = 1:K        
        if(ndims(G)==2)% G could be 10x2
            GT = G;
        else% G could be 10x2x2
            GT = G(:,:,k);
        end
        for i = 1:10
            if(GT(i,1)==0)% the root body which only have the parents of C
                Py = lognormpdf(dataset(n,i,1),P.clg(i).mu_y(k),P.clg(i).sigma_y(k));
                Px = lognormpdf(dataset(n,i,2),P.clg(i).mu_x(k),P.clg(i).sigma_x(k));
                Pa = lognormpdf(dataset(n,i,3),P.clg(i).mu_angle(k),P.clg(i).sigma_angle(k));
                LogP(k) = LogP(k)+Py+Px+Pa;
            else
                Temp = [1 reshape(dataset(n,GT(i,2),:),1,3)]';% not only have the parents of C,but also another body which in G(i,2)
                Py = lognormpdf(dataset(n,i,1),P.clg(i).theta(k,1:4)*Temp,P.clg(i).sigma_y(k));
                Px = lognormpdf(dataset(n,i,2),P.clg(i).theta(k,5:8)*Temp,P.clg(i).sigma_x(k));
                Pa = lognormpdf(dataset(n,i,3),P.clg(i).theta(k,9:12)*Temp,P.clg(i).sigma_angle(k));
                LogP(k) = LogP(k)+Py+Px+Pa;
            end
        end        
    end
    loglikelihood = loglikelihood+log(sum(exp(LogP)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
