% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

load PA9SampleCases
% EM algorithm
for iter=1:maxIter
    
    % M-STEP to estimate parameters for Gaussians
    % Fill in P.c, the initial state prior probability (NOT the class
    % probability as in PA8 and EM_cluster.m)
    % Fill in P.clg for each body part and each class
    % Make sure to choose the right parameterization based on G(i,1)
    % Hint: This part should be similar to your work from PA8 and EM_cluster.m
    
    P.c = zeros(1,K);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    for ll = 1:L
        P.c = P.c+ClassProb(actionData(ll).marg_ind(1),:);
%         tx = actionData(ll).marg_ind(1);
    end
    P.c = P.c/L;
    M = size(G,1);
    for k = 1:K
        for i = 1:M
            if(G(i,1)==0)% only have one parent
                [P.clg(i).mu_y(k) P.clg(i).sigma_y(k)] = FitG(poseData(:,i,1), ClassProb(:,k));
                [P.clg(i).mu_x(k) P.clg(i).sigma_x(k)] = FitG(poseData(:,i,2), ClassProb(:,k));
                [P.clg(i).mu_angle(k) P.clg(i).sigma_angle(k)] = FitG(poseData(:,i,3), ClassProb(:,k));
                P.clg(i).theta = [];
            else% similar PA8
                
                [temp P.clg(i).sigma_y(k)] = FitLG(poseData(:,i,1), ...
                    squeeze(poseData(:,G(i,2),:)), ClassProb(:,k));
                P.clg(i).theta(k,2:4) = temp(1:3);
                P.clg(i).theta(k,1) = temp(4);
                [temp P.clg(i).sigma_x(k)] = FitLG(poseData(:,i,2), ...
                    squeeze(poseData(:,G(i,2),:)), ClassProb(:,k));
                P.clg(i).theta(k,6:8) = temp(1:3);
                P.clg(i).theta(k,5) = temp(4);
                [temp P.clg(i).sigma_angle(k)] = FitLG(poseData(:,i,3), ...
                    squeeze(poseData(:,G(i,2),:)), ClassProb(:,k));
                P.clg(i).theta(k,10:12) = temp(1:3);
                P.clg(i).theta(k,9) = temp(4);
            end
        end
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % M-STEP to estimate parameters for transition matrix
    % Fill in P.transMatrix, the transition matrix for states
    % P.transMatrix(i,j) is the probability of transitioning from state i to state j
    P.transMatrix = zeros(K,K);
    
    % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
    P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE  
    % i get stuck in this , thanks to Vesselin Vassilev
    % https://class.coursera.org/pgm-2012-002/forum/thread?thread_id=1297
    LogPairP = PairProb;
    P.transMatrix = reshape(mean(LogPairP,1),3,3);
    % why not use 'sum' as below
    % P.transMatrix = reshape(sum(LogPairP,1),3,3);
    for i = 1:K
        P.transMatrix(i,:) = (P.transMatrix(i,:)+0.05)/sum((P.transMatrix(i,:)+0.05));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % E-STEP preparation: compute the emission model factors (emission probabilities)
    % in log space for each
    % of the poses in all actions = log( P(Pose | State) )
    % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
    
    logEmissionProb = zeros(N,K);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %    almost the same as ClassifyDataset of PA8, the only different of it is
    %    avoid multiplying probability of pose given state by probability of
    %    state
    %    follow by Taras Kobernik
    for n = 1:N        
        for k = 1:K
            if(ndims(G)==2)
                GT = G;
            else
                GT = G(:,:,k);
            end
            for i = 1:10
                if(GT(i,1)==0)% only have one parent
                    Py = lognormpdf(poseData(n,i,1),P.clg(i).mu_y(k),P.clg(i).sigma_y(k));
                    Px = lognormpdf(poseData(n,i,2),P.clg(i).mu_x(k),P.clg(i).sigma_x(k));
                    Pa = lognormpdf(poseData(n,i,3),P.clg(i).mu_angle(k),P.clg(i).sigma_angle(k));
                    logEmissionProb(n,k) = logEmissionProb(n,k)+Py+Px+Pa;
                else
                    Temp = [1 reshape(poseData(n,GT(i,2),:),1,3)]';
                    Py = lognormpdf(poseData(n,i,1),P.clg(i).theta(k,1:4)*Temp,P.clg(i).sigma_y(k));
                    Px = lognormpdf(poseData(n,i,2),P.clg(i).theta(k,5:8)*Temp,P.clg(i).sigma_x(k));
                    Pa = lognormpdf(poseData(n,i,3),P.clg(i).theta(k,9:12)*Temp,P.clg(i).sigma_angle(k));
                    logEmissionProb(n,k) = logEmissionProb(n,k)+Py+Px+Pa;
                end
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % E-STEP to compute expected sufficient statistics
    % ClassProb contains the conditional class probabilities for each pose
    % in all actions
    % PairProb contains the expected sufficient statistics for the
    % transition CPDs (pairwise transition probabilities)
    % Also compute log likelihood of dataset for this iteration
    % You should do inference and compute everything in log space, only
    % converting to probability space at the end
    % Hint: You should use the logsumexp() function here to do probability
    % normalization in log space to avoid numerical issues
    
    ClassProb = zeros(N,K);
    PairProb = zeros(V,K^2);
    loglikelihood(iter) = 0;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    temp_loglikelihood = 0;
    for i = 1:length(actionData)
        F = repmat(struct('var',[],'card',[],'val',[]),...
            1+size(actionData(i).marg_ind,2)+...
            size(actionData(i).pair_ind,2),1);        
        F(1).var = 1;
        F(1).card = K;
        F(1).val = log(P.c);
        
        for j = 1:size(actionData(i).marg_ind,2)% include marg_ind            
            F(j+1).var = j;
            F(j+1).card = K;
            F(j+1).val = [logEmissionProb(actionData(i).marg_ind(j),:)];
        end
        j = j+1;
        temp = log(P.transMatrix(:));
        for jj = 1:size(actionData(i).pair_ind,2)% include pair_ind
            %             F(j+jj).var = [actionData(i).marg_ind(jj) actionData(i).marg_ind(jj+1)];
            F(j+jj).var = [jj jj+1];
            F(j+jj).card = [K K];
            F(j+jj).val = [temp'];
        end
        [M, PCalibrated] = ComputeExactMarginalsHMM(F);% get the infer result
        % compute the loglikelihood
        temp_loglikelihood = temp_loglikelihood+sum(logsumexp(PCalibrated.cliqueList(1).val));
        
        for j = 1:length(M)% compute ClassProb
            temp = logsumexp(M(j).val);
            ClassProb(M(j).var+actionData(i).marg_ind(1)-1,:) = exp(M(j).val-temp);
        end
        for j = 1:length(PCalibrated.cliqueList)
            temp = logsumexp(PCalibrated.cliqueList(j).val);
            PairProb(PCalibrated.cliqueList(j).var(1)+actionData(i).pair_ind(1)-1,:) = exp(PCalibrated.cliqueList(j).val-temp);
        end
        clear F;
    end
    loglikelihood(iter) = temp_loglikelihood;
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Print out loglikelihood
    disp(sprintf('EM iteration %d: log likelihood: %f', ...
        iter, loglikelihood(iter)));
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
    
    % Check for overfitting by decreasing loglikelihood
    if iter > 1
        if loglikelihood(iter) < loglikelihood(iter-1)
            break;
        end
    end
    
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
