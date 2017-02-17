%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isMax
    for i = 1:N       % convert to log space
        P.cliqueList(i).val = log(P.cliqueList(i).val);
    end
    for t = 1:2*(N-1) % (N-1) edges; each edge visited twice
        [i, j] = GetNextCliques(P, MESSAGES);
        inputs = struct('var', [], 'card', [], 'val', []); % messages into clique(i)
        for k = 1:N
            if P.edges(i, k) && k~=j 
                inputs = FactorSum(inputs, MESSAGES(k, i));
            end
        end
        MESSAGES(i, j) = FactorSum(P.cliqueList(i), inputs);
        eliminateVars = setdiff(P.cliqueList(i).var, P.cliqueList(j).var);
        MESSAGES(i, j) = FactorMaxMarginalization(MESSAGES(i, j), eliminateVars);
    end
else
    for t = 1:2*(N-1) % (N-1) edges; each edge visited twice
        [i, j] = GetNextCliques(P, MESSAGES);
        inputs = struct('var', [], 'card', [], 'val', []); % messages into clique(i)
        for k = 1:N
            if P.edges(i, k) && k~=j 
                inputs = FactorProduct(inputs, MESSAGES(k, i));
            end
        end
        MESSAGES(i, j) = FactorProduct(P.cliqueList(i), inputs);
        eliminateVars = setdiff(P.cliqueList(i).var, P.cliqueList(j).var);
        MESSAGES(i, j) = FactorMarginalization(MESSAGES(i, j), eliminateVars);
        MESSAGES(i, j) = NormalizeFactorValues(MESSAGES(i, j));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isMax
    for i = 1:N
        for k = 1:N
            if P.edges(i, k) % sum all the neighbor inputs
                P.cliqueList(i) = FactorSum(P.cliqueList(i), MESSAGES(k, i));
            end
        end
        
    end
else
    for i = 1:N
        for k = 1:N
            if P.edges(i, k) % mupltiply all the neighbor inputs
                P.cliqueList(i) = FactorProduct(P.cliqueList(i), MESSAGES(k, i));
            end
        end
    end    
end

return
