% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over X_2 and X_3, which takes on a value of 1
    % if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.
    
    nll = 0;
    grad = zeros(size(theta));
    %%%
    % Your code here:
    % get the factors from the features
    factors = FeaturesToFactors(featureSet.features, theta, modelParams);
    
    % get and calibrate the clique tree
    P = CreateCliqueTree(factors);
    [P, logZ] = CliqueTreeCalibrate(P, 0);
    
    % get the indications from y and the features
    [indications, weightedIndications] = CalculateIndicators(y, featureSet.features, theta);
    
    % get the penalty term
    penalty = RegularizationPenalty(modelParams.lambda, theta);
    
    % calculate nll
    nll = logZ - sum(weightedIndications) + penalty;
    
    % get the expectation given theta
    [Etheta] = CalculateETheta(P, featureSet.numParams, featureSet.features);
    
    % get the imperical expectation given indications
    [Ed] = CalculateED(featureSet.features, featureSet.numParams, indications);
    
    % calculate gradient
    grad = Etheta - Ed + modelParams.lambda * theta;
end

% calculate the indiation of the indicator features given y
function [indications, weightedIndications] = CalculateIndicators(y, features, theta)
    indications = zeros(length(features), 1);
    weightedIndications = zeros(length(features), 1);
    for i = 1:length(features)
        if all(y(features(i).var)==features(i).assignment)
            indications(i) = 1;
            weightedIndications(i) = theta(features(i).paramIdx);
        end
    end
end

% calculate the regularization penalty
function [penalty] = RegularizationPenalty(lambda, theta)
    penalty = (lambda / 2.0) * sum(theta.^2);
end

% transform features to factors
function [factors] = FeaturesToFactors(features, theta, modelParams)
    factors = repmat(EmptyFactorStruct(), length(features), 1);
    for i = 1:length(features)
        % var is shared
        factors(i).var = features(i).var;
        % card depends on the number of hidden states
        factors(i).card = ones(1, length(features(i).var)) .* modelParams.numHiddenStates;
        % val should be in exponential space
        factors(i).val = ones(1, prod(factors(i).card));
        factors(i) = SetValueOfAssignment(factors(i), ...
            features(i).assignment, exp(theta(features(i).paramIdx)));
    end
end

% calculate the expectation of features given theta
function [Etheta] = CalculateETheta(P, numParams, features)
    Etheta = zeros(1, numParams);
    % for every feature, find its clique and calculate
    % the corresponding expectation from that clique
    for i = 1:length(features)
        thetaIdx = features(i).paramIdx;
        clique = 0;
        for j = 1:length(P.cliqueList)
            if all(ismember(features(i).var, P.cliqueList(j).var))
                clique = P.cliqueList(j);
                break;
            end
        end
        VarsToMarginalize = setdiff(clique.var, features(i).var);
        combinedFactor = FactorMarginalization(clique, VarsToMarginalize);
        idx = AssignmentToIndex(features(i).assignment, combinedFactor.card);
        Etheta(thetaIdx) = Etheta(thetaIdx) + ...
            combinedFactor.val(idx) / sum(combinedFactor.val);
    end
end

% calculate the imperical expectation given indications
function [Ed] = CalculateED(features, numParams, indications)
    Ed = zeros(1,numParams);
    for i = 1:length(features)
        Ed(features(i).paramIdx) = Ed(features(i).paramIdx) + indications(i);
    end
end