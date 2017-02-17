% Copyright (C) Daphne Koller, Stanford University, 2012

function EU = SimpleCalcExpectedUtility(I)

  % Inputs: An influence diagram, I (as described in the writeup).
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return Value: the expected utility of I
  % Given a fully instantiated influence diagram with a single utility node and decision node,
  % calculate and return the expected utility.  Note - assumes that the decision rule for the 
  % decision node is fully assigned.

  % In this function, we assume there is only one utility node.
  F = [I.RandomFactors I.DecisionFactors];
  U = I.UtilityFactors(1);
  EU = 0;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vars = unique([F(:).var]); % get all the variables involved

for i = 1:length(U.var)
    vars(vars == U.var(i)) = []; % elimnate the variables in utility distribution
end
  
factors = VariableElimination([F U], vars); % the combined distributions 
if length(factors) > 1
    f = FactorProduct(factors(1), factors(2));  % D(a)
    for i = 3:length(factors)
        f = FactorProduct(f, factors(i));
    end
else
    f = factors(1);
end

for i = 1:length(f.val)
    EU = EU + f.val(i); % assume there is only one utility node
end
end
