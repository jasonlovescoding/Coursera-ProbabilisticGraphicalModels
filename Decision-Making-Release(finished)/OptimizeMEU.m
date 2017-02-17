% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  MEU = 0;
  OptimalDecisionRule = struct('var', [], 'card', [], 'val', []);
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  % 
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
OptimalDecisionRule = CalculateExpectedUtilityFactor(I);

% it is guaranteed that decision var is var(1) in CalculateExpectedUtilityFactor
for i = 1:2:prod(OptimalDecisionRule.card)
    if (OptimalDecisionRule.val(i) > OptimalDecisionRule.val(i+1))
        OptimalDecisionRule.val(i) = 1;
        OptimalDecisionRule.val(i+1) = 0;
    else
        OptimalDecisionRule.val(i) = 0;
        OptimalDecisionRule.val(i+1) = 1;
    end
end

I.DecisionFactors(1) = OptimalDecisionRule;
MEU = SimpleCalcExpectedUtility(I);
end
