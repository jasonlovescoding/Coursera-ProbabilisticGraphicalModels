% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.  
  D = I.DecisionFactors;
  MEU = 0;
  OptimalDecisionRule = struct('var', [], 'card', [], 'val', []);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  % A decision rule for D assigns, for each joint assignment to D's parents, 
  % probability 1 to the best option from the EUF for that joint assignment 
  % to D's parents, and 0 otherwise.  Note that when D has no parents, it is
  % a degenerate case we can handle separately for convenience.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
Is = repmat(I, 1, length(I.UtilityFactors));
for i = 1:length(I.UtilityFactors)
    Is(i).UtilityFactors = [I.UtilityFactors(i)];
end

Fs = repmat(struct('var', [], 'card', [], 'val', []), 1, length(I.UtilityFactors));
for i = 1:length(I.UtilityFactors)
    Fs(i) = CalculateExpectedUtilityFactor(Is(i));
end

f = Fs(1);
if length(Fs) > 1
    f = FactorSum(Fs(1), Fs(2));
    for i = 3:length(Fs)
        f = FactorSum(f, Fs(i));
    end
end

EUF = struct('var', [], 'card', [], 'val', []);
EUF.var = D.var; 
EUF.card = D.card;
mapping = zeros(1, length(EUF.var));
for i = 1:length(EUF.var) % re-order EUF.var
    mapping(i) = find(f.var == EUF.var(i));
end

for idx = 1:prod(f.card) 
    assignment = IndexToAssignment(idx, f.card); 
    assignment = assignment(mapping);
    EUFidx = AssignmentToIndex(assignment, EUF.card);
    EUF.val(EUFidx) = f.val(idx);
end

OptimalDecisionRule = EUF;
for i = 1:2:prod(OptimalDecisionRule.card)
    if (OptimalDecisionRule.val(i) > OptimalDecisionRule.val(i+1))
        OptimalDecisionRule.val(i) = 1;
        OptimalDecisionRule.val(i+1) = 0;
    else
        OptimalDecisionRule.val(i) = 0;
        OptimalDecisionRule.val(i+1) = 1;
    end
end

MEUF = FactorProduct(EUF, OptimalDecisionRule);
for i = 1:length(MEUF.val)
    MEU = MEU + MEUF.val(i);
end
end
