% Copyright (C) Daphne Koller, Stanford University, 2012

function EUF = CalculateExpectedUtilityFactor( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: A factor over the scope of the decision rule D from I that
  % gives the conditional utility given each assignment for D.var
  %
  % Note - We assume I has a single decision node and utility node.
  EUF = struct('var', [], 'card', [], 'val', []);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
F = I.RandomFactors;
D = I.DecisionFactors;
U = I.UtilityFactors;

vars = unique([F(:).var]); % get all the variables involved

vars = setdiff(vars, D.var); % elimnate the variables in decision distribution

factors = VariableElimination([F U], vars); % the combined distributions
if length(factors) > 1
    f = FactorProduct(factors(1), factors(2));  % D(a)
    for i = 3:length(factors)
        f = FactorProduct(f, factors(i));
    end
else
    f = factors(1);
end

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
end  
