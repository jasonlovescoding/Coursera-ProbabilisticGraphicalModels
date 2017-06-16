function submit(part)
  addpath('./lib');

  conf.assignmentKey = 'HRaFgotIEeaoKxJCmMZ6SQ';
  conf.itemName = 'Learning with Incomplete Data';

  conf.partArrays = { ...
    { ...
      'BwB8H', ...
      { '' }, ...
      '', ...
    }, ...
    { ...
      'CWFts', ...
      { '' }, ...
      '', ...
    }, ...
    { ...
      '0tRrx', ...
      { '' }, ...
      '', ...
    }, ...
    { ...
      'EgIZ9', ...
      { '' }, ...
      '', ...
    }, ...
    { ...
      'Ev05z', ...
      { '' }, ...
      '', ...
    }, ...
    { ...
      'zPvH5', ...
      { '' }, ...
      '', ...
    }, ...
    { ...
      'BGZlk', ...
      { '' }, ...
      '', ...
    }, ...
  };

  conf.output = @output;
  submitWithConfiguration(conf);

end

% specifies which parts are test parts
function result = isTest(partIdx)
  if (mod(partIdx, 2) == 0)
      result = true;
  else
      result = false;
  end
end

function out = output(partId, auxstring)

  if partId == 1 %EM_cluster

    load 'PA9SampleCases.mat';
    [P ll CP] = EM_cluster(exampleINPUT.t1a1, exampleINPUT.t1a2, exampleINPUT.t1a3, exampleINPUT.t1a4);
    tmp = [[P.c] [P.clg.sigma_x] [P.clg.sigma_y] [P.clg.sigma_angle] [ll'] [CP(:)']];
    out = '';
    for idx = 1:length(tmp)
        out = [out ' ' SerializeFloat(tmp(idx))];
    end

  elseif partId == 2 %EM_cluster

    load 'submit_input.mat';
    [P ll CP] = EM_cluster(INPUT.t1a1, INPUT.t1a2, INPUT.t1a3, INPUT.t1a4);
    tmp = [[P.c] [P.clg.sigma_x] [P.clg.sigma_y] [P.clg.sigma_angle] [ll'] [CP(:)']];
    out = '';
    for idx = 1:length(tmp)
        out = [out ' ' SerializeFloat(tmp(idx))];
    end

  elseif partId == 3 %EM_HMM

    load 'PA9SampleCases.mat';
    [P ll CP PP] = EM_HMM(exampleINPUT.t2a1, exampleINPUT.t2a2, exampleINPUT.t2a3, exampleINPUT.t2a4, exampleINPUT.t2a5, exampleINPUT.t2a6);
    tmp = [[P.c] [P.clg.sigma_x] [P.clg.sigma_y] [P.clg.sigma_angle] [ll'] [CP(:)'] [PP(:)']];
    out = '';
    for idx = 1:length(tmp)
        out = [out ' ' SerializeFloat(tmp(idx))];
    end

  elseif partId == 4 %EM_HMM

    load 'submit_input.mat';
    [P ll CP PP] = EM_HMM(INPUT.t2a1, INPUT.t2a2, INPUT.t2a3, INPUT.t2a4, INPUT.t2a5, INPUT.t2a6);
    tmp = [[P.c] [P.clg.sigma_x] [P.clg.sigma_y] [P.clg.sigma_angle] [ll'] [CP(:)'] [PP(:)']];
    out = '';
    for idx = 1:length(tmp)
        out = [out ' ' SerializeFloat(tmp(idx))];
    end

  elseif partId == 5 %RecognizeActions

    load 'PA9SampleCases.mat';
    [acc pl] = RecognizeActions(exampleINPUT.t3a1, exampleINPUT.t3a2, exampleINPUT.t3a3, exampleINPUT.t3a4);
    tmp = [[acc] [pl']];
    out = '';
    for idx = 1:length(tmp)
        out = [out ' ' SerializeFloat(tmp(idx))];
    end

  elseif partId == 6 %RecognizeActions

    load 'submit_input.mat';
    [acc pl] = RecognizeActions(INPUT.t3a1, INPUT.t3a2, INPUT.t3a3, INPUT.t3a4);
    tmp = [[acc] [pl']];
    out = '';
    for idx = 1:length(tmp)
        out = [out ' ' SerializeFloat(tmp(idx))];
    end

  elseif partId == 7 %RecognizeUnknownActions

  	if (~exist('Predictions.mat'))
  		warning('You have not run SavePrediction.m before submitting.');
  		out = '';
  		return;
  	end
  	load Predictions.mat;
  	v = yourPredictions(:);
  	v = [numel(v); v];
  	out = SerializeIntVector(v);

  end

  out = strtrim(out);

  % end of output function.

end


function out = SerializeFloat( x )
  out = sprintf('%.4f', x);
end

function out = SerializeIntVector(x)
  % Serializes an integer vector.
  numLines = length(x);
  lines = cell(numLines,1);
  for i=1:numLines
    lines{i} = sprintf('%d\n', x(i));
  end
  out = sprintf('%s', lines{:});

end

function out = SerializeFactorsFg(F)
% Serializes a factor struct array into the .fg format for libDAI
% http://cs.ru.nl/~jorism/libDAI/doc/fileformats.html
%
% To avoid incompatibilities with EOL markers, make sure you write the
% string to a file using the appropriate file type ('wt' for windows, 'w'
% for unix)

  lines = cell(5*numel(F) + 1, 1);

  lines{1} = sprintf('%d\n', numel(F));
  lineIdx = 2;
  for i = 1:numel(F)
    lines{lineIdx} = sprintf('\n%d\n', numel(F(i).var));
    lineIdx = lineIdx + 1;

    lines{lineIdx} = sprintf('%s\n', num2str(F(i).var(:)')); % ensure that we put in a row vector
    lineIdx = lineIdx + 1;

    lines{lineIdx} = sprintf('%s\n', num2str(F(i).card(:)')); % ensure that we put in a row vector
    lineIdx = lineIdx + 1;

    lines{lineIdx} = sprintf('%d\n', numel(F(i).val));
    lineIdx = lineIdx + 1;

    % Internal storage of factor vals is already in the same indexing order
    % as what libDAI expects, so we don't need to convert the indices.
    vals = [0:(numel(F(i).val) - 1); F(i).val(:)'];
    lines{lineIdx} = sprintf('%d %0.8g\n', vals);
    lineIdx = lineIdx + 1;
  end

  out = sprintf('%s', lines{:});

end


function out = SerializeMEUOptimization(meu, optdr)
  optdr = SortFactorVars(optdr);
  optdr_part = SerializeFactorsFg(optdr);
  out = sprintf('%s\n%.4f\n', optdr_part, meu);
end


function f = SortAllFactors(factors)

  for i = 1:length(factors)
    factors(i) = SortFactorVars(factors(i));
  end

  varMat = vertcat(factors(:).var);
  [unused, order] = sortrows(varMat);

  f = factors(order);

end

function G = SortFactorVars(F)

  [sortedVars, order] = sort(F.var);
  G.var = sortedVars;

  G.card = F.card(order);
  G.val = zeros(numel(F.val), 1);

  assignmentsInF = IndexToAssignment(1:numel(F.val), F.card);
  assignmentsInG = assignmentsInF(:,order);
  G.val(AssignmentToIndex(assignmentsInG, G.card)) = F.val;

end

