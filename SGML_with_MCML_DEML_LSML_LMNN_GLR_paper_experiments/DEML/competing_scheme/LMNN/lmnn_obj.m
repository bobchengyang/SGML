function [C] = lmnn_obj(M, X, labels)

% Initialize some variables
[N, D] = size(X);
assert(length(labels) == N);
[lablist, ~, labels] = unique(labels);
K = length(lablist);
label_matrix = false(N, K);
label_matrix(sub2ind(size(label_matrix), (1:length(labels))', labels)) = true;
same_label = logical(double(label_matrix) * double(label_matrix'));

% Set learning parameters
mu = .5;                % weighting of pull and push terms
no_targets = 3;         % number of target neighbors

% Select target neighbors
sum_X = sum(X .^ 2, 2);
DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));
DD(~same_label) = Inf; DD(1:N + 1:end) = Inf;
[~, targets_ind] = sort(DD, 2, 'ascend');
targets_ind = targets_ind(:,1:no_targets);
targets = false(N, N);
targets(sub2ind([N N], vec(repmat((1:N)', [1 no_targets])), vec(targets_ind))) = true;

% Compute pulling term between target neigbhors to initialize gradient
slack = zeros(N, N, no_targets);

% Compute pairwise distances under current metric
XM = X * M;
sum_X = sum(XM .* X, 2);
DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (XM * X')));

% Compute value of slack variables
for i=1:no_targets
    slack(:,:,i) = ~same_label .* max(0, bsxfun(@minus, 1 + DD(sub2ind([N N], (1:N)', targets_ind(:,i))), DD));
end

% Compute value of cost function
C = (1 - mu) .* sum(DD(targets)) + ...  % push terms between target neighbors
    mu  .* sum(slack(:));          % pull terms between impostors

end

function x = vec(x)
x = x(:);
end
