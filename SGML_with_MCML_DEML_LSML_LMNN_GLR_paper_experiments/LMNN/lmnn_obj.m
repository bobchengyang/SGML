function [C] = lmnn_obj(targets_ind,...
    same_label,...
    M,...
    X,...
    N)

% Set learning parameters
mu = .5;                % weighting of pull and push terms
no_targets = 3;         % number of target neighbors

if N<no_targets
    no_targets=N;
end

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
