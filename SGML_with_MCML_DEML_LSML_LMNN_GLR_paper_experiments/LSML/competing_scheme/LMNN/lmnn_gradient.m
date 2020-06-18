function [G] = lmnn_gradient(M, X, labels)

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

% Compute pulling term between target neigbhors to initialize gradient
slack = zeros(N, N, no_targets);
G = zeros(D, D);
for i=1:no_targets
    G = G + (1 - mu) .* (X - X(targets_ind(:,i),:))' * (X - X(targets_ind(:,i),:));
end

% Compute pairwise distances under current metric
XM = X * M;
sum_X = sum(XM .* X, 2);
DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (XM * X')));

% Compute value of slack variables
old_slack = slack;
for i=1:no_targets
    slack(:,:,i) = ~same_label .* max(0, bsxfun(@minus, 1 + DD(sub2ind([N N], (1:N)', targets_ind(:,i))), DD));
end

% Perform gradient update
for i=1:no_targets
    
    % Add terms for new violations
    [r, c] = find(slack(:,:,i) > 0 & old_slack(:,:,i) == 0);
    G = G + mu .* ((X(r,:) - X(targets_ind(r, i),:))' * ...
        (X(r,:) - X(targets_ind(r, i),:)) - ...
        (X(r,:) - X(c,:))' * (X(r,:) - X(c,:)));
    
    % Remove terms for resolved violations
    [r, c] = find(slack(:,:,i) == 0 & old_slack(:,:,i) > 0);
    G = G - mu .* ((X(r,:) - X(targets_ind(r, i),:))' * ...
        (X(r,:) - X(targets_ind(r, i),:)) - ...
        (X(r,:) - X(c,:))' * (X(r,:) - X(c,:)));
end

end
