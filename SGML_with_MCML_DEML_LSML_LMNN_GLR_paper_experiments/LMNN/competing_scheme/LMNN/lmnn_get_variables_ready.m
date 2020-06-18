function [targets_ind,...
    same_label,...
    G] = lmnn_get_variables_ready(X,feature_N,N,labels)

% Initialize some variables
assert(length(labels) == N);
[lablist, ~, labels] = unique(labels);
K = length(lablist);
label_matrix = false(N, K);
label_matrix(sub2ind(size(label_matrix), (1:length(labels))', labels)) = true;
same_label = logical(double(label_matrix) * double(label_matrix'));

% Set learning parameters
mu = .5;                % weighting of pull and push terms
no_targets = 3;         % number of target neighbors

if N<no_targets
    no_targets=N;
end

% Select target neighbors
sum_X = sum(X .^ 2, 2);
DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));
DD(~same_label) = Inf; DD(1:N + 1:end) = Inf;
[~, targets_ind] = sort(DD, 2, 'ascend');
targets_ind = targets_ind(:,1:no_targets);

G=zeros(feature_N);
for i=1:no_targets
    G = G+(1 - mu) .* (X - X(targets_ind(:,i),:))' * (X - X(targets_ind(:,i),:));
end

end
