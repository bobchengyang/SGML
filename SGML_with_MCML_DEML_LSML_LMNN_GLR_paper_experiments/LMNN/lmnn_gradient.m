function [G] = lmnn_gradient(G,targets_ind,...
    same_label,...
    M,...
    X,...
    N,...
    n,...
    zz,...
    BCD,...
    remaining_idx,...
    nv)

% Set learning parameters
mu = .5;                % weighting of pull and push terms
no_targets = 3;         % number of target neighbors

if N<no_targets
    no_targets=N;
end

% Compute pulling term between target neigbhors to initialize gradient
slack = zeros(N, N, no_targets);
% if nv==n+(n*(n-1)/2) && BCD==0% full
%     G=zeros(n);
%     for i=1:no_targets
%         G = G+(1 - mu) .* (X - X(targets_ind(:,i),:))' * (X - X(targets_ind(:,i),:));
%     end
% elseif nv==n+n-1 % diagonals + one row/column of off-diagonals
%     G=zeros(1,nv);
%     for i=1:no_targets
%         G = G+(1 - mu) .* [2.*(X(:,BCD) - X(targets_ind(:,i),BCD))' * (X(:,remaining_idx) - X(targets_ind(:,i),remaining_idx)) ...
%             sum((X - X(targets_ind(:,i),:))'.^2,2)'];
%     end
% elseif nv==n % diagonals
%     G=zeros(1,nv);
%     for i=1:no_targets
%         G = G+(1 - mu) .* sum((X - X(targets_ind(:,i),:))'.^2,2)';       
%     end
% else % one row/column of off-diagonals
%     G=zeros(1,nv);
%     for i=1:no_targets
%         G = G+(1 - mu) .* 2 .* (X(:,BCD) - X(targets_ind(:,i),BCD))' * (X(:,remaining_idx) - X(targets_ind(:,i),remaining_idx));
%     end
% end

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

if nv==n+(n*(n-1)/2) && BCD==0% full
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
    %G=[2*G(zz);diag(G)];
elseif nv==n+n-1 % diagonals + one row/column of off-diagonals
    for i=1:no_targets
        
        % Add terms for new violations
        [r, c] = find(slack(:,:,i) > 0 & old_slack(:,:,i) == 0);
        G = G + [2 .* mu .* ((X(r,BCD) - X(targets_ind(r, i),BCD))' * ...
            (X(r,remaining_idx) - X(targets_ind(r, i),remaining_idx)) - ...
            (X(r,BCD) - X(c,BCD))' * (X(r,remaining_idx) - X(c,remaining_idx)))...
            mu .* (sum((X(r,:) - X(targets_ind(r, i),:))'.^2,2)' - ...
            sum((X(r,:) - X(c,:))'.^2,2)')];
        
        % Remove terms for resolved violations
        [r, c] = find(slack(:,:,i) == 0 & old_slack(:,:,i) > 0);
        G = G - [2 .* mu .* ((X(r,BCD) - X(targets_ind(r, i),BCD))' * ...
            (X(r,remaining_idx) - X(targets_ind(r, i),remaining_idx)) - ...
            (X(r,BCD) - X(c,BCD))' * (X(r,remaining_idx) - X(c,remaining_idx)))...
            mu .* (sum((X(r,:) - X(targets_ind(r, i),:))'.^2,2)' - ...
            sum((X(r,:) - X(c,:))'.^2,2)')];
        
    end
elseif nv==n % diagonals
    for i=1:no_targets
        
        % Add terms for new violations
        [r, c] = find(slack(:,:,i) > 0 & old_slack(:,:,i) == 0);
        G = G + mu .* (sum((X(r,:) - X(targets_ind(r, i),:))'.^2,2)' - ...
            sum((X(r,:) - X(c,:))'.^2,2)');
        
        % Remove terms for resolved violations
        [r, c] = find(slack(:,:,i) == 0 & old_slack(:,:,i) > 0);
        G = G - mu .* (sum((X(r,:) - X(targets_ind(r, i),:))'.^2,2)' - ...
            sum((X(r,:) - X(c,:))'.^2,2)');
        
    end
else % one row/column of off-diagonals
    for i=1:no_targets
        
        % Add terms for new violations
        [r, c] = find(slack(:,:,i) > 0 & old_slack(:,:,i) == 0);
        G = G + 2 .* mu .* ((X(r,BCD) - X(targets_ind(r, i),BCD))' * ...
            (X(r,remaining_idx) - X(targets_ind(r, i),remaining_idx)) - ...
            (X(r,BCD) - X(c,BCD))' * (X(r,remaining_idx) - X(c,remaining_idx)));
        
        % Remove terms for resolved violations
        [r, c] = find(slack(:,:,i) == 0 & old_slack(:,:,i) > 0);
        G = G - 2 .* mu .* ((X(r,BCD) - X(targets_ind(r, i),BCD))' * ...
            (X(r,remaining_idx) - X(targets_ind(r, i),remaining_idx)) - ...
            (X(r,BCD) - X(c,BCD))' * (X(r,remaining_idx) - X(c,remaining_idx)));
        
    end
end

end
