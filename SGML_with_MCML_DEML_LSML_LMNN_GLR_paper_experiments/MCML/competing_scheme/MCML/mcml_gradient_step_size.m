function [dA] = mcml_gradient_step_size(A, X, P, n, d, zz, nv, BCD, remaining_idx, s_k)
%MCML_GRAD Computes MCML cost function and gradient
%
%   [C, dA] = mcml_grad(x, X, P)
%
% Computes MCML cost function and gradient.
%
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology


% Decode solution
%[n, d] = size(X);
%A = reshape(x, [d d]);

% Compute conditional probabilities for current solution

D = zeros(n, n);
for i=1:n
    diffX = bsxfun(@minus, X(i,:), X(i + 1:end,:));
    dist = sum((diffX * A) .* diffX, 2);
    D(i + 1:end, i) = dist;
    D(i, i + 1:end) = dist';
end
Q = exp(-D);
Q(1:n+1:end) = 0;
Q = bsxfun(@rdivide, Q, sum(Q, 2));
Q = max(Q, realmin);

% Compute cost function
%C = sum(P(:) .* log(P(:) ./ Q(:)));

% Compute gradient with respect to A
PQ = P - Q;

if nv==d+(d*(d-1)/2) && BCD==0 % full
    dA = zeros(d, d);
    
    for i=1:n
        diffX = bsxfun(@minus, X(i,:), X); % 1 x d - n x d =  n x d
        dA = dA + bsxfun(@times, PQ(i,:), diffX') * diffX; % 1 x n .* d x n * n x d = d x d
    end
    dA=[2*dA(zz);diag(dA)];
elseif nv==d+d-1 % diagonal + one row/column of off-diagonal
    dA = zeros(d+d-1, 1);
    for i=1:n
        diffX = bsxfun(@minus, X(i,:), X); % 1 x d - n x d =  n x d
        
        dA = dA + [2*(PQ(i,:).*diffX(:,BCD)' * diffX(:,remaining_idx))'; sum(bsxfun(@times, PQ(i,:), diffX') .* diffX',2)];
    end
elseif nv==d % diagonal
    dA = zeros(d, 1);
    for i=1:n
        diffX = bsxfun(@minus, X(i,:), X); % 1 x d - n x d =  n x d
        dA = dA + sum(bsxfun(@times, PQ(i,:), diffX') .* diffX',2); % sum(1 x n .* d x n .* d x n,2) = d x 1
    end
else % one row/column of off-diagonal
    dA = zeros(d-1, 1);
    for i=1:n
        diffX = bsxfun(@minus, X(i,:), X); % 1 x d - n x d =  n x d
        
        dA = dA + 2*(PQ(i,:).*diffX(:,BCD)' * diffX(:,remaining_idx))'; % (1 x n .* 1 x n * n x (d-1))' = (d-1) x 1
    end
end
dA=dA.*s_k;
end
