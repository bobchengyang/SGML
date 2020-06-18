function [dA1,dA2] = mcml_gradient_step_size_NR(A, X, P, n, d, zz, nv, BCD, remaining_idx, s_k)
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
D_1st_d = D;
for i=1:n
    diffX = bsxfun(@minus, X(i,:), X(i + 1:end,:)); % 1 x d .- (n-1) * d = (n-1) * d
    dist = sum((diffX * A) .* diffX, 2); % (n-1) * d -> (n-1) * 1
    dist_1st_d=sum(diffX .* diffX, 2); % (n-1) * d -> (n-1) * 1
    D(i + 1:end, i) = dist;
    D(i, i + 1:end) = dist';
    D_1st_d(i + 1:end, i) = dist_1st_d;
    D_1st_d(i, i + 1:end) = dist_1st_d';    
end
Q0 = exp(-D);
Q0(1:n+1:end) = 0;
Q = bsxfun(@rdivide, Q0, sum(Q0, 2));
Q = max(Q, realmin);

% Compute cost function
%C = sum(P(:) .* log(P(:) ./ Q(:)));

% Compute gradient with respect to A

PQ=P-Q;
PQ_2nd_d=Q.*D_1st_d-(Q0.*sum(Q0.*D_1st_d,2))./(sum(Q0,2).^2);

if nv==d+(d*(d-1)/2) && BCD==0 % full
    dA1 = zeros(d, d);
    dA2=dA1;
    
    for i=1:n
        diffX = bsxfun(@minus, X(i,:), X); % 1 x d - n x d =  n x d
        dA1 = dA1 + bsxfun(@times, PQ(i,:), diffX') * diffX; % 1 x n .* d x n * n x d = d x d
        dA2 = dA2 + bsxfun(@times, PQ_2nd_d(i,:), diffX') * diffX; % 1 x n .* d x n * n x d = d x d
    end
    dA1=[2*dA1(zz);diag(dA1)];
    dA2=[2*dA2(zz);diag(dA2)];
elseif nv==d+d-1 % diagonal + one row/column of off-diagonal
    dA1 = zeros(d+d-1, 1);
    dA2=dA1;
    for i=1:n
        diffX = bsxfun(@minus, X(i,:), X); % 1 x d - n x d =  n x d
        dA1 = dA1 + [(PQ(i,:).*diffX(:,BCD)' * diffX(:,remaining_idx))'; sum(bsxfun(@times, PQ(i,:), diffX') .* diffX',2)];
        dA2 = dA2 + [(PQ_2nd_d(i,:).*diffX(:,BCD)' * diffX(:,remaining_idx))'; sum(bsxfun(@times, PQ_2nd_d(i,:), diffX') .* diffX',2)];
    end
elseif nv==d % diagonal
    dA1 = zeros(d, 1);
    dA2=dA1;
    for i=1:n
        diffX = bsxfun(@minus, X(i,:), X); % 1 x d - n x d =  n x d
        dA1 = dA1 + sum(bsxfun(@times, PQ(i,:), diffX') .* diffX',2); % sum(1 x n .* d x n .* d x n,2) = d x 1
        dA2 = dA2 + sum(bsxfun(@times, PQ_2nd_d(i,:), diffX') .* diffX',2); % sum(1 x n .* d x n .* d x n,2) = d x 1
    end
else % one row/column of off-diagonal
    dA1 = zeros(d-1, 1);
    dA2=dA1;
    for i=1:n
        diffX = bsxfun(@minus, X(i,:), X); % 1 x d - n x d =  n x d
        dA1 = dA1 + (PQ(i,:).*diffX(:,BCD)' * diffX(:,remaining_idx))'; % (1 x n .* 1 x n * n x (d-1))' = (d-1) x 1
        dA2 = dA2 + (PQ_2nd_d(i,:).*diffX(:,BCD)' * diffX(:,remaining_idx))'; % (1 x n .* 1 x n * n x (d-1))' = (d-1) x 1
    end
end

if nv==d+d-1 && BCD~=0
    dA1(1:d-1)=dA1(1:d-1).*2;
    dA2(1:d-1)=dA2(1:d-1).*2;
elseif nv==d-1
    dA1=dA1.*2;
    dA2=dA2.*2;
end

dA1=dA1.*s_k;
dA2=dA2.*s_k.*s_k;
end
