function fs_1st_d = fS1(X, S, A, N, d, nv, BCD, remaining_idx)

% the gradient of the similarity constraint function w.r.t. A
% f = \sum_{ij}(x_i-x_j)A(x_i-x_j)' = \sum_{ij}d_ij*A*d_ij'
% df/dA = d(d_ij*A*d_ij')/dA
%
% note that d_ij*A*d_ij' = tr(d_ij*A*d_ij') = tr(d_ij'*d_ij*A)
% so, d(d_ij*A*d_ij')/dA = d_ij'*d_ij

%[N d] = size(X);

fudge = 0.000001;  % regularizes derivates a little if necessary
if nv==d+(d*(d-1)/2) % full
    fs_1st_d=zeros(d);
elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
    fs_1st_d=zeros(d+d-1,1);
elseif nv==d % diagonals
    fs_1st_d=zeros(d,1);
else % one row/column of off-diagonals
    fs_1st_d=zeros(d-1,1);
end

for i = 1:N
  for j= i+1:N
    if S(i,j) == 1
      d_ij = X(i,:) - X(j,:);
      % distij = d_ij * A * d_ij';         % distance between 'i' and 'j'    
      % full first derivative of the distance constraints

if nv==d+(d*(d-1)/2) % full
    fs_1st_d = fs_1st_d + d_ij'*d_ij; % d x d
elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
    fs_1st_d=fs_1st_d+[d_ij(BCD).*d_ij(remaining_idx)' ; d_ij'.*d_ij']; % (d + d - 1) x 1
elseif nv==d % diagonals
    fs_1st_d=fs_1st_d+d_ij'.*d_ij'; % d x 1
else % one row/column of off-diagonals
    fs_1st_d=fs_1st_d+d_ij(BCD).*d_ij(remaining_idx)'; % (d - 1) x 1
end

    end  
  end
end
