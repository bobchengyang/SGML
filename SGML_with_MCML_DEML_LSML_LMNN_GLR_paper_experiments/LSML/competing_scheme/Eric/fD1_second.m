function [fd_1st_d,fd_2nd_d] = fD1_second(D, A, d, nv, BCD, remaining_idx,length_D)

% ---------------------------------------------------------------------------
% the gradient of the dissimilarity constraint function w.r.t. A
%
% for example, let distance by L1 norm:
% f = f(\sum_{ij \in D} \sqrt{(x_i-x_j)A(x_i-x_j)'})
% df/dA_{kl} = f'* d(\sum_{ij \in D} \sqrt{(x_i-x_j)^k*(x_i-x_j)^l})/dA_{kl}
%
% note that d_ij*A*d_ij' = tr(d_ij*A*d_ij') = tr(d_ij'*d_ij*A)
% so, d(d_ij*A*d_ij')/dA = d_ij'*d_ij
%     df/dA = f'(\sum_{ij \in D} \sqrt{tr(d_ij'*d_ij*A)})
%             * 0.5*(\sum_{ij \in D} (1/sqrt{tr(d_ij'*d_ij*A)})*(d_ij'*d_ij))
% ---------------------------------------------------------------------------
     
%length_D = size(D,1);

core=sum(D*A.*D,2).^(1/2); % length_D x 1
core=reshape(core, [1 1 length_D]); % 1 x 1 x length_D 
D_vec=reshape(D',[d 1 length_D]); % d x 1 x length_D

if nv==d+(d*(d-1)/2) % full
    D_vec=D_vec.*permute(D_vec,[2 1 3]); % d x d x length_D
    %fd_1st_d=0.5*(D'*D)/core; % d x d
elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
    D_vec=[D_vec(BCD,:,:).*D_vec(remaining_idx,:,:);D_vec.*D_vec]; % (d+d-1) x 1 x length_D
    %fd_1st_d=0.5*[sum(D(:,BCD)'.*D(:,remaining_idx)',2);sum(D'.^2,2)]/core; % (d + d - 1) x 1
elseif nv==d % diagonals
    D_vec=D_vec.*D_vec; % d x 1 x length_D
    %fd_1st_d=0.5*sum(D'.^2,2)/core; % d x 1
else % one row/column of off-diagonals
    D_vec=D_vec(BCD,:,:).*D_vec(remaining_idx,:,:); % (d-1) x 1 x length_D
    %fd_1st_d=0.5*sum(D(:,BCD)'.*D(:,remaining_idx)',2)/core; % (d - 1) x 1
end

fd_1st_d=0.5*sum(D_vec./core,3);
fd_2nd_d=-(0.5^2)*sum(D_vec.*D_vec./(core.^3),3);

% sum_dist = 0.000001; %sum_deri =  zeros(d); 
% 
% if nv==d+(d*(d-1)/2) % full
%     sum_deri=zeros(d); % d x d
%     sum_deri_second=zeros(d);
% elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
%     sum_deri=zeros(d+d-1,1); % (d + d - 1) x 1
%     sum_deri_second=zeros(d+d-1,1);
% elseif nv==d % diagonals
%     sum_deri=zeros(d,1); % d x 1
%     sum_deri_second=zeros(d,1);
% else % one row/column of off-diagonals
%     sum_deri=zeros(d-1,1); % (d - 1) x 1
%     sum_deri_second=zeros(d-1,1);
% end
% 
% for i = 1:N
%   for j= i+1:N     % count each pair once
%     if D(i,j) == 1
%       d_ij = X(i,:) - X(j,:);
%       [~, deri_d_ij, deri2_d_ij] = distance1(A, d_ij, nv, BCD, remaining_idx, d);
%       %sum_dist = sum_dist +  dist_ij;
%       sum_deri = sum_deri + deri_d_ij;
%       sum_deri_second = sum_deri_second + deri2_d_ij;
%     end  
%   end
% end
% %sum_dist
% % fd_1st_d = dgF2(sum_dist)*sum_deri;
% fd_1st_d = sum_deri;
% fd_2nd_d = sum_deri_second;
% 
% % ------------------------------------------------
% 
% 
% % ___________derivative of cover function 1_________
% function z = dgF1(y)
% z = 1;
% 
% % ___________derivative of cover function 2_________
% function z = dgF2(y)
% z = 1/y;
% 
% 
% 
% function [dist_ij, deri_d_ij, deri2_d_ij] = distance1(A, d_ij, nv, BCD, remaining_idx, d)
% % distance and derivative of distance using distance1: distance(d) = L1
% fudge = 0.000001;  % regularizes derivates a little
% 
% M_ij0 = d_ij'*d_ij;
% 
% if nv==d+(d*(d-1)/2) % full
%     M_ij=M_ij0; % d x d
% elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
%     M_ij=[d_ij(BCD).*d_ij(remaining_idx)' ; d_ij'.*d_ij']; % (d + d - 1) x 1
% elseif nv==d % diagonals
%     M_ij=d_ij'.*d_ij'; % d x 1
% else % one row/column of off-diagonals
%     M_ij=d_ij(BCD).*d_ij(remaining_idx)'; % (d - 1) x 1
% end
%    
%       dist_ij = sqrt(trace(M_ij0*A));
% 
%       % derivative of dist_ij w.r.t. A
%       deri_d_ij = 0.5*M_ij/(dist_ij+fudge); 
%       deri2_d_ij = -0.5^2*M_ij.*M_ij/(dist_ij^3+fudge); 
% 
% 
% function [dist_ij, deri_d_ij] = distance2(A, d_ij)
% % distance and derivative of distance using distance2: distance(d) = sqrt(L1)
% fudge = 0.000001;  % regularizes derivates a little
% 
%       M_ij = d_ij'*d_ij;
%       L2 = trace(M_ij*A);           % L2 norm
%       dist_ij = sqrt(sqrt(L2));
% 
%       % derivative of dist_ij w.r.t. A
%       deri_d_ij = 0.25*M_ij/(L2^(3/4)+fudge); 
% 
% 
% function [dist_ij, deri_d_ij] = distance3(A, d_ij)
% % distance and derivative of distance using distance3: 1-exp(-\beta*L1)
% fudge = 0.000001;  % regularizes derivates a little
% 
%       beta = 0.5;
%       M_ij = d_ij'*d_ij;
%       L1 = sqrt(trace(M_ij*A));
%       dist_ij = 1 - exp(-beta*L1);
% 
%       % derivative of dist_ij w.r.t. A
%       deri_d_ij = 0.5*beta*exp(-beta*L1)*M_ij/(L1+fudge);
% 
