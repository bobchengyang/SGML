function [M_current_eigenvector0,scaled_M_,scaled_factors_] = optimization_M_scalars(M_temp_best,feature_N,bins_i,M_current_eigenvector0,bins_temp)
scaled_M_=diag(diag(M_temp_best));
scaled_factors_=eye(feature_N);

M_updated_current = M_temp_best(bins_temp==bins_i,bins_temp==bins_i);
temp_dim = size(M_updated_current,1);

% rng(lobpcg_random_control);
% tic;
% [M_current_eigenvector0,~] = ...
%     optimization_M_lobpcg(randn(size(M_updated_current,1),1),M_updated_current,1e-4,200);
% toc;
% tic;
if temp_dim>1
    [M_current_eigenvector0,~] = ...
       optimization_M_lobpcg(M_current_eigenvector0,M_updated_current,1e-4,200);
else
    M_current_eigenvector0=M_updated_current;
end
% toc;
%norm(M_temp_best*M_current_eigenvector-ld*M_current_eigenvector)
%scaling_matrix_0 = diag(1./M_current_eigenvector0(:,1));
%scaling_matrix_0_inv = diag(M_current_eigenvector0(:,1));
%scaled_M_0 = scaling_matrix_0 * M_updated_current * scaling_matrix_0_inv;
%scaled_factors_0 = scaling_matrix_0 * ones(temp_dim) * scaling_matrix_0_inv;

a=M_current_eigenvector0(:,1);

scaled_M_0 = (1./a) .* M_updated_current .* a';
scaled_factors_0 = (1./a) .* ones(temp_dim) .* a';

scaled_M_(bins_temp==bins_i,bins_temp==bins_i)=scaled_M_0;
scaled_factors_(bins_temp==bins_i,bins_temp==bins_i)=scaled_factors_0;
end

