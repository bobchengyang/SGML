%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **computing the first eigenvector of block M and then scalars
%
% author: Cheng Yang
% date: June 16th, 2020
%=================================================================
function [M_current_eigenvector0,scaled_M_,scaled_factors_] = scalars(M_temp_best,feature_N,bins_i,M_current_eigenvector0,bins_temp)
scaled_M_=diag(diag(M_temp_best));
scaled_factors_=eye(feature_N);

M_updated_current = M_temp_best(bins_temp==bins_i,bins_temp==bins_i);
temp_dim = size(M_updated_current,1);

if temp_dim>1
    [M_current_eigenvector0,~] = ...
        lobpcg_fv(M_current_eigenvector0,M_updated_current,1e-4,200);
else
    M_current_eigenvector0=M_updated_current;
end

a=M_current_eigenvector0(:,1);

scaled_M_0 = (1./a) .* M_updated_current .* a';
scaled_factors_0 = (1./a) .* ones(temp_dim) .* a';

scaled_M_(bins_temp==bins_i,bins_temp==bins_i)=scaled_M_0;
scaled_factors_(bins_temp==bins_i,bins_temp==bins_i)=scaled_factors_0;
end

