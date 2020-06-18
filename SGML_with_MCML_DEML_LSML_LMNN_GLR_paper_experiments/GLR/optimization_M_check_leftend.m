function [leftend_diff,...
    lower_bounds,...
    scaled_M,...
    scaled_factors] = optimization_M_check_leftend(M,...
    feature_N,...
    lobpcg_random_control,...
    rho)
%OPTIMIZATION_M_CHECK_LEFTEND Summary of this function goes here
%   Detailed explanation goes here
rng(lobpcg_random_control);
fe = ...
    optimization_M_lobpcg(randn(feature_N,1),M,1e-12,200);
scaling_matrix = diag(1./fe(:,1));
scaling_matrix_inv = diag(fe(:,1));
scaled_M = scaling_matrix * M * scaling_matrix_inv;
scaled_factors = scaling_matrix * ones(feature_N) * scaling_matrix_inv;

scaled_M_=scaled_M;
scaled_M_(logical(eye(feature_N)))=0;
sum_scale=sum(abs(scaled_M_),2);
lower_bounds = sum_scale+rho;

leftend=diag(M)-sum_scale;
leftend_diff=mean(leftend-mean(leftend));
end

