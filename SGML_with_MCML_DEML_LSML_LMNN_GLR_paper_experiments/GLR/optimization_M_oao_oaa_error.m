function [error_count_total_linear ] = ...
    optimization_M_oao_oaa_error(error_count_total_linear,...
    error_count_linear_fold_i)
%oao_oaa_error Summary of this function goes here
%   Detailed explanation goes here

error_count_total_linear = error_count_total_linear + error_count_linear_fold_i;


end

