function [error_rate_linear] = optimization_M_net_error(test_count_total, ...
    error_count_total_linear)
%NET_ERROR Summary of this function goes here
%   Detailed explanation goes here

   error_rate_linear = error_count_total_linear/test_count_total;

end

