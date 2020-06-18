%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **compute objective funtion's gradient
%
% author: Cheng Yang
% email me any questions: cheng.yang@ieee.org
% date: June 16th, 2020
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [ G ] = compute_gradient( N, n, c, M, y, k, BCD, remaining_idx )
e=exp(-sum(c*M.*c,2));
c=reshape(c', [n 1 N^2]);
if k==n+(n*(n-1)/2) && BCD==0% full
    G=reshape(sum(e.*y.*-reshape(c.*reshape(c,[1 n N^2]), [n^2 N^2])',1), [n n]);
elseif k==2*n-1 % one row/column of off-diagonals and diagonals
    G=reshape(sum(e.*y.*-reshape([2*c(BCD,:,:).*c(remaining_idx,:,:);c.^2], [2*n-1 N^2])',1), [2*n-1 1]);
elseif k==n % diagonals
    G=reshape(sum(e.*y.*-reshape(c.^2, [n N^2])',1), [n 1]);
else % one row/column of off-diagonals
    G=reshape(sum(e.*y.*-reshape(2*c(BCD,:,:).*c(remaining_idx,:,:), [n-1 N^2])',1), [n-1 1]);
end
end

