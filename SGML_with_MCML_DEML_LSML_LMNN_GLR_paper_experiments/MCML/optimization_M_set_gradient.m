function [ G ] = optimization_M_set_gradient( N, n, c, M, y, k, BCD )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% y=(x-x.').^2;
% [N,n]=size(feature);
% y=reshape(y,[N^2 1]);
% a=reshape(feature,[N 1 n]);
% c=reshape(a-permute(a,[2 1 3]),[N^2 n]);
e=exp(-sum(c*M.*c,2));
c=reshape(c', [n 1 N^2]);
if k==n+(n*(n-1)/2) % full
    G=reshape(sum(e.*y.*-reshape(c.*reshape(c,[1 n N^2]), [n^2 N^2])',1), [n n]);
elseif k==2*n-1 % one row/column of off-diagonals and diagonals
    remaining_idx=1:n;
    remaining_idx(BCD)=[];
    G=reshape(sum(e.*y.*-reshape([2*c(BCD,:,:).*c(remaining_idx,:,:);c.^2], [2*n-1 N^2])',1), [2*n-1 1]);
elseif k==n % diagonals
    G=reshape(sum(e.*y.*-reshape(c.^2, [n N^2])',1), [n 1]);
else % one row/column of off-diagonals
    remaining_idx=1:n;
    remaining_idx(BCD)=[];
    G=reshape(sum(e.*y.*-reshape(2*c(BCD,:,:).*c(remaining_idx,:,:), [n-1 N^2])',1), [n-1 1]);
end
end

