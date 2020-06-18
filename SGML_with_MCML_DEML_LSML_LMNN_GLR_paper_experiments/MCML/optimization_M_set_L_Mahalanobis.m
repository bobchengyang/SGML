function [ L ] = optimization_M_set_L_Mahalanobis( N, c, M )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% [N,n]=size(feature);
% feature=reshape(feature,[N 1 n]);
% c=reshape(feature-permute(feature,[2 1 3]),[N^2 n]);
W=exp(-sum(c*M.*c,2));
W=reshape(W, [N N]);
W(logical(eye(N))) = 0;
L = diag(sum(W))-W;
end

