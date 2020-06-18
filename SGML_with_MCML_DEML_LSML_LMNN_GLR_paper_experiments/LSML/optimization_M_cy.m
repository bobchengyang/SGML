function [c,y] = optimization_M_cy(feature,x)
%OPTIMIZATION_M_CY Summary of this function goes here
%   Detailed explanation goes here
y=(x-x.').^2;
[N,n]=size(feature);
y=reshape(y,[N^2 1]);
a=reshape(feature,[N 1 n]);
c=reshape(a-permute(a,[2 1 3]),[N^2 n]);
end

