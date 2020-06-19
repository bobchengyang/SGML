function [ G ] = PDcone_compute_gradient( N, n, c, M, y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% y=(x-x.').^2;
% [N,n]=size(feature);
% y=reshape(y,[N^2 1]);
% a=reshape(feature,[N 1 n]);
% c=reshape(a-permute(a,[2 1 3]),[N^2 n]);
e=exp(-sum(c*M.*c,2));
c=reshape(c', [n 1 N^2]);
d=-reshape(c.*reshape(c,[1 n N^2]), [n^2 N^2])';
G_core=e.*y.*d;
G=reshape(sum(G_core,1), [n n]);
%G2=sum(permute(reshape(G_core,[N^2 1 n^2]),[3 2 1]).*reshape(d',[1 n^2 N^2]),3);
end

