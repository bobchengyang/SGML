function [ G1, G2 ] = optimization_M_set_step_NR( N, n, c, M, y, k, BCD, remaining_idx, s_k, zz )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% y=(x-x.').^2;
% [N,n]=size(feature);
% y=reshape(y,[N^2 1]);
% a=reshape(feature,[N 1 n]);
% c=reshape(a-permute(a,[2 1 3]),[N^2 n]);
e=exp(-sum(c*M.*c,2));
c=reshape(c', [n 1 N^2]);
if k==n+(n*(n-1)/2) && BCD==0 % full
    factor0=-reshape(c.*reshape(c,[1 n N^2]), [n^2 N^2])';
    G1=reshape(sum(e.*y.*factor0,1), [n n]);
    G1=[2*G1(zz);diag(G1)];
    G2=reshape(sum(e.*y.*factor0.*factor0,1), [n n]);
    G2=[2*G2(zz);diag(G2)];
elseif k==2*n-1 % one row/column of off-diagonals and diagonals
    factor0=-reshape([c(BCD,:,:).*c(remaining_idx,:,:);c.^2], [2*n-1 N^2])';
    G1=reshape(sum(e.*y.*factor0,1), [2*n-1 1]);
    G2=reshape(sum(e.*y.*factor0.*factor0,1), [2*n-1 1]);
elseif k==n % diagonals
    factor0=-reshape(c.^2, [n N^2])';
    G1=reshape(sum(e.*y.*factor0,1), [n 1]);
    G2=reshape(sum(e.*y.*factor0.*factor0,1), [n 1]);
else % one row/column of off-diagonals
    factor0=-reshape(c(BCD,:,:).*c(remaining_idx,:,:), [n-1 N^2])';
    G1=reshape(sum(e.*y.*factor0,1), [n-1 1]);
    G2=reshape(sum(e.*y.*factor0.*factor0,1), [n-1 1]);
end

if k==2*n-1 && BCD~=0 % one row/column of off-diagonals and diagonals
    G1(1:n-1)=G1(1:n-1).*2;
    G2(1:n-1)=G2(1:n-1).*2;
end

if k==n-1 % one row/column of off-diagonals
    G1=G1.*2;
    G2=G2.*2;
end

G1=G1.*s_k; % first derivative wrt gamma
G2=G2.*s_k.*s_k; % second derivative wrt gamma

end

