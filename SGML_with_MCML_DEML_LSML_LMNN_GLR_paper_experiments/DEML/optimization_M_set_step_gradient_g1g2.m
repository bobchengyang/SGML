function [ G,G2 ] = optimization_M_set_step_gradient_g1g2( feature, M, x, s_k, BCD )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

y=(x-x.').^2;
[N,n]=size(feature);
y=reshape(y,[N^2 1]);
a=reshape(feature,[N 1 n]);
c=reshape(a-permute(a,[2 1 3]),[N^2 n]);
e=exp(-sum(c*M.*c,2));
c=reshape(c', [n 1 N^2]);
d=-reshape(c.*reshape(c,[1 n N^2]), [n^2 N^2])';
G_core=e.*y.*d;
G=reshape(sum(G_core,1), [n n]);

s_k_mask=zeros(n);
if length(s_k)==n+(n*(n-1)/2)
    current_number=0;
    for i=1:n-1
        s_k_mask(i,i+1:end)=s_k(current_number+1:current_number+n-i)';
        s_k_mask(i+1:end,i)=s_k_mask(i,i+1:end);
        current_number=current_number+n-i;
    end
    s_k_mask(logical(eye(n)))=s_k(current_number+1:end);
else
    remaining_idx=1:n;
    remaining_idx(BCD)=[];
    s_k_mask(BCD,remaining_idx)=s_k(1:n-1);
    s_k_mask(remaining_idx,BCD)=s_k_mask(BCD,remaining_idx);
    s_k_mask(logical(eye(n)))=s_k(n-1+1:end);
end

G=G.*s_k_mask;

G2=reshape(sum(G_core.*d,1), [n n]).*s_k_mask;
end

