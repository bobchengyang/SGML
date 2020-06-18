function [ G,G2 ] = optimization_M_set_wei_g1g2( feature, M, x, o, BCD )
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

if o==n
    pick_idx=eye(n);
    d=d(:,logical(pick_idx(:))');
    G_core=e.*y.*d;
    G=sum(G_core,1)';
    G2=sum(permute(reshape(G_core,[N^2 1 o]),[3 2 1]).*reshape(d',[1 o N^2]),3);
else
    remaining_idx=1:n;
    remaining_idx(BCD)=[];
    pick_idx=zeros(n);
    pick_idx(BCD,remaining_idx)=1;
    pick_idx(remaining_idx,BCD)=pick_idx(BCD,remaining_idx);
    d=d(:,logical(pick_idx(:))');
    G_core=e.*y.*d;
    G=sum(G_core,1)';
    Gm=zeros(n);
    Gm(logical(pick_idx))=G;
    Gm=Gm+Gm';
    G=Gm(BCD,remaining_idx)';
    G2=sum(permute(reshape(G_core,[N^2 1 2*o]),[3 2 1]).*reshape(d',[1 2*o N^2]),3);
end

end

