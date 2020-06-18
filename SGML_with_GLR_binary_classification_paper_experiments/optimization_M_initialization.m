function [M] = optimization_M_initialization(n,mode)
%OPTIMIZATION_M_INITIALIZATION Summary of this function goes here
%   Detailed explanation goes here
if mode==0
    rng(0);
    ghost=abs(1/n*1e-2*randn(n));
    ghost=ghost+ghost';
    M=ones(n);
    m=ones(n);
    for i=2:n-1
        rng(i);
        M(logical(triu(M,i)))=M(logical(triu(M,i)))*-1;
    end
    N=M';
    M(logical(tril(m,-1)))=N(logical(tril(m,-1)));
    M=M.*ghost;
    M(logical(eye(n)))=1;
else
    M=eye(n);
    m=logical(ones(n));
    M(triu(m,1))=0.1;
    M(triu(m,2))=0;
    M(tril(m,-1))=0.1;
    M(tril(m,-2))=0;
    M=sparse(M);
end
end

