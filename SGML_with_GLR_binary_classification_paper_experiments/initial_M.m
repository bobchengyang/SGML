%=================================================================
% Signed Graph Metric Learing (SGML) via Gershgorin Disc Alignment
% **initialize M as a dense/sparse matrix
%
% author: Cheng Yang
% email me any questions: cheng.yang@ieee.org
% date: June 16th, 2020
% please kindly cite the paper: 
% ['Signed Graph Metric Learning via Gershgorin Disc Alignment', 
% Cheng Yang, Gene Cheung, Wei Hu, 
% https://128.84.21.199/abs/2006.08816]
%=================================================================
function [M] = initial_M(n,mode)

if mode==1
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

