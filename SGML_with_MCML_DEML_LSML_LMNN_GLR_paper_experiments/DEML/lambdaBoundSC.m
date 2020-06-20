function [ lambda ] = lambdaBoundSC( A )
%lambdaBound: output the lower bound of the smallest negative eigenvalue
%using Haynsworth Inertia Additivity and Schur Complement 

[M,N] = size(A);
% D = diag(sum(A'));
% L = D-A; 
L=A;
% r = 6;
% epsilon = 0.5;      % selecting this epsilon is very tricky!!
% r =50;
% epsilon = 0.09;      % selecting this epsilon is very tricky!!
r =100;
epsilon = 0.5; 
%****** step 0: need to choose index of the two sets ******
if N >= r,
    done = 0;
    ind1 = [1:r];
    ind2 = [r+1:N];
else
    done = 1;
    [V,lam] = eig(L);
    lamMin = min(diag(lam));
    lambda = min(0, lamMin);
    return;
end

%****** step 1: find value to make L_{1,1} PSD ******
L11 = L(ind1,ind1);
[V,lam] = eig(L11);
lamMin = min(diag(lam));
kappa = min(0, lamMin - epsilon);
cL = L - kappa*eye(N);

%****** step 2: find value to make L_{1,1}'s SC PSD ******
cL11 = cL(ind1,ind1);
cL12 = cL(ind1,ind2);
cL22 = cL(ind2,ind2);

SC = cL22 - cL12'*cL11^(-1)*cL12;
if length(ind2) <= r,
    [V,lam] = eig(SC);
    lamMin = min(diag(lam));
    eta = min(0, lamMin);
    lambda = kappa + eta;
    return;
else
    eta = lambdaBoundSC(SC);
    lambda = kappa + eta;
    return;
end