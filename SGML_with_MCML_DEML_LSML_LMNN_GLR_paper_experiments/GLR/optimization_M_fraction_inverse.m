function [M_fraction_inv] = optimization_M_fraction_inverse(M,nv,n_feature,BCD,remaining_idx)

% Step 1: calculating the Matrix of Minors
% Step 2: turn that into the Matrix of Cofactors
% Step 3: the Adjugate
% Step 4: multiply that by 1/Determinant

sign_mask=ones(1,n_feature);
sign_mask(2:2:end)=-sign_mask(2:2:end);

if nv==n_feature+(n_feature*(n_feature-1)/2) % full
    M_fraction_inv=inv(M);
elseif nv==n_feature+n_feature-1 % diagonal + one row/column of off-diagonals
    % Step 1: calculating the Matrix of Minors
    % Step 2: turn that into the Matrix of Cofactors
    % Step 3: the Adjugate
    % Step 4: multiply that by 1/Determinant
elseif nv==n_feature % diagonal
    % Step 1: calculating the Matrix of Minors
    MoM=zeros(n_feature,1);
    MoM0=zeros(1,n_feature);
    for i=1:n_feature
        ridx=1:n_feature;
        ridx(i)=[];
        MoM(i)=det(M(ridx,ridx)); % too slow...
        MoM0(i)=det(M(2:end,ridx)); % too slow...
    end
    % Step 2: turn that into the Matrix of Cofactors (no need)
    % Step 3: the Adjugate (no need)
    % Step 4: multiply that by 1/Determinant
    M_fraction_inv=MoM/sum(M(1,:).*MoM0.*sign_mask);
else % one row/column of off-diagonals
    % Step 1: calculating the Matrix of Minors
    % Step 2: turn that into the Matrix of Cofactors
    % Step 3: the Adjugate
    % Step 4: multiply that by 1/Determinant
end

end

