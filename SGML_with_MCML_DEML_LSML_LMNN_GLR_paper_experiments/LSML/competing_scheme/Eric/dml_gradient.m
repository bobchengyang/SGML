function [G_eric] = dml_gradient(A, D, d, nv, BCD, remaining_idx, length_D)

%length_D = size(D,1);

core=sum(D*A.*D,2).^(1/2); % length_D x 1
core=reshape(core, [1 1 length_D]); % 1 x 1 x length_D 
D_vec=reshape(D',[d 1 length_D]); % d x 1 x length_D

if nv==d+(d*(d-1)/2) % full
    D_vec=D_vec.*permute(D_vec,[2 1 3]); % d x d x length_D
elseif nv==d+d-1 % diagonals + one row/column of off-diagonals
    D_vec=[D_vec(BCD,:,:).*D_vec(remaining_idx,:,:);D_vec.*D_vec]; % (d+d-1) x 1 x length_D
elseif nv==d % diagonals
    D_vec=D_vec.*D_vec; % d x 1 x length_D
else % one row/column of off-diagonals
    D_vec=D_vec(BCD,:,:).*D_vec(remaining_idx,:,:); % (d-1) x 1 x length_D
end

G_eric=0.5*sum(D_vec./core,3);

if nv==d+d-1 % diagonals + one row/column of off-diagonals
    G_eric(1:d-1)=G_eric(1:d-1).*2;
end

if nv==d-1 % one row/column of off-diagonals
    G_eric=G_eric.*2;
end

end

