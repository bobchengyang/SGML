function [ M_lc ] = optimization_M_PDcone_y(M,...
    S_upper,...
    lr,...
    G1)

    M_lc = M - lr * G1;
    
    [PDcone_v, PDcone_d] = eig(M_lc); % eigen-decomposition of M
    ind=find(diag(PDcone_d)>0);
    M_lc=PDcone_v(:,ind) * PDcone_d(ind,ind) * PDcone_v(:,ind)';
    if sum(diag(M_lc))>S_upper
        factor_for_diag = sum(diag(M_lc))/S_upper;
        M_lc = M_lc/factor_for_diag;
    end
    
    M_lc=(M_lc+M_lc')/2;
end

















