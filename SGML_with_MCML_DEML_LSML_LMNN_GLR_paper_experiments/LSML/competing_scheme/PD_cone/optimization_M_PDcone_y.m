function [ M,...
    time_eig] = optimization_M_PDcone_y(...
    S,...
    D,...
    M,...
    S_upper,...
    time_eig,...
    time_i,...
    lr,...
    n_feature,...
    nv)

    G1 = lsml_gradient(M,S,D,n_feature,nv,0,0);
%     L_constant = sqrt(max(eig(G2'*G2)));
%     M = M - (step_scale/L_constant) * G1;
    M = M - lr * G1;
    %t_PDcone=tic;
    tic;
    [PDcone_v, PDcone_d] = eig(M); % eigen-decomposition of M
    time_eig(time_i)=time_eig(time_i)+toc;
    
    ind=find(diag(PDcone_d)>0);
    M=PDcone_v(:,ind) * PDcone_d(ind,ind) * PDcone_v(:,ind)';
    %toc(t_PDcone);
    
    if sum(diag(M))>S_upper
        factor_for_diag = sum(diag(M))/S_upper;
        M = M/factor_for_diag;
    end

end

















