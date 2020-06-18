function [ M,...
    time_eig] = optimization_M_PDcone_y(...
    partial_sample,...
    n_feature,...
    c,...
    y,...
    M,...
    S_upper,...
    time_eig,...
    time_i,...
    lr)

    [ G1 ] = optimization_M_set_gradient_PD( partial_sample, n_feature, c, M, y);
%     L_constant = sqrt(max(eig(G2'*G2)));
%     M = M - (step_scale/L_constant) * G1;
    M = M - lr * G1;
    %t_PDcone=tic;
    teig=tic;
    [PDcone_v, PDcone_d] = eig(M); % eigen-decomposition of M
    time_eig(time_i)=time_eig(time_i)+toc(teig);
    
    ind=find(diag(PDcone_d)>0);
    M=PDcone_v(:,ind) * PDcone_d(ind,ind) * PDcone_v(:,ind)';
    %toc(t_PDcone);
    
    if sum(diag(M))>S_upper
        factor_for_diag = sum(diag(M))/S_upper;
        M = M/factor_for_diag;
    end

end

















