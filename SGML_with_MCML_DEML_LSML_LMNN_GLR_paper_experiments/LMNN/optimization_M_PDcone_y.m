function [ M,...
    time_eig] = optimization_M_PDcone_y(...
    partial_sample,...
    n_feature,...
    targets_ind,...
    same_label,...
    M,...
    S_upper,...
    time_eig,...
    time_i,...
    lr,...
    partial_feature,...
    G)

[G1] = lmnn_gradient_PD(G,targets_ind,...
    same_label,...
    M,...
    partial_feature,...
    partial_sample);

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