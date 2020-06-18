function [ M,lr,lower_bounds ] = optimization_M_Block_CD(...
    partial_sample,...
    n_feature,...
    targets_ind,...
    same_label,...
    lr,...
    m11,...
    M21,...
    M22_min_eig,...
    M,...
    BCD,...
    tol_set,...
    remaining_idx,...
    lower_bounds,...
    partial_feature,...
    G,...
    dia_idx)


tol = Inf;
counter = 0;

objective_previous = lmnn_obj(targets_ind,same_label,M,partial_feature,partial_sample);

while tol > tol_set
    
    [G1] = lmnn_gradient(2*G(BCD,remaining_idx),targets_ind,...
    same_label,...
    M,...
    partial_feature,...
    partial_sample,...
    n_feature,...
    0,...
    BCD,...
    remaining_idx,...
    n_feature-1);
    G1=G1';
    if counter == 0
        M21_step = M21 - lr * G1;
    else
        M21_step = M21_updated - lr * G1;
    end
    
    if norm(M21_step) < sqrt(M22_min_eig*m11)
        M21_updated = M21_step;
    else
        M21_updated = (M21_step/norm(M21_step)) * sqrt(M22_min_eig*m11);
    end
    
    %====check lower bounds=====
    M_temp=M;
    M_temp(BCD,remaining_idx)=real(M21_updated');
    M_temp(remaining_idx,BCD)=M_temp(BCD,remaining_idx);
    lower_bounds_temp=M_temp;
    lower_bounds_temp(dia_idx)=0;
    lower_bounds_temp=sum(abs(lower_bounds_temp),2);
    if sum(lower_bounds_temp)>n_feature % invalid result, not going to update M
        return
    end
    %===========================
    
    M(BCD,remaining_idx)=real(M21_updated');
    M(remaining_idx,BCD)=M(BCD,remaining_idx);
    
    lower_bounds=M;
    lower_bounds(dia_idx)=0;
    lower_bounds=sum(abs(lower_bounds),2);
    
    objective_current = lmnn_obj(targets_ind,same_label,M,partial_feature,partial_sample);
    
    if objective_current>objective_previous
        lr=lr/2;
    else
        lr=lr*(1+1e-2);
    end
    tol = norm(objective_current - objective_previous);
    counter = counter + 1;
    objective_previous = objective_current;
    if counter>1e3 % not converged
        break
    end
end

end