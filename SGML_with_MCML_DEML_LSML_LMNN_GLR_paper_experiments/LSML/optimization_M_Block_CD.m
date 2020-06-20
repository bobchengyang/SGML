function [ M,lr,lower_bounds ] = optimization_M_Block_CD(...
    S,...
    D,...
    n_feature,...
    lr,...
    m11,...
    M21,...
    M22_min_eig,...
    M,...
    BCD,...
    tol_set,...
    remaining_idx,...
    lower_bounds,...
    dia_idx)

tol = Inf;
counter = 0;

objective_previous = lsml_obj(M,S,D);

while tol > tol_set
    
    G1 = lsml_gradient(M,S,D,n_feature,n_feature-1,BCD,remaining_idx);
    
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
    
    objective_current = lsml_obj(M,S,D);
    
    if objective_current==0
        lower_bounds(BCD)=sum(abs(M21_updated));
        return
    end
    
    if objective_current>objective_previous
        lr=lr/2;%avoid any solutions that end up with a larger obj.    else
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